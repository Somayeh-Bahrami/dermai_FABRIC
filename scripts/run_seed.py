"""One FABRIC run: train one model with one training seed on the fixed split, then
evaluate all three axes on the same checkpoint and the same test images.

    python scripts/run_seed.py --check-split-only
    python scripts/run_seed.py --model efficientnet --seed 42
    python scripts/run_seed.py --model vit --seed 42

Writes results/final/<model>_s<seed>/:
    config_used.yaml, checkpoint_sha256.txt, test_image_ids.txt, training_log.csv,
    classification.json, faithfulness_{native,ig,random,center}.{csv,npz},
    localization_{native,ig,center}.csv (IoU, pointing game at 0 and 15 px),
    heatmaps_{native,ig}/ (a fixed qualitative subset unless --keep-all-heatmaps), summary.json

Model selection uses validation macro-F1 only; the test split is evaluated once,
on the selected checkpoint. Smoke test on CPU (no downloads):
    python scripts/run_seed.py --model vit --seed 42 --tiny-model --limit 8 --epochs-override 1 \
        --data-dir <synthetic tree> --out-root <scratch>
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dermai.attrollout import AttentionRollout  # noqa: E402
from dermai.config import Config  # noqa: E402
from dermai.data import CLASSES, LABEL_TO_INDEX, DataModule, load_metadata, split_metadata  # noqa: E402
from dermai.explain import gradcam_explainer, ig_explainer, rollout_explainer, write_heatmaps  # noqa: E402
from dermai.faithfulness import (  # noqa: E402
    DeletionInsertion,
    GaussianBlurSubstrate,
    MeanFillSubstrate,
    center_ordering,
    evaluate_orderings,
    heatmap_dir_ordering,
    macro_average,
    random_ordering,
    target_class_from_filename,
    write_results,
)
from dermai.gradcam import GradCAM  # noqa: E402
from dermai.integrated_gradients import IntegratedGradients  # noqa: E402
from dermai.localization import evaluate_protocol  # noqa: E402
from dermai.models import ModelFactory  # noqa: E402
from dermai.baselines import center_prior_heatmap  # noqa: E402
from dermai.trainer import Trainer  # noqa: E402
from dermai.utils import Timer, build_image_id_to_path, get_logger, pick_device, set_seed  # noqa: E402

logger = get_logger()

CONFIGS = {"efficientnet": ROOT / "configs" / "efficientnet.yaml", "vit": ROOT / "configs" / "vit.yaml"}
REFERENCE_IDS = ROOT / "results" / "localization_efficientnet_best.csv"
NUM_QUALITATIVE = 40  # heatmaps kept per explainer: the first 40 test ids in sorted order, same in every run
FAITHFULNESS = {"deletion_substrate": "blur", "insertion_substrate": "mean", "step_pixels": 512}
IG = {"steps": 32, "baseline": "zeros (normalized space)", "target": "predicted class"}


def reference_test_ids() -> list[str]:
    with REFERENCE_IDS.open() as handle:
        return sorted(row["image_id"] for row in csv.DictReader(handle))


def check_split(data_dir: Path, split_seed: int) -> bool:
    test_ids = sorted(split_metadata(load_metadata(data_dir), split_seed)["test"].image_id)
    reference = reference_test_ids()
    same = test_ids == reference
    print(f"split_seed {split_seed}: {len(test_ids)} test ids, reference {len(reference)}, identical: {same}")
    return same


# ----------------------------------------------------------------------
# Model construction (real: HF checkpoints; tiny: random init, CPU smoke test only)
# ----------------------------------------------------------------------
def build_processor(config: Config, tiny: bool):
    if not tiny:
        return ModelFactory.processor(config.model_id)
    from transformers import ViTImageProcessor
    return ViTImageProcessor(size={"height": 224, "width": 224},
                             image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])


def build_model(config: Config, tiny: bool):
    if not tiny:
        return ModelFactory.build(config.model_id, dropout=config.dropout,
                                  attention_dropout=config.attention_dropout,
                                  drop_connect_rate=config.drop_connect_rate)
    from transformers import AutoModelForImageClassification, EfficientNetConfig, ViTConfig
    labels = dict(num_labels=len(CLASSES), id2label=dict(enumerate(CLASSES)), label2id=LABEL_TO_INDEX)
    if "vit" in config.model_id:
        model_config = ViTConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                                 intermediate_size=128, image_size=224, patch_size=16, **labels)
    else:
        model_config = EfficientNetConfig(image_size=224, width_coefficient=1.0, depth_coefficient=1.0,
                                          hidden_dim=1280, dropout_rate=0.2, **labels)  # B0 shape, random weights
    model = AutoModelForImageClassification.from_config(model_config)
    # HF's random init (small-std weights, BatchNorm gammas near 0) collapses activations through a
    # deep CNN, giving constant outputs and all-zero heatmaps; standard init keeps the smoke test informative.
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        elif isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    return model


def load_eval_model(checkpoint_dir: Path, model_name: str, device: torch.device):
    from transformers import AutoModelForImageClassification
    kwargs = {"attn_implementation": "eager"} if model_name == "vit" else {}  # trap T4: rollout needs eager
    return AutoModelForImageClassification.from_pretrained(checkpoint_dir, **kwargs).to(device).eval()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    return result.stdout.strip() + ("-dirty" if dirty else "")


def find_mask_dir(data_dir: Path) -> Path:
    first = next(Path(data_dir).rglob("*_segmentation.png"), None)
    if first is None:
        raise FileNotFoundError(f"no *_segmentation.png masks under {data_dir}")
    return first.parent


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def faithfulness_summary(results) -> dict:
    deletion = np.array([r.deletion_auc for r in results])
    insertion = np.array([r.insertion_auc for r in results])
    targets = np.array([r.target_class for r in results])
    return {"n": len(results),
            "deletion_auc_mean": float(deletion.mean()), "insertion_auc_mean": float(insertion.mean()),
            "deletion_auc_macro": macro_average(deletion, targets),
            "insertion_auc_macro": macro_average(insertion, targets)}


def localization_summary(rows: list[dict]) -> dict:
    iou = np.array([r["iou"] for r in rows])
    return {"n": len(rows), "iou_mean": float(iou.mean()), "iou_sd": float(iou.std()),
            "pointing_tol0": float(np.mean([r["pointing_hit_tol0"] for r in rows])),
            "pointing_tol15": float(np.mean([r["pointing_hit_tol15"] for r in rows]))}


def prune_heatmaps(heatmap_dir: Path, keep_ids: set[str]) -> None:
    for path in heatmap_dir.glob("*.npy"):
        if path.stem.split("__")[0] not in keep_ids:
            path.unlink()


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(CONFIGS))
    parser.add_argument("--seed", type=int, help="training seed (42, 43, 44); the split stays at split_seed")
    parser.add_argument("--check-split-only", action="store_true", help="verify the 1,002 test ids and exit")
    parser.add_argument("--data-dir", type=Path, default=None, help="override config data_dir")
    parser.add_argument("--out-root", type=Path, default=ROOT / "results" / "final")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "outputs" / "final")
    parser.add_argument("--skip-train", action="store_true",
                        help="reuse an existing checkpoint in --checkpoint-root (e.g. after a session timeout)")
    parser.add_argument("--ig-internal-batch-size", type=int, default=32, help="lower on CUDA OOM")
    parser.add_argument("--keep-all-heatmaps", action="store_true")
    parser.add_argument("--device", default=None)
    smoke = parser.add_argument_group("smoke test only (never for paper numbers)")
    smoke.add_argument("--limit", type=int, default=None, help="cap every split at N images")
    smoke.add_argument("--epochs-override", type=int, default=None, help="epochs per phase")
    smoke.add_argument("--tiny-model", action="store_true", help="random-init models, no downloads")
    args = parser.parse_args()

    base = Config.from_yaml(CONFIGS[args.model or "efficientnet"])
    data_dir = args.data_dir or (ROOT / base.data_dir)
    if args.check_split_only:
        sys.exit(0 if check_split(data_dir, base.split_seed) else 1)
    if args.model is None or args.seed is None:
        parser.error("--model and --seed are required")
    is_smoke = args.limit is not None or args.epochs_override is not None or args.tiny_model

    run_name = f"{args.model}_s{args.seed}"
    out_dir = args.out_root / run_name
    checkpoint_root = args.checkpoint_root / run_name
    phases = base.phases
    if args.epochs_override is not None:
        phases = [dataclasses.replace(p, epochs=args.epochs_override) for p in phases]
    config = dataclasses.replace(base, seed=args.seed, data_dir=data_dir, output_dir=checkpoint_root,
                                 experiment_tag=run_name, phases=phases,
                                 num_workers=0 if is_smoke else base.num_workers)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    set_seed(config.seed)
    device = pick_device(args.device or config.device)
    logger.info("run %s  device %s  training seed %d  split seed %d%s", run_name, device.type,
                config.seed, config.split_seed, "  (SMOKE TEST)" if is_smoke else "")

    # Data: fixed split (trap T1), verified against the published test ids on real runs.
    processor = build_processor(config, args.tiny_model)
    data = DataModule(config.data_dir, processor, config.batch_size, config.num_workers, config.split_seed,
                      augment=config.augment)
    data.setup()
    if args.limit is not None:
        # Smoke test: keep every class in train/val (class weights need all 7), cap test at N images.
        per_class = args.limit // len(CLASSES) + 1
        data.splits = {name: (frame.head(args.limit) if name == "test" else frame.groupby("dx").head(per_class))
                       .reset_index(drop=True) for name, frame in data.splits.items()}
    test_ids = data.test_image_ids()
    if not is_smoke and test_ids != reference_test_ids():
        raise SystemExit("test split differs from results/localization_efficientnet_best.csv; aborting (trap T1)")
    (out_dir / "test_image_ids.txt").write_text("\n".join(test_ids) + "\n")

    # Train with validation-only selection, then evaluate the test split once.
    timer = Timer()
    model = build_model(config, args.tiny_model)
    trainer = Trainer(model, data, config, device)
    checkpoint_dir = trainer.checkpoint_dir
    if not args.skip_train:
        if (config.output_dir / "ablation_log.csv").exists():
            (config.output_dir / "ablation_log.csv").unlink()
        trainer.fit()
    test_metrics = trainer.test()
    timings["train_and_test_s"] = timer.elapsed()
    val_rows = [r for r in csv.DictReader((config.output_dir / "ablation_log.csv").open()) if r["phase"] != "test"]
    shutil.copy(config.output_dir / "ablation_log.csv", out_dir / "training_log.csv")
    best_val = max(val_rows, key=lambda r: float(r["val_macro_f1"])) if val_rows else {}

    weights = checkpoint_dir / "model.safetensors"
    (out_dir / "checkpoint_sha256.txt").write_text(f"{sha256(weights)}  {weights.relative_to(ROOT) if weights.is_relative_to(ROOT) else weights}\n")
    with (out_dir / "config_used.yaml").open("w") as handle:
        raw = dataclasses.asdict(config)
        raw["data_dir"], raw["output_dir"] = str(config.data_dir), str(config.output_dir)
        yaml.safe_dump(raw, handle, sort_keys=False)

    classification = {
        "selection": "best validation macro-F1 over all epochs of all phases",
        "val_macro_f1_selected": float(best_val["val_macro_f1"]) if best_val else None,
        "val_balanced_accuracy_selected": float(best_val["val_balanced_accuracy"]) if best_val else None,
        "selected_phase": best_val.get("phase"), "selected_epoch": best_val.get("epoch"),
        "test_macro_f1": test_metrics["macro_f1"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_per_class_f1": dict(zip(CLASSES, test_metrics["per_class_f1"])),
        "n_test": len(test_ids),
    }
    (out_dir / "classification.json").write_text(json.dumps(classification, indent=2))

    # Explanations from the selected checkpoint only.
    eval_model = load_eval_model(checkpoint_dir, args.model, device)
    loader = data.loader("test")
    native_dir, ig_dir = out_dir / "heatmaps_native", out_dir / "heatmaps_ig"
    for directory in (native_dir, ig_dir):
        shutil.rmtree(directory, ignore_errors=True)
    timer = Timer()
    if args.model == "efficientnet":
        with GradCAM(eval_model) as gradcam:
            write_heatmaps(gradcam_explainer(gradcam), loader, native_dir, device)
        native_name = "gradcam"
    else:
        write_heatmaps(rollout_explainer(AttentionRollout(eval_model)), loader, native_dir, device)
        native_name = "rollout"
    timings["native_heatmaps_s"] = timer.elapsed()
    timer = Timer()
    for parameter in eval_model.parameters():
        parameter.requires_grad_(False)
    ig = IntegratedGradients(eval_model, steps=IG["steps"], internal_batch_size=args.ig_internal_batch_size)
    write_heatmaps(ig_explainer(ig), loader, ig_dir, device)
    timings["ig_heatmaps_s"] = timer.elapsed()

    native_paths, ig_paths = build_image_id_to_path(native_dir), build_image_id_to_path(ig_dir)
    assert sorted(native_paths) == test_ids == sorted(ig_paths), "heatmaps do not cover the test split"

    # Faithfulness: every ordering on the same images, same target (the predicted class).
    timer = Timer()
    evaluator = DeletionInsertion(eval_model, device, deletion_substrate=GaussianBlurSubstrate(),
                                  insertion_substrate=MeanFillSubstrate(), step_pixels=FAITHFULNESS["step_pixels"])
    orderings = {"native": heatmap_dir_ordering(native_paths), "ig": heatmap_dir_ordering(ig_paths),
                 "random": random_ordering(), "center": center_ordering()}
    faith = evaluate_orderings(evaluator, loader, orderings,
                               target_for=lambda image_id: target_class_from_filename(native_paths[image_id].stem),
                               image_ids=set(test_ids), logger=logger)
    fractions = evaluator.step_fractions(224 * 224)
    for name, rows in faith.items():
        write_results(out_dir / f"faithfulness_{name}.csv", rows, fractions)
    timings["faithfulness_s"] = timer.elapsed()

    # Localization: IoU (top 20%) and pointing game at 0 and 15 px.
    mask_dir = find_mask_dir(config.data_dir)
    center = center_prior_heatmap()
    sources = {"native": lambda i: np.load(native_paths[i]), "ig": lambda i: np.load(ig_paths[i]),
               "center": lambda i: center}
    localization = {}
    for name, load in sources.items():
        rows = evaluate_protocol(((i, load(i)) for i in test_ids), mask_dir)
        write_csv(out_dir / f"localization_{name}.csv", rows)
        localization[name] = localization_summary(rows)

    if not args.keep_all_heatmaps:
        keep = set(test_ids[:NUM_QUALITATIVE])
        prune_heatmaps(native_dir, keep)
        prune_heatmaps(ig_dir, keep)

    summary = {
        "run": run_name, "model": args.model, "model_id": config.model_id, "native_explainer": native_name,
        "training_seed": config.seed, "split_seed": config.split_seed, "smoke_test": is_smoke,
        "git_commit": git_commit(), "checkpoint_sha256": sha256(weights),
        "n_test": len(test_ids), "classification": classification,
        "faithfulness": {name: faithfulness_summary(rows) for name, rows in faith.items()},
        "faithfulness_protocol": FAITHFULNESS, "ig_protocol": IG,
        "localization": localization,
        "localization_protocol": {"iou_threshold": "top 20% of pixels", "pointing_tolerances_px": [0, 15],
                                  "mask_resize": "nearest neighbor to 224x224"},
        "timings_s": timings,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": __import__("transformers").__version__,
                     "captum": __import__("captum").__version__},
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("done: %s  test macro-F1 %.4f  results in %s", run_name, test_metrics["macro_f1"], out_dir)


if __name__ == "__main__":
    main()
