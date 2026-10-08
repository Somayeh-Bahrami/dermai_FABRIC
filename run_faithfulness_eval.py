import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dermai.config import Config
from dermai.data import DataModule
from dermai.faithfulness import (
    DeletionInsertion,
    GaussianBlurSubstrate,
    MeanFillSubstrate,
    center_ordering,
    evaluate_orderings,
    heatmap_dir_ordering,
    random_ordering,
    target_class_from_filename,
)
from dermai.models import ModelFactory
from dermai.utils import Timer, build_image_id_to_path, get_logger, pick_device

logger = get_logger()

SUBSTRATES = {"mean": MeanFillSubstrate, "blur": GaussianBlurSubstrate}


def main() -> None:
    parser = argparse.ArgumentParser(description="RISE deletion/insertion faithfulness evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="HF Hub repo id or local checkpoint directory")
    parser.add_argument("--heatmap-dir", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", required=True, type=Path, help="per-image CSV path; curves saved alongside as .npz")
    parser.add_argument("--deletion-substrate", default="blur", choices=list(SUBSTRATES),
                        help="fill for removed pixels: blur (default) or mean")
    parser.add_argument("--insertion-substrate", default="mean", choices=list(SUBSTRATES),
                        help="starting canvas for insertion: mean (default) or blur")
    parser.add_argument("--step-pixels", type=int, default=512)
    parser.add_argument("--random-control", action="store_true", help="also score a random saliency ordering")
    parser.add_argument("--center-control", action="store_true",
                        help="also score the model-free center-prior ordering (center pixels first)")
    parser.add_argument("--limit", type=int, default=None, help="evaluate at most this many images")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    device = pick_device(args.device or config.device)

    processor = ModelFactory.processor(args.checkpoint)
    model = ModelFactory.load(args.checkpoint)
    data = DataModule(config.data_dir, processor, config.batch_size, config.num_workers, config.split_seed)
    data.setup()

    id_to_heatmap = build_image_id_to_path(args.heatmap_dir)
    evaluator = DeletionInsertion(
        model, device,
        deletion_substrate=SUBSTRATES[args.deletion_substrate](),
        insertion_substrate=SUBSTRATES[args.insertion_substrate](),
        step_pixels=args.step_pixels,
    )
    orderings = {"heatmap": heatmap_dir_ordering(id_to_heatmap)}
    if args.random_control:
        orderings["random"] = random_ordering()
    if args.center_control:
        orderings["center"] = center_ordering()

    total = min(len(id_to_heatmap), args.limit or len(id_to_heatmap))
    logger.info("%s  device %s  %d images  del %s  ins %s  step %d  orderings %s",
                config.run_name, device, total, args.deletion_substrate, args.insertion_substrate,
                args.step_pixels, list(orderings))

    timer = Timer()
    results = evaluate_orderings(
        evaluator, data.loader(args.split), orderings,
        target_for=lambda image_id: target_class_from_filename(id_to_heatmap[image_id].stem),
        image_ids=set(id_to_heatmap), limit=args.limit, logger=logger,
    )
    num_pixels = int(np.prod(np.load(next(iter(id_to_heatmap.values()))).shape))
    _write(args.output, results, evaluator.step_fractions(num_pixels))
    logger.info("wrote %d results to %s in %s", len(results["heatmap"]), args.output, Timer.format(timer.elapsed()))


def _write(output: Path, results: dict, fractions: np.ndarray) -> None:
    """CSV for the explainer ordering; one .npz holding every ordering's curves
    (keys deletion, insertion, deletion_random, insertion_center, ...)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "target_class", "deletion_auc", "insertion_auc"])
        for r in results["heatmap"]:
            writer.writerow([r.image_id, r.target_class, r.deletion_auc, r.insertion_auc])

    curves = {"fractions": fractions}
    for name, rows in results.items():
        suffix = "" if name == "heatmap" else f"_{name}"
        curves[f"deletion{suffix}"] = np.stack([r.deletion_curve for r in rows])
        curves[f"insertion{suffix}"] = np.stack([r.insertion_curve for r in rows])
    np.savez(output.with_suffix(".npz"), **curves)


if __name__ == "__main__":
    main()
