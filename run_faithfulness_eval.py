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
    parser.add_argument("--limit", type=int, default=None, help="evaluate at most this many images")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    device = pick_device(args.device or config.device)

    processor = ModelFactory.processor(args.checkpoint)
    model = ModelFactory.load(args.checkpoint)
    data = DataModule(config.data_dir, processor, config.batch_size, config.num_workers, config.seed)
    data.setup()

    id_to_heatmap = build_image_id_to_path(args.heatmap_dir)
    evaluator = DeletionInsertion(
        model, device,
        deletion_substrate=SUBSTRATES[args.deletion_substrate](),
        insertion_substrate=SUBSTRATES[args.insertion_substrate](),
        step_pixels=args.step_pixels,
    )
    rng = np.random.default_rng(config.seed)

    total = min(len(id_to_heatmap), args.limit or len(id_to_heatmap))
    logger.info("%s  device %s  %d images  del %s  ins %s  step %d",
                config.run_name, device, total, args.deletion_substrate, args.insertion_substrate, args.step_pixels)

    results, random_results = [], []
    num_pixels = None
    timer = Timer()
    for batch in data.loader(args.split):
        for i, image_id in enumerate(batch["image_id"]):
            if image_id not in id_to_heatmap:
                continue
            heatmap = np.load(id_to_heatmap[image_id])
            num_pixels = heatmap.size
            target = target_class_from_filename(id_to_heatmap[image_id].stem)
            pixel_values = batch["pixel_values"][i]
            results.append(evaluator.run_single(pixel_values, heatmap, image_id, target))
            if args.random_control:
                noise = rng.random(heatmap.shape).astype(np.float32)
                random_results.append(evaluator.run_single(pixel_values, noise, image_id, target))
            if len(results) % 50 == 0:
                logger.info("  %d/%d images  %s", len(results), total, Timer.format(timer.elapsed()))
            if args.limit and len(results) >= args.limit:
                break
        if args.limit and len(results) >= args.limit:
            break

    _write(args.output, results, random_results if args.random_control else None,
           evaluator.step_fractions(num_pixels))
    logger.info("wrote %d results to %s in %s", len(results), args.output, Timer.format(timer.elapsed()))


def _write(output: Path, results, random_results, fractions: np.ndarray) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "target_class", "deletion_auc", "insertion_auc"])
        for r in results:
            writer.writerow([r.image_id, r.target_class, r.deletion_auc, r.insertion_auc])

    curves = {
        "fractions": fractions,
        "deletion": np.stack([r.deletion_curve for r in results]),
        "insertion": np.stack([r.insertion_curve for r in results]),
    }
    if random_results:
        curves["deletion_random"] = np.stack([r.deletion_curve for r in random_results])
        curves["insertion_random"] = np.stack([r.insertion_curve for r in random_results])
    np.savez(output.with_suffix(".npz"), **curves)


if __name__ == "__main__":
    main()
