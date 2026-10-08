import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dermai.attrollout import AttentionRollout, RolloutConfig
from dermai.config import Config
from dermai.data import CLASSES, DataModule
from dermai.models import ModelFactory
from dermai.utils import Timer, get_logger, pick_device, denormalize_image, overlay_heatmap

logger = get_logger()

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate attention-rollout heatmaps for a ViT checkpoint")
    parser.add_argument("--config", required=True, help="YAML config providing data_dir/batch_size/seed")
    parser.add_argument("--checkpoint", required=True, help="HF Hub repo id or local checkpoint directory")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None, help="defaults to <output_dir>/rollout/<checkpoint-name>/<split>")
    parser.add_argument("--alpha", type=float, default=0.45, help="heatmap overlay opacity")
    parser.add_argument("--head-fusion", default="mean", choices=["mean", "max", "min"])
    parser.add_argument("--discard-ratio", type=float, default=0.0, help="fraction of lowest attention entries to zero out per row before rollout")
    parser.add_argument("--zip", action="store_true", help="also archive the output directory as a .zip")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    device = pick_device(args.device or config.device)

    processor = ModelFactory.processor(args.checkpoint)
    # Load with eager attention so the model materializes attention matrices.
    # transformers 5.x defaults to SDPA, a fused kernel that does NOT return
    # attention weights, so output_attentions=True would yield an empty tuple and
    # break rollout. eager is numerically identical, just unfused. This is the same
    # as ModelFactory.load (a plain from_pretrained) plus the one kwarg rollout needs.
    from transformers import AutoModelForImageClassification
    model = AutoModelForImageClassification.from_pretrained(
        args.checkpoint, attn_implementation="eager"
    ).to(device)

    data = DataModule(config.data_dir, processor, config.batch_size, config.num_workers, config.seed)
    data.setup()
    loader = data.loader(args.split)

    output_dir = (
        Path(args.output) if args.output
        else config.output_dir / "rollout" / Path(args.checkpoint).name / args.split
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("model %s  split %s  %d images -> %s", args.checkpoint, args.split, len(data.splits[args.split]), output_dir)

    mean, std = processor.image_mean, processor.image_std
    rollout_config = RolloutConfig(head_fusion=args.head_fusion, discard_ratio=args.discard_ratio)
    rollout = AttentionRollout(model, rollout_config)

    timer = Timer()
    written = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        result = rollout(pixel_values)
        predicted = result.logits.argmax(dim=1)
        for i, image_id in enumerate(batch["image_id"]):
            true_name = CLASSES[labels[i].item()]
            pred_name = CLASSES[predicted[i].item()]
            stem = f"{image_id}__true-{true_name}__pred-{pred_name}"
            np.save(output_dir / f"{stem}.npy", result.heatmap[i].detach().cpu().numpy())
            image = denormalize_image(pixel_values[i], mean, std)
            overlay = overlay_heatmap(image, result.heatmap[i], alpha=args.alpha)
            overlay.save(output_dir / f"{stem}.png")
            written += 1
    logger.info("wrote %d heatmaps in %s", written, Timer.format(timer.elapsed()))

    if args.zip:
        archive = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir)
        logger.info("zipped to %s", archive)


if __name__ == "__main__":
    main()