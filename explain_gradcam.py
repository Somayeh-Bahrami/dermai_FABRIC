import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dermai.config import Config
from dermai.data import CLASSES, DataModule
from dermai.gradcam import GradCAM
from dermai.models import ModelFactory
from dermai.utils import Timer, denormalize_image, get_logger, overlay_heatmap, pick_device

logger = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for a CNN checkpoint")
    parser.add_argument("--config", required=True, help="YAML config providing data_dir/batch_size/seed")
    parser.add_argument("--checkpoint", required=True, help="HF Hub repo id or local checkpoint directory")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None, help="defaults to <output_dir>/gradcam/<checkpoint-name>/<split>")
    parser.add_argument("--alpha", type=float, default=0.45, help="heatmap overlay opacity")
    parser.add_argument(
        "--target",
        default="predicted",
        choices=["predicted", "true"],
        help="which class's logit to explain: the model's own prediction, or the ground-truth label",
    )
    parser.add_argument("--zip", action="store_true", help="also archive the output directory as a .zip")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    device = pick_device(args.device or config.device)

    processor = ModelFactory.processor(args.checkpoint)
    model = ModelFactory.load(args.checkpoint).to(device)

    data = DataModule(config.data_dir, processor, config.batch_size, config.num_workers, config.seed)
    data.setup()
    loader = data.loader(args.split)

    output_dir = Path(args.output) if args.output else config.output_dir / "gradcam" / Path(args.checkpoint).name / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("model %s  split %s  %d images -> %s", args.checkpoint, args.split, len(data.splits[args.split]), output_dir)

    mean, std = processor.image_mean, processor.image_std
    timer = Timer()
    written = 0
    with GradCAM(model) as gradcam:
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            with torch.no_grad():
                predicted = model(pixel_values=pixel_values).logits.argmax(dim=1)
            target_class = labels if args.target == "true" else predicted
            result = gradcam(pixel_values, target_class=target_class)
            for i, image_id in enumerate(batch["image_id"]):
                true_name = CLASSES[labels[i].item()]
                pred_name = CLASSES[predicted[i].item()]
                cam_name = CLASSES[target_class[i].item()]
                stem = f"{image_id}__true-{true_name}__pred-{pred_name}__cam-{cam_name}"

                np.save(output_dir / f"{stem}.npy", result.cam[i].detach().cpu().numpy())

                image = denormalize_image(pixel_values[i], mean, std)
                overlay = overlay_heatmap(image, result.cam[i], alpha=args.alpha)
                overlay.save(output_dir / f"{stem}.png")
                written += 1

    logger.info("wrote %d heatmaps in %s", written, Timer.format(timer.elapsed()))

    if args.zip:
        archive = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir)
        logger.info("zipped to %s", archive)


if __name__ == "__main__":
    main()
