import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dermai.config import Config
from dermai.data import DataModule
from dermai.explain import ig_explainer, write_heatmaps
from dermai.integrated_gradients import IntegratedGradients
from dermai.models import ModelFactory
from dermai.utils import Timer, get_logger, pick_device

logger = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Integrated Gradients heatmaps for any checkpoint")
    parser.add_argument("--config", required=True, help="YAML config providing data_dir/batch_size/split_seed")
    parser.add_argument("--checkpoint", required=True, help="HF Hub repo id or local checkpoint directory")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None, help="defaults to <output_dir>/ig/<checkpoint-name>/<split>")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--internal-batch-size", type=int, default=32, help="lower this on CUDA OOM")
    parser.add_argument("--png", action="store_true", help="also write overlay PNGs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    device = pick_device(args.device or config.device)
    processor = ModelFactory.processor(args.checkpoint)
    model = ModelFactory.load(args.checkpoint).to(device).eval()
    data = DataModule(config.data_dir, processor, config.batch_size, config.num_workers, config.split_seed)
    data.setup()

    output_dir = Path(args.output) if args.output else config.output_dir / "ig" / Path(args.checkpoint).name / args.split
    timer = Timer()
    ig = IntegratedGradients(model, steps=args.steps, internal_batch_size=args.internal_batch_size)
    written = write_heatmaps(ig_explainer(ig), data.loader(args.split), output_dir, device, limit=args.limit,
                             png_mean_std=(processor.image_mean, processor.image_std) if args.png else None)
    logger.info("wrote %d IG heatmaps to %s in %s", written, output_dir, Timer.format(timer.elapsed()))


if __name__ == "__main__":
    main()
