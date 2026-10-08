import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dermai.config import Config
from dermai.data import DataModule
from dermai.models import ModelFactory
from dermai.trainer import Trainer
from dermai.utils import Timer, get_logger, pick_device, set_seed

logger = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-test", action="store_true",
                        help="skip final test-set evaluation (use during ablation sweeps)")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    set_seed(config.seed)
    device = pick_device(args.device or config.device)
    logger.info("model %s  device %s  seed %d", config.run_name, device.type, config.seed)

    processor = ModelFactory.processor(config.model_id)
    data = DataModule(
        config.data_dir, processor, config.batch_size, config.num_workers, config.seed,
        augment=config.augment,
    )
    setup_timer = Timer()
    data.setup()
    logger.info("data ready in %s  splits %s", Timer.format(setup_timer.elapsed()), data.split_sizes())

    model = ModelFactory.build(
        config.model_id,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        drop_connect_rate=config.drop_connect_rate,
    )
    logger.info("dropout config: %s", ModelFactory.describe_dropout(model))
    trainer = Trainer(model, data, config, device)
    trainer.fit()
    if not args.skip_test:
        trainer.test()


if __name__ == "__main__":
    main()