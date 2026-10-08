import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from matplotlib import colormaps

LOGGER_NAME = "dermai"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def pick_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def denormalize_image(pixel_values: torch.Tensor, mean: list[float], std: list[float]) -> Image.Image:
    """Undo a HF image processor's normalization to recover the image the model saw."""
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)
    image = (pixel_values.detach().cpu() * std_t + mean_t).clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def overlay_heatmap(image: Image.Image, cam: torch.Tensor, alpha: float = 0.45) -> Image.Image:
    heatmap = (colormaps["jet"](cam.detach().cpu().numpy())[:, :, :3] * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap).resize(image.size, resample=Image.BILINEAR)
    return Image.blend(image.convert("RGB"), heatmap_image, alpha)


def build_image_id_to_path(heatmap_dir: Path) -> dict[str, Path]:
    """Maps image_id -> heatmap .npy path, recovering the id from the first
    "__"-separated token (image_ids contain a single underscore, e.g. ISIC_0024313)."""
    mapping: dict[str, Path] = {}
    for path in sorted(Path(heatmap_dir).glob("*.npy")):
        image_id = path.stem.split("__")[0]
        if image_id in mapping:
            raise ValueError(f"duplicate image_id {image_id!r} in {heatmap_dir} ({mapping[image_id].name}, {path.name})")
        mapping[image_id] = path
    return mapping


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    @staticmethod
    def format(seconds: float) -> str:
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes}m{secs:02d}s"
