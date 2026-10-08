from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
LABEL_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}


class TrainAugmentation:
    """Lightweight, dependency-free train-time augmentation for dermoscopic images.

    Operates on the raw PIL image, before the model-specific processor runs, so
    it's identical regardless of which architecture consumes the output. Only
    ever applied to the train split -- val/test must stay unaugmented so metrics
    stay comparable across runs.

    Geometric ops (flip/rotate) are safe here because dermoscopic lesions have
    no canonical orientation. Color jitter is mild since diagnostic color cues
    (e.g. blue-white veil, pigment network) shouldn't be pushed too far from
    their true appearance.
    """

    def __init__(
        self,
        rotation_degrees: float = 15.0,
        brightness_jitter: float = 0.2,
        contrast_jitter: float = 0.2,
    ) -> None:
        self.rotation_degrees = rotation_degrees
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

        brightness_factor = 1.0 + random.uniform(-self.brightness_jitter, self.brightness_jitter)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)

        contrast_factor = 1.0 + random.uniform(-self.contrast_jitter, self.contrast_jitter)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)

        return image


class HAM10000Dataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_paths: dict[str, Path],
        processor,
        transform=None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_paths = image_paths
        self.processor = processor
        self.transform = transform  # applied to the raw PIL image, before processor

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        image = Image.open(self.image_paths[row.image_id]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        pixel_values = self.processor(image, return_tensors="pt")["pixel_values"][0]
        label = torch.tensor(LABEL_TO_INDEX[row.dx], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": label, "image_id": row.image_id}


class DataModule:
    def __init__(
        self,
        data_dir: Path,
        processor,
        batch_size: int,
        num_workers: int,
        seed: int,
        augment: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.splits: dict[str, pd.DataFrame] = {}
        self.train_transform = TrainAugmentation() if augment else None

    def setup(self) -> None:
        metadata = pd.read_csv(self.data_dir / "HAM10000_metadata", sep=None, engine="python")
        image_paths = self._index_images()
        metadata = metadata[metadata.image_id.isin(image_paths)].reset_index(drop=True)
        self.image_paths = image_paths
        train_val, test = self._group_split(metadata, test_fraction_splits=10)
        train, val = self._group_split(train_val, test_fraction_splits=9)
        self.splits = {"train": train, "val": val, "test": test}

    def _index_images(self) -> dict[str, Path]:
        return {path.stem: path for path in self.data_dir.rglob("*.jpg")}

    def _group_split(self, frame: pd.DataFrame, test_fraction_splits: int):
        splitter = StratifiedGroupKFold(n_splits=test_fraction_splits, shuffle=True, random_state=self.seed)
        keep_idx, hold_idx = next(splitter.split(frame, frame.dx, groups=frame.lesion_id))
        return frame.iloc[keep_idx].reset_index(drop=True), frame.iloc[hold_idx].reset_index(drop=True)

    def class_weights(self) -> torch.Tensor:
        counts = self.splits["train"].dx.value_counts()
        total = counts.sum()
        weights = [total / (len(CLASSES) * counts[name]) for name in CLASSES]
        return torch.tensor(weights, dtype=torch.float)

    def loader(self, split: str) -> DataLoader:
        transform = self.train_transform if split == "train" else None
        dataset = HAM10000Dataset(self.splits[split], self.image_paths, self.processor, transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=self.num_workers,
        )

    def split_sizes(self) -> dict[str, int]:
        return {name: len(frame) for name, frame in self.splits.items()}