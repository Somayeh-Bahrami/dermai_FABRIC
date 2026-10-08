"""Trap T1: the test split must depend on split_seed only, and must equal the 1,002
images behind the published seed-42 results."""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dermai.data import CLASSES, load_metadata, split_metadata

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_IDS = ROOT / "results" / "localization_efficientnet_best.csv"
DATA_DIR = ROOT / "data"


def reference_test_ids() -> list[str]:
    with REFERENCE_IDS.open() as handle:
        return sorted(row["image_id"] for row in csv.DictReader(handle))


def synthetic_metadata(num_lesions: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for lesion in range(num_lesions):
        dx = CLASSES[lesion % len(CLASSES)]
        for _ in range(rng.integers(1, 3)):
            rows.append({"lesion_id": f"HAM_{lesion:07d}", "image_id": f"ISIC_{len(rows):07d}", "dx": dx})
    return pd.DataFrame(rows)


def test_reference_has_1002_unique_ids():
    ids = reference_test_ids()
    assert len(ids) == 1002 and len(set(ids)) == 1002


def test_split_ignores_training_seed():
    # Training seeds only reach DataModule through Config.seed; the split takes split_seed alone.
    metadata = synthetic_metadata()
    splits = [split_metadata(metadata, split_seed=42) for _ in (42, 43, 44)]
    for split in splits[1:]:
        for name in ("train", "val", "test"):
            assert list(split[name].image_id) == list(splits[0][name].image_id)


def test_splits_are_lesion_disjoint():
    splits = split_metadata(synthetic_metadata(), split_seed=42)
    lesions = {name: set(frame.lesion_id) for name, frame in splits.items()}
    assert not lesions["train"] & lesions["test"]
    assert not lesions["train"] & lesions["val"]
    assert not lesions["val"] & lesions["test"]


@pytest.mark.skipif(not (DATA_DIR / "HAM10000_metadata").exists(), reason="HAM10000 metadata not downloaded")
def test_real_test_split_matches_published_results():
    # Assumes all 10,015 images are present, which is how the published results were produced.
    test_ids = sorted(split_metadata(load_metadata(DATA_DIR), split_seed=42)["test"].image_id)
    assert test_ids == reference_test_ids()
