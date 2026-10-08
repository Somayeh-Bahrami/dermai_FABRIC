"""Model-free center-prior localization baseline on the same test images and evaluator settings.

Usage:
    python scripts/center_prior_baseline.py \
        --ids-from results/localization_efficientnet_best.csv \
        --mask-dir data/HAM10000_segmentations_lesion_tschandl \
        --output results/localization_center_prior.csv

Reference result (2026-10-07, 1,002 test images): IoU 0.476 +/- 0.172,
pointing game 0.995 (tolerance 15 px) / 0.963 (tolerance 0).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "dermai"))
from localization import LocalizationEvaluator  # noqa: E402

SIZE = 224


def center_prior_heatmap(size: int = SIZE) -> np.ndarray:
    rows, cols = np.mgrid[0:size, 0:size]
    center = (size - 1) / 2
    distance_squared = (rows - center) ** 2 + (cols - center) ** 2
    return 1.0 - distance_squared / distance_squared.max()


def load_mask(mask_dir: Path, image_id: str) -> np.ndarray:
    mask = Image.open(mask_dir / f"{image_id}_segmentation.png").convert("L")
    return np.array(mask.resize((SIZE, SIZE), Image.NEAREST)) > 127


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-from", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    image_ids = [row["image_id"] for row in csv.DictReader(args.ids_from.open())]
    heatmap = center_prior_heatmap()
    evaluators = {tolerance: LocalizationEvaluator("percentile", 80.0, pointing_tolerance=tolerance)
                  for tolerance in (0, 15)}

    rows = []
    for image_id in image_ids:
        mask = load_mask(args.mask_dir, image_id)
        result = evaluators[0].evaluate_single(heatmap, mask, image_id)
        hit_tolerant = evaluators[15].compute_pointing_game(heatmap, mask)
        rows.append({"image_id": image_id, "iou": result.iou, "pointing_hit_tol0": result.pointing_game_hit,
                     "pointing_hit_tol15": hit_tolerant, "lesion_area_fraction": float(mask.mean())})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ious = np.array([row["iou"] for row in rows])
    print(f"n={len(rows)} IoU {ious.mean():.3f} +/- {ious.std():.3f} "
          f"PG tol0 {np.mean([r['pointing_hit_tol0'] for r in rows]):.3f} "
          f"PG tol15 {np.mean([r['pointing_hit_tol15'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
