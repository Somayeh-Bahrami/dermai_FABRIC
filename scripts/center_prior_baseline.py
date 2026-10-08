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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dermai.baselines import center_prior_heatmap  # noqa: E402
from dermai.localization import evaluate_protocol  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-from", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    image_ids = [row["image_id"] for row in csv.DictReader(args.ids_from.open())]
    heatmap = center_prior_heatmap()
    rows = evaluate_protocol(((image_id, heatmap) for image_id in image_ids), args.mask_dir)

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
