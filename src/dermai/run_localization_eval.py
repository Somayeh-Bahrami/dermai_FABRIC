"""
Step 10 (CNN) / Step 11 (ViT) -- applies the IoU/pointing-game framework to a
directory of raw heatmap .npy arrays and produces the mean +/- std table
needed for Overleaf.

Updated to parse Ariel's actual filename convention directly, since no
separate manifest.csv is produced:

    {image_id}__true-{true_label}__pred-{pred_label}__cam-{cam_label}.npy
    e.g. ISIC_0024313__true-mel__pred-nv__cam-nv.npy

image_id is recovered by splitting on the first "__" (image_ids themselves
contain a single underscore, e.g. "ISIC_0024313", so splitting on the
double-underscore separator is safe and unambiguous).

Usage (Step 10, CNN):
    python run_localization_eval.py \
        --heatmap-dir path/to/downloaded/gradcam/test \
        --mask-dir data/HAM10000_segmentations_lesion_tschandl \
        --model-name EfficientNet-B0 \
        --output results/localization_efficientnet.csv

Usage (Step 11, ViT) once ViT raw arrays exist in the same style:
    python run_localization_eval.py \
        --heatmap-dir path/to/downloaded/rollout/test \
        --mask-dir data/HAM10000_segmentations_lesion_tschandl \
        --model-name ViT-B-16 \
        --output results/localization_vit.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from localization import LocalizationEvaluator, summarize
from utils import build_image_id_to_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate localization (IoU + pointing game) for one model's heatmaps")
    parser.add_argument("--heatmap-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", required=True, type=Path,
                        help="per-image CSV output path")
    parser.add_argument("--threshold-method", default="percentile",
                        choices=["percentile", "fixed", "otsu"])
    parser.add_argument("--threshold-value", type=float, default=80.0)
    parser.add_argument("--pointing-tolerance", type=int, default=15,
                        help="pointing-game hit tolerance in pixels (Zhang et al. default 15; 0 = exact pixel)")
    parser.add_argument("--mask-suffix", default="_segmentation.png")
    args = parser.parse_args()

    id_to_path = build_image_id_to_path(args.heatmap_dir)
    print(f"found {len(id_to_path)} heatmap files for {args.model_name}")

    evaluator = LocalizationEvaluator(
        threshold_method=args.threshold_method, threshold_value=args.threshold_value,
        pointing_tolerance=args.pointing_tolerance)

    results = []
    missing_masks = []
    for image_id, heatmap_path in id_to_path.items():
        mask_path = Path(args.mask_dir) / f"{image_id}{args.mask_suffix}"
        if not mask_path.exists():
            missing_masks.append(image_id)
            continue

        heatmap = np.load(heatmap_path)
        mask_img = Image.open(mask_path).convert("L")
        gt_mask = np.array(mask_img) > 127

        if heatmap.shape != gt_mask.shape:
            mask_img = mask_img.resize(
                (heatmap.shape[1], heatmap.shape[0]), resample=Image.NEAREST)
            gt_mask = np.array(mask_img) > 127

        results.append(evaluator.evaluate_single(
            heatmap, gt_mask, image_id=image_id))

    if missing_masks:
        print(
            f"WARNING: {len(missing_masks)} images had no matching ground-truth mask and were skipped, e.g. {missing_masks[:5]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_id", "iou", "pointing_game_hit", "heatmap_positive_fraction"])
        for r in results:
            writer.writerow(
                [r.image_id, r.iou, r.pointing_game_hit, r.heatmap_positive_fraction])

    summary = summarize(results, args.model_name)
    print("\n--- Summary (paste into Overleaf table) ---")
    print(f"Model: {summary['model']}")
    print(f"N images: {summary['n_images']}")
    print(f"IoU: {summary['iou_mean']:.3f} +/- {summary['iou_std']:.3f}")
    print(f"Pointing-game hit-rate: {summary['pointing_game_hit_rate']:.3f}")
    print(f"\nPer-image results written to: {args.output}")


if __name__ == "__main__":
    main()
