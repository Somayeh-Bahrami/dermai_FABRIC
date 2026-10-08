"""
Localization evaluation framework (Step 9).

Computes IoU and pointing-game hit-rate between an explanation heatmap
(Grad-CAM for the CNN, Attention Rollout for the ViT) and the HAM10000
ground-truth lesion segmentation mask.

Deliberately architecture-agnostic: this module does not know or care
whether a heatmap came from Grad-CAM or Attention Rollout. As long as a
heatmap is a single-channel numpy array of shape (H, W) with values in
[0, 1] and matches the mask's resolution, it works.

Usage:
    evaluator = LocalizationEvaluator(threshold_method="percentile", threshold_value=80)
    result = evaluator.evaluate_single(heatmap, gt_mask)
    print(result.iou, result.pointing_game_hit)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class LocalizationResult:
    image_id: str
    iou: float
    pointing_game_hit: bool
    # sanity-check: % of pixels kept after thresholding
    heatmap_positive_fraction: float


class LocalizationEvaluator:
    """Computes IoU and pointing-game metrics between a heatmap and a binary mask.

    threshold_method:
        "percentile"  -> keep top `threshold_value` percent of heatmap values (default, recommended)
        "fixed"       -> keep pixels with value >= threshold_value (threshold_value in [0, 1])
        "otsu"        -> automatic Otsu threshold (no threshold_value needed)
    """

    def __init__(self, threshold_method: str = "percentile", threshold_value: float = 80.0,
                 pointing_tolerance: int = 15) -> None:
        if threshold_method not in {"percentile", "fixed", "otsu"}:
            raise ValueError(f"unknown threshold_method: {threshold_method}")
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.pointing_tolerance = pointing_tolerance

    # ------------------------------------------------------------------
    # Thresholding: continuous heatmap -> binary "attended region" mask
    # ------------------------------------------------------------------
    def binarize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        heatmap = np.asarray(heatmap, dtype=np.float64)
        if heatmap.ndim != 2:
            raise ValueError(
                f"expected a 2D heatmap (H, W), got shape {heatmap.shape}")

        if self.threshold_method == "percentile":
            # e.g. threshold_value=80 -> keep the hottest 20% of pixels
            cutoff = np.percentile(heatmap, self.threshold_value)
            return heatmap >= cutoff

        if self.threshold_method == "fixed":
            return heatmap >= self.threshold_value

        # otsu
        return self._otsu_threshold(heatmap)

    @staticmethod
    def _otsu_threshold(heatmap: np.ndarray) -> np.ndarray:
        scaled = (heatmap * 255).clip(0, 255).astype(np.uint8)
        hist, _ = np.histogram(scaled, bins=256, range=(0, 256))
        total = scaled.size
        sum_all = np.dot(np.arange(256), hist)
        sum_bg, weight_bg, best_between, best_thresh = 0.0, 0.0, 0.0, 0
        for t in range(256):
            weight_bg += hist[t]
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break
            sum_bg += t * hist[t]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_all - sum_bg) / weight_fg
            between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if between > best_between:
                best_between, best_thresh = between, t
        return scaled >= best_thresh

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def compute_iou(self, heatmap: np.ndarray, gt_mask: np.ndarray) -> float:
        pred_mask = self.binarize_heatmap(heatmap)
        gt_mask = gt_mask.astype(bool)
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        if union == 0:
            return 0.0
        return float(intersection) / float(union)

    def compute_pointing_game(self, heatmap: np.ndarray, gt_mask: np.ndarray) -> bool:
        peak_row, peak_col = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        gt_mask = gt_mask.astype(bool)
        if gt_mask[peak_row, peak_col]:
            return True
        if self.pointing_tolerance <= 0:
            return False
        return self._within_tolerance(gt_mask, peak_row, peak_col)

    def _within_tolerance(self, gt_mask: np.ndarray, peak_row: int, peak_col: int) -> bool:
        tau = self.pointing_tolerance
        rows, cols = gt_mask.shape
        row_lo, row_hi = max(0, peak_row - tau), min(rows, peak_row + tau + 1)
        col_lo, col_hi = max(0, peak_col - tau), min(cols, peak_col + tau + 1)
        window = gt_mask[row_lo:row_hi, col_lo:col_hi]
        if not window.any():
            return False
        window_rows, window_cols = np.nonzero(window)
        distances_squared = (window_rows + row_lo - peak_row) ** 2 + (window_cols + col_lo - peak_col) ** 2
        return bool((distances_squared <= tau * tau).any())

    def evaluate_single(self, heatmap: np.ndarray, gt_mask: np.ndarray, image_id: str = "") -> LocalizationResult:
        if heatmap.shape != gt_mask.shape:
            raise ValueError(
                f"heatmap shape {heatmap.shape} != mask shape {gt_mask.shape} for image_id={image_id!r}; "
                "resize one to match the other before calling evaluate_single."
            )
        pred_mask = self.binarize_heatmap(heatmap)
        gt_bool = gt_mask.astype(bool)
        intersection = np.logical_and(pred_mask, gt_bool).sum()
        union = np.logical_or(pred_mask, gt_bool).sum()
        iou = float(intersection) / float(union) if union > 0 else 0.0
        hit = self.compute_pointing_game(heatmap, gt_bool)
        positive_fraction = float(pred_mask.sum()) / float(pred_mask.size)
        return LocalizationResult(
            image_id=image_id,
            iou=iou,
            pointing_game_hit=hit,
            heatmap_positive_fraction=positive_fraction,
        )

    # ------------------------------------------------------------------
    # Batch evaluation over a whole test set
    # ------------------------------------------------------------------
    def evaluate_batch(
        self,
        heatmap_dir: Path,
        mask_dir: Path,
        manifest_image_ids: list[str],
        mask_suffix: str = "_segmentation.png",
        heatmap_suffix: str = ".npy",
        target_size: tuple[int, int] | None = None,
    ) -> list[LocalizationResult]:
        """Loops over image_ids, loads the matching heatmap .npy + mask .png, evaluates each.

        target_size: if the heatmap and mask resolutions differ (they shouldn't, but
        just in case a model uses a different input size), resize the mask to this
        (H, W) with nearest-neighbor interpolation to preserve the binary mask.
        """
        results = []
        for image_id in manifest_image_ids:
            heatmap_path = Path(heatmap_dir) / f"{image_id}{heatmap_suffix}"
            mask_path = Path(mask_dir) / f"{image_id}{mask_suffix}"

            if not heatmap_path.exists():
                raise FileNotFoundError(
                    f"missing heatmap for {image_id}: {heatmap_path}")
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"missing ground-truth mask for {image_id}: {mask_path}")

            heatmap = np.load(heatmap_path)
            mask_img = Image.open(mask_path).convert("L")
            if target_size is not None:
                mask_img = mask_img.resize(
                    (target_size[1], target_size[0]), resample=Image.NEAREST)
            gt_mask = (np.array(mask_img) > 127)

            if heatmap.shape != gt_mask.shape:
                mask_img = Image.fromarray(gt_mask.astype(np.uint8) * 255).resize(
                    (heatmap.shape[1], heatmap.shape[0]
                     ), resample=Image.NEAREST
                )
                gt_mask = np.array(mask_img) > 127

            results.append(self.evaluate_single(
                heatmap, gt_mask, image_id=image_id))
        return results


# ----------------------------------------------------------------------
# Aggregation helper: turns a list of per-image results into the
# mean +/- std summary table needed for Overleaf (Steps 10 & 11 DoD).
# ----------------------------------------------------------------------
def summarize(results: list[LocalizationResult], model_name: str) -> dict:
    ious = np.array([r.iou for r in results])
    hits = np.array([r.pointing_game_hit for r in results])
    return {
        "model": model_name,
        "n_images": len(results),
        "iou_mean": float(ious.mean()),
        "iou_std": float(ious.std()),
        "pointing_game_hit_rate": float(hits.mean()),
    }


def results_to_dataframe(results: list[LocalizationResult]):
    import pandas as pd
    return pd.DataFrame([asdict(r) for r in results])
