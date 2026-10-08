"""RISE deletion/insertion faithfulness metrics (Petsiuk et al., 2018).

Deletion removes the highest-saliency pixels first and tracks P(target class);
a faithful heatmap causes a fast drop, hence low AUC. Insertion adds those same
pixels into a substrate canvas; a faithful heatmap causes a fast rise, hence high
AUC. Architecture-agnostic: it only reads the saliency ranking, so it treats
Grad-CAM and attention rollout identically.

The module also exposes the offline analysis helpers (raw RISE AUC, signed AOPC
relative to a random control, per-class macro-averaging) shared by the reporting
and plotting scripts.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.functional import gaussian_blur

from .data import CLASSES, LABEL_TO_INDEX


@dataclass
class FaithfulnessResult:
    image_id: str
    target_class: int
    deletion_auc: float
    insertion_auc: float
    deletion_curve: np.ndarray
    insertion_curve: np.ndarray


class Substrate:
    """Replacement content for perturbed pixels, built in the model's normalized space."""

    def build(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MeanFillSubstrate(Substrate):
    """Zeros in normalized space, i.e. the per-channel mean color (RISE gray baseline)."""

    def build(self, image: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(image)


class GaussianBlurSubstrate(Substrate):
    def __init__(self, kernel_size: int = 11, sigma: float = 5.0) -> None:
        self.kernel_size = kernel_size
        self.sigma = sigma

    def build(self, image: torch.Tensor) -> torch.Tensor:
        return gaussian_blur(image, [self.kernel_size, self.kernel_size], [self.sigma, self.sigma])


def target_class_from_filename(stem: str) -> int:
    """Recover the model's clean prediction from the heatmap filename. Works for
    both the Grad-CAM (has a cam- token) and rollout (no cam- token) conventions."""
    for token in stem.split("__"):
        if token.startswith("pred-"):
            return LABEL_TO_INDEX[token[len("pred-"):]]
    raise ValueError(f"no pred- token in heatmap filename: {stem!r}")


class DeletionInsertion:
    def __init__(self, model: torch.nn.Module, device: torch.device,
                 deletion_substrate: Substrate, insertion_substrate: Substrate,
                 step_pixels: int = 512, forward_batch_size: int = 32) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.deletion_substrate = deletion_substrate
        self.insertion_substrate = insertion_substrate
        self.step_pixels = step_pixels
        self.forward_batch_size = forward_batch_size

    @torch.no_grad()
    def run_single(self, pixel_values: torch.Tensor, heatmap: np.ndarray,
                   image_id: str, target_class: int) -> FaithfulnessResult:
        image = pixel_values.to(self.device)
        if heatmap.shape != tuple(image.shape[-2:]):
            raise ValueError(f"heatmap {heatmap.shape} != image {tuple(image.shape[-2:])} for {image_id!r}")

        keep = self._keep_masks(heatmap).to(self.device)
        deletion_stack = image * keep + self.deletion_substrate.build(image) * (1 - keep)
        insertion_stack = image * (1 - keep) + self.insertion_substrate.build(image) * keep

        deletion_curve = self._probability_curve(deletion_stack, target_class)
        insertion_curve = self._probability_curve(insertion_stack, target_class)
        return FaithfulnessResult(
            image_id=image_id,
            target_class=target_class,
            deletion_auc=self._auc(deletion_curve),
            insertion_auc=self._auc(insertion_curve),
            deletion_curve=deletion_curve,
            insertion_curve=insertion_curve,
        )

    def step_fractions(self, num_pixels: int) -> np.ndarray:
        return self._step_boundaries(num_pixels) / num_pixels

    def _step_boundaries(self, num_pixels: int) -> np.ndarray:
        boundaries = np.arange(0, num_pixels + 1, self.step_pixels)
        if boundaries[-1] != num_pixels:
            boundaries = np.append(boundaries, num_pixels)
        return boundaries

    def _keep_masks(self, heatmap: np.ndarray) -> torch.Tensor:
        height, width = heatmap.shape
        num_pixels = height * width
        order = np.argsort(heatmap.ravel())[::-1]
        rank = np.empty(num_pixels, dtype=np.int64)
        rank[order] = np.arange(num_pixels)
        rank = rank.reshape(height, width)
        keep = rank[None] >= self._step_boundaries(num_pixels)[:, None, None]
        return torch.from_numpy(keep.astype(np.float32)).unsqueeze(1)

    def _probability_curve(self, stack: torch.Tensor, target_class: int) -> np.ndarray:
        probabilities = [
            self.model(pixel_values=chunk).logits.softmax(dim=1)[:, target_class]
            for chunk in stack.split(self.forward_batch_size)
        ]
        return torch.cat(probabilities).cpu().numpy()

    @staticmethod
    def _auc(curve: np.ndarray) -> float:
        return float((curve.sum() - curve[0] / 2 - curve[-1] / 2) / (len(curve) - 1))


def endpoints(curve: np.ndarray, mode: str) -> tuple[float, float]:
    """Returns (p_clean, p_base) for a deletion or insertion curve. Deletion starts
    intact and ends at the substrate; insertion is the reverse."""
    intact, substrate = float(curve[0]), float(curve[-1])
    return (intact, substrate) if mode == "deletion" else (substrate, intact)


def aopc_vs_baseline(curve: np.ndarray, mode: str) -> float:
    """Signed area of the probability change relative to the intact prediction
    (Samek et al., 2017). Deletion integrates the drop P_clean - P(k); insertion the
    gain P(k) - P_base. Left unclipped: negative values are meaningful."""
    p_clean, p_base = endpoints(curve, mode)
    signed = (p_clean - curve) if mode == "deletion" else (curve - p_base)
    return DeletionInsertion._auc(signed)


def load_run(prefix: str) -> tuple[np.ndarray, dict]:
    """Loads a saved run by path prefix, returning the per-image target classes and
    the curve arrays (deletion/insertion, plus their random controls if present)."""
    rows = list(csv.DictReader(open(f"{prefix}.csv")))
    target_classes = np.array([int(r["target_class"]) for r in rows])
    return target_classes, np.load(f"{prefix}.npz")


def raw_aucs(curves: np.ndarray) -> np.ndarray:
    return np.array([DeletionInsertion._auc(curve) for curve in curves])


def aopc_aucs(curves: np.ndarray, mode: str) -> np.ndarray:
    return np.array([aopc_vs_baseline(curve, mode) for curve in curves])


def macro_average(values: np.ndarray, target_classes: np.ndarray) -> float:
    """Mean over the classes of each class's mean, so the majority class cannot dominate."""
    class_means = [values[target_classes == c].mean()
                   for c in range(len(CLASSES)) if (target_classes == c).any()]
    return float(np.mean(class_means))
