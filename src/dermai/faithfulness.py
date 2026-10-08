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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.functional import gaussian_blur

from .baselines import center_prior_heatmap
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


# ----------------------------------------------------------------------
# Evaluation loop shared by run_faithfulness_eval.py and scripts/run_seed.py.
# An ordering maps image_id -> saliency heatmap; controls ignore the model.
# ----------------------------------------------------------------------
Ordering = Callable[[str], np.ndarray]


def random_ordering(size: int = 224) -> Ordering:
    """Random pixel ordering seeded by the image id, so every model and training
    seed sees the identical random control for a given image (paired comparisons)."""
    def ordering(image_id: str) -> np.ndarray:
        seed = int("".join(ch for ch in image_id if ch.isdigit()) or 0)
        return np.random.default_rng(seed).random((size, size)).astype(np.float32)
    return ordering


def center_ordering(size: int = 224) -> Ordering:
    """Fixed center-out ordering (model-free center prior), identical for every image."""
    heatmap = center_prior_heatmap(size, tie_break=True)
    return lambda image_id: heatmap


def heatmap_dir_ordering(id_to_path: dict[str, Path]) -> Ordering:
    return lambda image_id: np.load(id_to_path[image_id])


def evaluate_orderings(evaluator: DeletionInsertion, loader, orderings: dict[str, Ordering],
                       target_for: Callable[[str], int], image_ids: set[str],
                       limit: int | None = None, log_every: int = 50, logger=None) -> dict[str, list[FaithfulnessResult]]:
    """Scores every ordering on the same images and the same target class per image."""
    results: dict[str, list[FaithfulnessResult]] = {name: [] for name in orderings}
    done = 0
    for batch in loader:
        for i, image_id in enumerate(batch["image_id"]):
            if image_id not in image_ids:
                continue
            target = target_for(image_id)
            pixel_values = batch["pixel_values"][i]
            for name, ordering in orderings.items():
                results[name].append(evaluator.run_single(pixel_values, ordering(image_id), image_id, target))
            done += 1
            if logger is not None and done % log_every == 0:
                logger.info("  faithfulness %d images", done)
            if limit is not None and done >= limit:
                return results
    return results


def write_results(output: Path, results: list[FaithfulnessResult], fractions: np.ndarray) -> None:
    """Per-image CSV plus a same-named .npz with the step fractions and both curve stacks."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "target_class", "deletion_auc", "insertion_auc"])
        for r in results:
            writer.writerow([r.image_id, r.target_class, r.deletion_auc, r.insertion_auc])
    np.savez(output.with_suffix(".npz"), fractions=fractions,
             deletion=np.stack([r.deletion_curve for r in results]),
             insertion=np.stack([r.insertion_curve for r in results]))


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
