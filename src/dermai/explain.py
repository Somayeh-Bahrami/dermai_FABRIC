"""Shared heatmap writer for every explainer (Grad-CAM, rollout, IG).

Filenames follow the existing scheme so all downstream evaluators work unchanged:
    {image_id}__true-{label}__pred-{label}[__cam-{label}].npy
The cam- token is written only for class-specific explainers; rollout is
class-agnostic, so its files carry no cam- token.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from .data import CLASSES
from .utils import denormalize_image, overlay_heatmap

# explain(pixel_values) -> (heatmap (B,H,W) in [0,1], explained class (B,) or None, logits (B,C))
Explainer = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]]


def write_heatmaps(explain: Explainer, loader, output_dir: Path, device: torch.device,
                   limit: int | None = None, png_mean_std: tuple | None = None, alpha: float = 0.45) -> int:
    """Runs `explain` over a loader and saves one .npy (and optionally a PNG overlay) per image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]
        heatmap, explained, logits = explain(pixel_values)
        predicted = logits.argmax(dim=1)
        for i, image_id in enumerate(batch["image_id"]):
            stem = f"{image_id}__true-{CLASSES[labels[i].item()]}__pred-{CLASSES[predicted[i].item()]}"
            if explained is not None:
                stem += f"__cam-{CLASSES[explained[i].item()]}"
            np.save(output_dir / f"{stem}.npy", heatmap[i].detach().cpu().numpy().astype(np.float32))
            if png_mean_std is not None:
                image = denormalize_image(pixel_values[i], *png_mean_std)
                overlay_heatmap(image, heatmap[i], alpha=alpha).save(output_dir / f"{stem}.png")
            written += 1
            if limit is not None and written >= limit:
                return written
    return written


def gradcam_explainer(gradcam) -> Explainer:
    def explain(pixel_values):
        with torch.no_grad():
            predicted = gradcam.model(pixel_values=pixel_values).logits.argmax(dim=1)
        result = gradcam(pixel_values, target_class=predicted)
        return result.cam, result.target_class, result.logits
    return explain


def rollout_explainer(rollout) -> Explainer:
    def explain(pixel_values):
        result = rollout(pixel_values)
        return result.heatmap, None, result.logits
    return explain


def ig_explainer(ig) -> Explainer:
    def explain(pixel_values):
        result = ig(pixel_values)
        return result.heatmap, result.target_class, result.logits
    return explain
