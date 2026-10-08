"""Integrated Gradients (Sundararajan et al., 2017) as one shared, class-specific
explainer for both architectures, so faithfulness and localization can be compared
with the explanation method held fixed.

Protocol: 32 steps, baseline = zeros in the model's normalized space (the
per-channel mean color, matching the RISE mean-fill substrate), target = the
model's predicted class, attribution = |IG| summed over channels, then min-max
normalized per image to [0, 1] at input resolution (224x224).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from captum.attr import IntegratedGradients as CaptumIG


@dataclass
class IGOutput:
    heatmap: torch.Tensor       # (batch, height, width), normalized to [0, 1]
    target_class: torch.Tensor  # (batch,)
    logits: torch.Tensor        # (batch, num_classes)


class _LogitsOnly(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


class IntegratedGradients:
    def __init__(self, model: nn.Module, steps: int = 32, internal_batch_size: int | None = 32) -> None:
        self.model = model
        self.steps = steps
        self.internal_batch_size = internal_batch_size
        self._ig = CaptumIG(_LogitsOnly(model))

    def __call__(self, pixel_values: torch.Tensor, target_class: torch.Tensor | None = None) -> IGOutput:
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            logits = self.model(pixel_values=pixel_values).logits
        if target_class is None:
            target_class = logits.argmax(dim=1)
        attributions = self._ig.attribute(
            pixel_values,
            baselines=torch.zeros_like(pixel_values),
            target=target_class,
            n_steps=self.steps,
            internal_batch_size=self.internal_batch_size,
        )
        heatmap = self._normalize(attributions.detach().abs().sum(dim=1))
        self.model.train(was_training)
        return IGOutput(heatmap=heatmap, target_class=target_class, logits=logits.detach())

    @staticmethod
    def _normalize(heatmap: torch.Tensor) -> torch.Tensor:
        flat_min = heatmap.flatten(1).min(dim=1).values.view(-1, 1, 1)
        flat_max = heatmap.flatten(1).max(dim=1).values.view(-1, 1, 1)
        return (heatmap - flat_min) / (flat_max - flat_min).clamp_min(1e-8)
