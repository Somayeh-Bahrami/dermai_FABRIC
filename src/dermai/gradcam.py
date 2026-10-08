from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CAMOutput:
    cam: torch.Tensor  # (batch, height, width), normalized to [0, 1]
    target_class: torch.Tensor  # (batch,)
    logits: torch.Tensor  # (batch, num_classes)


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) for this project's CNN (EfficientNet-B0).

    Hooks the last conv activation before global pooling and backprops from a
    caller-chosen class logit to turn that activation into a heatmap.
    """

    def __init__(self, model: nn.Module, freeze_params: bool = True) -> None:
        self.model = model
        if freeze_params:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        target_layer = self._target_layer(model)
        self._handles = [
            target_layer.register_forward_hook(self._store_activations),
            target_layer.register_full_backward_hook(self._store_gradients),
        ]

    @staticmethod
    def _target_layer(model: nn.Module) -> nn.Module:
        backbone = getattr(model, "efficientnet", None)
        if backbone is None:
            raise ValueError(
                f"GradCAM expects an EfficientNetForImageClassification model, got {type(model).__name__}"
            )
        return backbone.encoder.top_activation

    def _store_activations(self, module: nn.Module, inputs, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _store_gradients(self, module: nn.Module, grad_input, grad_output: tuple[torch.Tensor, ...]) -> None:
        self._gradients = grad_output[0].detach()

    def __call__(self, pixel_values: torch.Tensor, target_class: torch.Tensor) -> CAMOutput:
        was_training = self.model.training
        self.model.eval()
        pixel_values = pixel_values.clone().requires_grad_(True)

        logits = self.model(pixel_values=pixel_values).logits
        self.model.zero_grad(set_to_none=True)
        logits.gather(1, target_class.unsqueeze(1)).sum().backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

        self.model.train(was_training)
        return CAMOutput(cam=self._normalize(cam), target_class=target_class, logits=logits.detach())

    @staticmethod
    def _normalize(cam: torch.Tensor) -> torch.Tensor:
        flat_min = cam.flatten(1).min(dim=1).values.view(-1, 1, 1)
        flat_max = cam.flatten(1).max(dim=1).values.view(-1, 1, 1)
        return (cam - flat_min) / (flat_max - flat_min).clamp_min(1e-8)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
