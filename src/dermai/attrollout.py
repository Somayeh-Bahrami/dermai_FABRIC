from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .utils import get_logger

logger = get_logger()


@dataclass
class RolloutConfig:
    """Design knobs for attention rollout. Defaults match the common
    reference implementations (Abnar & Zuidema 2020; vit-explain)."""

    head_fusion: str = "mean"      # "mean" | "max" | "min"
    discard_ratio: float = 0.0     # zero out lowest-attention entries per row before rollout, 0.0 = disabled
    residual_alpha: float = 0.5    # weight on identity in the augmented matrix; not exposed as "off"


@dataclass
class RolloutOutput:

    heatmap: torch.Tensor       # (batch, height, width), normalized to [0, 1]
    target_class: torch.Tensor  # (batch,)
    logits: torch.Tensor        # (batch, num_classes)


class AttentionRollout:
    """Attention rollout (Abnar & Zuidema, 2020) for any HF ViT-style model that
    exposes per-layer attention weights via output_attentions=True.
    """
    def __init__(self, model: torch.nn.Module, config: RolloutConfig | None = None) -> None:
        self.model = model
        self.config = config or RolloutConfig()
        self._validate_model()
        # Some model configs have output_attentions disabled globally, in which case
        # the runtime kwarg alone may yield an empty attentions tuple. Set it on the
        # config too so attention weights are actually returned.
        self.model.config.output_attentions = True

    def _validate_model(self) -> None:
        # Fail fast and loud rather than silently returning garbage if someone
        # points this at a non-ViT checkpoint or a wrapper that swallows kwargs.
        if not hasattr(self.model.config, "num_attention_heads"):
            raise TypeError(
                f"{type(self.model).__name__} does not look like a ViT-style model "
                "(missing num_attention_heads in config)."
            )

    @torch.no_grad()
    def __call__(self, pixel_values: torch.Tensor, target_class: torch.Tensor | None = None) -> RolloutOutput:
        """pixel_values: (B, 3, H, W) already preprocessed for the model.
        """
        was_training = self.model.training
        self.model.eval()
        device = next(self.model.parameters()).device
        pixel_values = pixel_values.to(device)

        outputs = self.model(pixel_values=pixel_values, output_attentions=True)
        attentions = outputs.attentions  # tuple of (B, num_heads, N, N), len = num_layers
        logits = outputs.logits

        if not attentions:
            raise RuntimeError(
                "model returned no attention weights (outputs.attentions is empty). "
                "The model is not surfacing attention maps -- attention rollout cannot run. "
                "This usually means the checkpoint's model class does not expose "
                "output_attentions, or attention is computed by a fused/optimized kernel "
                "(e.g. SDPA) that doesn't return the attention matrix. See notes below."
            )

        if target_class is None:
            target_class = logits.argmax(dim=1)

        cls_weights = self._rollout(attentions)                              # (B, N-1)
        heatmap = self._to_spatial(cls_weights, pixel_values.shape[-2:])      # (B, H, W)
        self.model.train(was_training)
        return RolloutOutput(heatmap=heatmap, target_class=target_class, logits=logits.detach())

    def _fuse_heads(self, layer_attention: torch.Tensor) -> torch.Tensor:
        """layer_attention: (B, num_heads, N, N) -> (B, N, N)"""
        if self.config.head_fusion == "mean":
            return layer_attention.mean(dim=1)
        if self.config.head_fusion == "max":
            return layer_attention.max(dim=1).values
        if self.config.head_fusion == "min":
            return layer_attention.min(dim=1).values
        raise ValueError(f"Unknown head_fusion: {self.config.head_fusion}")

    def _apply_discard(self, fused: torch.Tensor) -> torch.Tensor:
        """Zero out the globally lowest discard_ratio fraction of attention
        entries, protecting the CLS self entry so the class token is never
        dropped. No-op when discard_ratio == 0. Rows are renormalized after
        residual augmentation in _rollout, matching Abnar & Zuidema."""
        if self.config.discard_ratio <= 0.0:
            return fused
        B, N, _ = fused.shape
        flat = fused.reshape(B, N * N)
        k = int(N * N * self.config.discard_ratio)
        if k == 0:
            return fused
        _, idx = flat.topk(k, dim=-1, largest=False)
        discard = torch.zeros_like(flat, dtype=torch.bool).scatter_(-1, idx, True)
        discard[:, 0] = False
        return flat.masked_fill(discard, 0.0).view(B, N, N)

    def _rollout(self, attentions: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Fuses heads, applies residual augmentation, and multiplies attention
        matrices layer by layer. Returns the CLS row (excluding CLS-to-CLS) of
        the final rolled-out matrix: (B, N-1)."""
        B, _, N, _ = attentions[0].shape
        device = attentions[0].device
        result = torch.eye(N, device=device).unsqueeze(0).expand(B, N, N).clone()

        alpha = self.config.residual_alpha
        identity = torch.eye(N, device=device).unsqueeze(0)

        for layer_attention in attentions:
            fused = self._fuse_heads(layer_attention)          # (B, N, N)
            fused = self._apply_discard(fused)
            augmented = alpha * fused + (1 - alpha) * identity  # residual term, not optional
            augmented = augmented / augmented.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            result = torch.bmm(augmented, result)

        cls_row = result[:, 0, 1:]  # drop CLS-to-CLS entry, keep CLS-to-patch weights
        return cls_row

    def _to_spatial(self, cls_weights: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        """cls_weights: (B, num_patches) -> (B, H, W), upsampled and normalized per-image."""
        B, num_patches = cls_weights.shape
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise ValueError(
                f"num_patches={num_patches} is not a perfect square (grid_size={grid_size}); "
                "check patch size / image size assumptions."
            )
        grid = cls_weights.view(B, 1, grid_size, grid_size)
        upsampled = F.interpolate(grid, size=output_size, mode="bilinear", align_corners=False)
        upsampled = upsampled.squeeze(1)  # (B, H, W)

        flat_min = upsampled.flatten(1).min(dim=1).values.view(B, 1, 1)
        flat_max = upsampled.flatten(1).max(dim=1).values.view(B, 1, 1)
        normalized = (upsampled - flat_min) / (flat_max - flat_min).clamp_min(1e-8)
        return normalized