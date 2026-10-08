from __future__ import annotations

import logging
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

from .data import CLASSES, LABEL_TO_INDEX
from .utils import get_logger

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logger = get_logger()


class ModelFactory:
    @staticmethod
    def model_type(model_id: str) -> str:
        """Peeks at the HF config's model_type without loading model weights.
        Used to auto-select architecture-dependent defaults (e.g. augmentation)
        from a single shared YAML config where only model_id changes."""
        return AutoConfig.from_pretrained(model_id).model_type

    @staticmethod
    def build(
        model_id: str,
        dropout: float | None = None,
        attention_dropout: float | None = None,
        drop_connect_rate: float | None = None,
    ) -> nn.Module:
        config_kwargs = dict(
            num_labels=len(CLASSES),
            id2label={i: name for name, i in LABEL_TO_INDEX.items()},
            label2id=LABEL_TO_INDEX,
            ignore_mismatched_sizes=True,
        )
        if dropout is not None or attention_dropout is not None or drop_connect_rate is not None:
            config_kwargs.update(
                ModelFactory._dropout_config_kwargs(
                    model_id, dropout, attention_dropout, drop_connect_rate
                )
            )
        return AutoModelForImageClassification.from_pretrained(model_id, **config_kwargs)

    @staticmethod
    def _dropout_config_kwargs(
        model_id: str,
        dropout: float | None,
        attention_dropout: float | None,
        drop_connect_rate: float | None,
    ) -> dict:
        model_type = AutoConfig.from_pretrained(model_id).model_type
        kwargs: dict = {}

        if model_type == "vit":
            if dropout is not None:
                kwargs["hidden_dropout_prob"] = dropout
            if attention_dropout is not None:
                kwargs["attention_probs_dropout_prob"] = attention_dropout
            if drop_connect_rate is not None:
                logger.warning(
                    "drop_connect_rate=%.3f requested but model_type=%r has no MBConv "
                    "blocks (no stochastic depth mechanism); ignoring.",
                    drop_connect_rate, model_type
                )
        elif model_type == "efficientnet":
            if dropout is not None:
                kwargs["dropout_rate"] = dropout
            if drop_connect_rate is not None:
                kwargs["drop_connect_rate"] = drop_connect_rate
            if attention_dropout is not None:
                logger.warning(
                    "attention_dropout=%.3f requested but model_type=%r has no attention "
                    "mechanism; ignoring.", attention_dropout, model_type
                )
        else:
            raise TypeError(
                f"dropout overrides are not implemented for model_type={model_type!r}; "
                "add a branch in ModelFactory._dropout_config_kwargs for this architecture."
            )
        return kwargs

    @staticmethod
    def load(checkpoint: str) -> nn.Module:
        return AutoModelForImageClassification.from_pretrained(checkpoint)

    @staticmethod
    def processor(model_id: str):
        return AutoImageProcessor.from_pretrained(model_id)

    @staticmethod
    def set_unfreeze_depth(model: nn.Module, depth: int | str) -> None:
        """Freezes the whole backbone, then selectively re-enables gradients.

        depth:
            0       -> classifier head only, backbone fully frozen
            "all"   -> full fine-tuning, every parameter trainable
            int > 0 -> classifier head + the last `depth` backbone blocks
                       (ViT encoder layers, or EfficientNet MBConv blocks)

        Always safe to call between phases/epochs: it resets requires_grad
        from scratch each time rather than incrementally toggling.
        """
        for parameter in model.parameters():
            parameter.requires_grad = False
        for name, parameter in model.named_parameters():
            if name.startswith("classifier"):
                parameter.requires_grad = True

        if depth == "all":
            for parameter in model.parameters():
                parameter.requires_grad = True
            return
        if depth == 0:
            return

        blocks = ModelFactory._backbone_blocks(model)
        total = len(blocks)
        if not isinstance(depth, int) or depth < 0:
            raise ValueError(f"depth must be 0, a positive int, or 'all', got {depth!r}")
        if depth > total:
            raise ValueError(
                f"requested unfreeze depth {depth} exceeds available blocks ({total}) "
                f"for model_type={model.config.model_type!r}"
            )
        for block in blocks[total - depth:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

        top = ModelFactory._pre_head_layer(model)
        if top is not None:
            for parameter in top.parameters():
                parameter.requires_grad = True

    @staticmethod
    def freeze_norm_statistics(model: nn.Module) -> None:
        """requires_grad=False leaves batch-norm running statistics updating, so hold them in eval mode too."""
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    @staticmethod
    def _backbone_blocks(model: nn.Module):
        model_type = getattr(model.config, "model_type", None)
        if model_type == "vit":
            # transformers < 5.x nested layers under an explicit `encoder` module;
            # transformers >= 5.13 flattened ViTModel to expose `layers` directly.
            if hasattr(model.vit, "encoder"):
                return model.vit.encoder.layer
            return model.vit.layers
        if model_type == "efficientnet":
            if hasattr(model.efficientnet, "encoder"):
                return model.efficientnet.encoder.blocks
            return model.efficientnet.blocks
        raise TypeError(
            f"unfreeze-by-depth is not implemented for model_type={model_type!r}; "
            "add a branch in ModelFactory._backbone_blocks for this architecture. "
            "Run inspect_model.py to see the actual attribute layout for your installed "
            "transformers version before guessing at the path."
        )

    @staticmethod
    def _pre_head_layer(model: nn.Module):
        model_type = getattr(model.config, "model_type", None)
        if model_type == "vit":
            return model.vit.layernorm
        if model_type == "efficientnet":
            return model.efficientnet.encoder.top_activation
        return None

    @staticmethod
    def describe_dropout(model: nn.Module) -> dict[str, float | None]:
        model_type = getattr(model.config, "model_type", None)
        if model_type == "vit":
            return {
                "model_type": model_type,
                "hidden_dropout_prob": getattr(model.config, "hidden_dropout_prob", None),
                "attention_probs_dropout_prob": getattr(
                    model.config, "attention_probs_dropout_prob", None
                ),
            }
        if model_type == "efficientnet":
            return {
                "model_type": model_type,
                "dropout_rate": getattr(model.config, "dropout_rate", None),
                "drop_connect_rate": getattr(model.config, "drop_connect_rate", None),
            }
        return {"model_type": model_type}

    @staticmethod
    def describe_backbone(model: nn.Module) -> dict[str, int]:
        blocks = ModelFactory._backbone_blocks(model)
        return {"model_type": model.config.model_type, "num_blocks": len(blocks)}