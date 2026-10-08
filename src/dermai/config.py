from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Phase:
    name: str
    epochs: int
    lr: float
    unfreeze_depth: int | str = 0  # 0 = classifier only, "all" = full fine-tune, int = last N blocks

    def __post_init__(self) -> None:
        if isinstance(self.unfreeze_depth, str) and self.unfreeze_depth != "all":
            raise ValueError(f"unfreeze_depth string values must be 'all', got {self.unfreeze_depth!r}")
        if isinstance(self.unfreeze_depth, int) and self.unfreeze_depth < 0:
            raise ValueError(f"unfreeze_depth must be >= 0, got {self.unfreeze_depth}")

    @classmethod
    def from_dict(cls, raw: dict) -> "Phase":
        # Backward compatibility: older configs used a boolean unfreeze_backbone
        # (True = fully unfrozen, False = classifier-only frozen backbone).
        raw = dict(raw)
        if "unfreeze_backbone" in raw and "unfreeze_depth" not in raw:
            legacy = raw.pop("unfreeze_backbone")
            raw["unfreeze_depth"] = "all" if legacy else 0
        return cls(**raw)

@dataclass(frozen=True)
class Config:
    model_id: str
    phases: list[Phase]
    data_dir: Path
    output_dir: Path
    batch_size: int = 32
    num_workers: int = 0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    device: str = "auto"
    experiment_tag: str | None = None  # e.g. "unfreeze4" -- disambiguates ablation runs on checkpoint_dir
    dropout: float | None = None            # ViT: hidden_dropout_prob | EfficientNet: dropout_rate. None = HF default.
    attention_dropout: float | None = None  # ViT only: attention_probs_dropout_prob. Ignored for EfficientNet.
    drop_connect_rate: float | None = None  # EfficientNet only: stochastic depth inside MBConv blocks. Ignored for ViT.
    augment: bool = False  # apply train-time image augmentation (flip/rotate/color-jitter). Train split only.
    class_weighted_loss: bool = True  # weight CrossEntropyLoss by inverse class frequency. False = plain unweighted CE.

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text())
        phases = [Phase.from_dict(p) for p in raw.pop("phases")]
        raw["data_dir"] = Path(raw["data_dir"])
        raw["output_dir"] = Path(raw["output_dir"])
        return cls(phases=phases, **raw)

    @property
    def run_name(self) -> str:
        base = self.model_id.split("/")[-1]
        if self.experiment_tag:
            return f"{base}__{self.experiment_tag}"
        # Fall back to auto-deriving a tag from the last phase's unfreeze_depth
        # so ablation configs still get distinct checkpoint dirs even if you
        # forget to set experiment_tag explicitly.
        last_depth = self.phases[-1].unfreeze_depth

        return f"{base}__depth-{last_depth}"