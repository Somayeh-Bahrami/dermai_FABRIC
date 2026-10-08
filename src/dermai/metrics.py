from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score


class ClassificationMetrics:
    def __init__(self) -> None:
        self.predictions: list[int] = []
        self.targets: list[int] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        self.predictions.extend(logits.argmax(dim=1).cpu().tolist())
        self.targets.extend(targets.cpu().tolist())

    def compute(self) -> dict[str, float]:
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        return {
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        }
