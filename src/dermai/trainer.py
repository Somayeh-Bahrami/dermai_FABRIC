from __future__ import annotations

import csv

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from .config import Config, Phase
from .data import DataModule
from .metrics import ClassificationMetrics
from .models import ModelFactory
from .utils import Timer, get_logger

logger = get_logger()

ABLATION_LOG_FIELDS = [
    "run_name", "model_id", "phase", "epoch", "unfreeze_depth",
    "trainable_params", "total_params", "trainable_pct",
    "lr", "batch_size", "weight_decay", "warmup_ratio",
    "dropout", "attention_dropout", "drop_connect_rate", "augment", "class_weighted_loss",
    "train_loss", "train_macro_f1", "train_balanced_accuracy",
    "val_macro_f1", "val_balanced_accuracy",
    "test_macro_f1", "test_balanced_accuracy",
]


class Trainer:
    def __init__(self, model: nn.Module, data: DataModule, config: Config, device: torch.device) -> None:
        self.model = model.to(device)
        self.data = data
        self.config = config
        self.device = device
        self.use_amp = device.type == "cuda"
        class_weights = data.class_weights().to(device) if config.class_weighted_loss else None
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.train_loader = data.loader("train")
        self.val_loader = data.loader("val")
        self.test_loader = data.loader("test")
        self.best_metric = 0.0
        self.checkpoint_dir = config.output_dir / config.run_name
        self._current_depth: int | str = 0
        self._current_param_counts: tuple[int, int] = (0, 0)
        self._current_lr: float = 0.0

    def fit(self) -> None:
        run_timer = Timer()
        for phase in self.config.phases:
            self._run_phase(phase)
        logger.info("training complete in %s, best val macro-F1 %.4f", Timer.format(run_timer.elapsed()), self.best_metric)

    def test(self) -> dict[str, float]:
        self.model = type(self.model).from_pretrained(self.checkpoint_dir).to(self.device)
        metrics = self.evaluate(self.test_loader)
        logger.info("test  macro_f1 %.4f  bal_acc %.4f", metrics["macro_f1"], metrics["balanced_accuracy"])
        self._log_test_row(metrics)
        return metrics

    def _run_phase(self, phase: Phase) -> None:
        ModelFactory.set_unfreeze_depth(self.model, phase.unfreeze_depth)
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self._current_depth = phase.unfreeze_depth
        self._current_param_counts = (trainable, total)
        self._current_lr = phase.lr
        logger.info("phase '%s': %d epochs, lr %.0e, unfreeze_depth=%s, %d/%d trainable params (%.1f%%)",
                    phase.name, phase.epochs, phase.lr, phase.unfreeze_depth, trainable, total, 100 * trainable / total)
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                          lr=phase.lr, weight_decay=self.config.weight_decay)
        steps = phase.epochs * len(self.train_loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(self.config.warmup_ratio * steps), steps)
        for epoch in range(1, phase.epochs + 1):
            self._train_epoch(optimizer, scheduler, phase, epoch)

    def _train_epoch(self, optimizer, scheduler, phase: Phase, epoch: int) -> None:
        self.model.train()
        if phase.unfreeze_depth == 0:
            # Fully-frozen-backbone phase: requires_grad=False alone doesn't stop
            # BatchNorm running statistics from updating, so hold them in eval
            # mode too -- otherwise "frozen" backbone stats still drift.
            ModelFactory.freeze_norm_statistics(self.model)
        timer = Timer()
        running_loss = 0.0
        for batch in self.train_loader:
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            optimizer.zero_grad()
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(pixel_values=pixel_values).logits
                loss = self.loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        train_loss = running_loss / len(self.train_loader)
        # Extra eval-mode pass over the training set, so train_macro_f1/
        # train_balanced_accuracy are computed the same way (dropout off) as
        # the val/test metrics, giving a genuinely comparable train-val gap.
        train_metrics = self.evaluate(self.train_loader)
        val_metrics = self.evaluate(self.val_loader)
        logger.info("[%s] epoch %d  train_loss %.4f  train_macro_f1 %.4f  val_macro_f1 %.4f  val_bal_acc %.4f  (%s)",
                    phase.name, epoch, train_loss, train_metrics["macro_f1"],
                    val_metrics["macro_f1"], val_metrics["balanced_accuracy"], Timer.format(timer.elapsed()))
        self._log_ablation_row(phase.name, epoch, train_loss, train_metrics, val_metrics)
        self._save_if_best(val_metrics["macro_f1"])

    def _log_ablation_row(self, phase_name: str, epoch: int, train_loss: float,
                           train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
        """Appends one row per epoch to a shared CSV across all runs, so the
        ablation table (accuracy/F1 vs. unfreeze depth) can be built directly
        from this file rather than re-parsing console logs."""
        trainable, total = self._current_param_counts
        self._append_ablation_row({
            "run_name": self.config.run_name,
            "model_id": self.config.model_id,
            "phase": phase_name,
            "epoch": epoch,
            "unfreeze_depth": self._current_depth,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(100 * trainable / total, 2) if total else 0.0,
            "lr": self._current_lr,
            "batch_size": self.config.batch_size,
            "weight_decay": self.config.weight_decay,
            "warmup_ratio": self.config.warmup_ratio,
            "dropout": self.config.dropout,
            "attention_dropout": self.config.attention_dropout,
            "drop_connect_rate": self.config.drop_connect_rate,
            "augment": self.config.augment,
            "class_weighted_loss": self.config.class_weighted_loss,
            "train_loss": round(train_loss, 6),
            "train_macro_f1": round(train_metrics["macro_f1"], 6),
            "train_balanced_accuracy": round(train_metrics["balanced_accuracy"], 6),
            "val_macro_f1": round(val_metrics["macro_f1"], 6),
            "val_balanced_accuracy": round(val_metrics["balanced_accuracy"], 6),
        })

    def _log_test_row(self, metrics: dict[str, float]) -> None:
        """Appends one final row (phase='test') with test-set metrics, once
        per run -- after test() reloads the best checkpoint from this run.
        Train/val columns are left blank since this row isn't tied to a
        single training epoch."""
        trainable, total = self._current_param_counts
        self._append_ablation_row({
            "run_name": self.config.run_name,
            "model_id": self.config.model_id,
            "phase": "test",
            "epoch": "",
            "unfreeze_depth": self._current_depth,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(100 * trainable / total, 2) if total else 0.0,
            "lr": self._current_lr,
            "batch_size": self.config.batch_size,
            "weight_decay": self.config.weight_decay,
            "warmup_ratio": self.config.warmup_ratio,
            "dropout": self.config.dropout,
            "attention_dropout": self.config.attention_dropout,
            "drop_connect_rate": self.config.drop_connect_rate,
            "augment": self.config.augment,
            "class_weighted_loss": self.config.class_weighted_loss,
            "test_macro_f1": round(metrics["macro_f1"], 6),
            "test_balanced_accuracy": round(metrics["balanced_accuracy"], 6),
        })

    def _append_ablation_row(self, row: dict) -> None:
        log_path = self.config.output_dir / "ablation_log.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ABLATION_LOG_FIELDS, restval="")
            if is_new:
                writer.writeheader()
            writer.writerow(row)

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        self.model.eval()
        metrics = ClassificationMetrics()
        for batch in loader:
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            logits = self.model(pixel_values=pixel_values).logits
            metrics.update(logits, labels)
        return metrics.compute()

    def _save_if_best(self, metric: float) -> None:
        if metric <= self.best_metric:
            return
        self.best_metric = metric
        self.model.save_pretrained(self.checkpoint_dir)
        self.data.processor.save_pretrained(self.checkpoint_dir)