"""Plots mean deletion and insertion curves from a saved run, overlaying both
models. Curves are macro-averaged over the 7 classes (mean of per-class mean
curves) to match the per-class AUC analysis, so the majority class cannot
dominate; the shaded band is +/- 1 std across the per-class means. The
random-ordering control is drawn dashed on every panel.

Usage:
    python scripts/plot_faithfulness.py \
        --results EfficientNet-B0=results/faithfulness_efficientnet \
                  ViT-B/16=results/faithfulness_vit \
        --output results/faithfulness_curves.pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dermai.data import CLASSES
from dermai.faithfulness import DeletionInsertion, load_run

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def macro_curve(curves: np.ndarray, target_classes: np.ndarray):
    class_curves = np.stack([curves[target_classes == c].mean(axis=0)
                             for c in range(len(CLASSES)) if (target_classes == c).any()])
    return class_curves.mean(axis=0), class_curves.std(axis=0)


def draw(ax, fractions, curves, target_classes, label, color, dashed=False):
    mean, std = macro_curve(curves, target_classes)
    auc = DeletionInsertion._auc(mean)
    ax.plot(fractions, mean, color=color, lw=2, linestyle="--" if dashed else "-",
            label=f"{label} (AUC {auc:.3f})")
    if not dashed:
        ax.fill_between(fractions, mean - std, mean + std, color=color, alpha=0.15)


def panel(ax, key, title, xlabel, entries):
    for color, (label, (target_classes, data)) in zip(COLORS, entries.items()):
        draw(ax, data["fractions"], data[key], target_classes, label, color)
        random_key = f"{key}_random"
        if random_key in data:
            draw(ax, data["fractions"], data[random_key], target_classes, f"{label} random", color, dashed=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Macro-averaged P(predicted class)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot deletion/insertion faithfulness curves")
    parser.add_argument("--results", nargs="+", required=True, help="LABEL=path-prefix entries (no extension)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    entries = {item.split("=", 1)[0]: load_run(item.split("=", 1)[1]) for item in args.results}

    fig, (deletion_ax, insertion_ax) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    panel(deletion_ax, "deletion", "Deletion (lower AUC is more faithful)", "Fraction of pixels removed", entries)
    panel(insertion_ax, "insertion", "Insertion (higher AUC is more faithful)", "Fraction of pixels added back", entries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
