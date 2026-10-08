"""Aggregates results/final/<model>_s<seed>/ runs into paper tables and PROVENANCE.md.

    python scripts/aggregate.py                      # real runs only
    python scripts/aggregate.py --runs-root <dir> --allow-smoke --out <dir>   # smoke test

Seeds: mean +/- SD (sample SD, ddof=1) over training seeds of each run-level metric.
Images: paired bootstrap 95% CIs (10,000 draws, percentile) for differences between
two conditions on the same test images; each image's value is first averaged over
seeds. Faithfulness contrasts use the plain mean over images (the tables use the
macro average over predicted classes, as in the paper); this is stated in the output.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"efficientnet": "EfficientNet-B0", "vit": "ViT-B/16"}
NATIVE = {"efficientnet": "Grad-CAM", "vit": "Attention rollout"}
FAITH_ORDERINGS = {"native": None, "ig": "Integrated Gradients", "random": "Random order (control)",
                   "center": "Center prior (control)"}
LOC_SOURCES = {"native": None, "ig": "Integrated Gradients", "center": "Center prior (no model)"}
DRAWS = 10_000


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_runs(runs_root: Path, allow_smoke: bool) -> dict[str, list[Path]]:
    runs: dict[str, list[Path]] = defaultdict(list)
    for summary_path in sorted(runs_root.glob("*_s*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if summary["smoke_test"] and not allow_smoke:
            continue
        runs[summary["model"]].append(summary_path.parent)
    return runs


def per_image(run_dirs: list[Path], filename: str, column: str) -> pd.Series:
    """Per-image value averaged over seeds; asserts every seed covers the same images."""
    frames = [pd.read_csv(d / filename).set_index("image_id")[column].astype(float) for d in run_dirs]
    for frame in frames[1:]:
        assert sorted(frame.index) == sorted(frames[0].index), f"{filename}: image sets differ across seeds"
    return pd.concat(frames, axis=1).mean(axis=1).sort_index()


def seed_values(run_dirs: list[Path], getter) -> np.ndarray:
    return np.array([getter(json.loads((d / "summary.json").read_text())) for d in run_dirs], dtype=float)


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def mean_sd(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def paired_bootstrap(a: pd.Series, b: pd.Series, rng: np.random.Generator) -> dict:
    assert list(a.index) == list(b.index), "paired bootstrap needs the same images in the same order"
    diff = (a - b).to_numpy()
    idx = rng.integers(0, len(diff), size=(DRAWS, len(diff)))
    boot = diff[idx].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"n_images": len(diff), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "diff": float(diff.mean()), "ci_low": float(low), "ci_high": float(high)}


def fmt(mean: float, sd: float, digits: int = 3) -> str:
    if np.isnan(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def latex_table(header: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    cols = "l" * 2 + "c" * (len(header) - 2)
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "results" / "final")
    parser.add_argument("--out", type=Path, default=None, help="defaults to <runs-root>/tables")
    parser.add_argument("--provenance", type=Path, default=ROOT / "results" / "PROVENANCE.md")
    parser.add_argument("--paper-tables", default=str(ROOT / "paper" / "tables"),
                        help="also copy the .tex tables here (empty string to skip)")
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    out = args.out or args.runs_root / "tables"
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.runs_root, args.allow_smoke)
    if not runs:
        raise SystemExit(f"no runs found under {args.runs_root}")
    rng = np.random.default_rng(0)
    report: dict = {"seeds": {m: [json.loads((d / "summary.json").read_text())["training_seed"] for d in ds]
                              for m, ds in runs.items()}}
    seeds_note = ", ".join(f"{MODELS[m]} n={len(ds)}" for m, ds in runs.items())

    # Classification
    rows, report["classification"] = [], {}
    for model, dirs in runs.items():
        entry = {key: mean_sd(seed_values(dirs, lambda s, k=key: s["classification"][k]))
                 for key in ("val_macro_f1_selected", "test_macro_f1", "test_balanced_accuracy")}
        report["classification"][model] = entry
        rows.append([MODELS[model], str(len(dirs))] + [fmt(*entry[k]) for k in entry])
    (out / "classification.tex").write_text(latex_table(
        ["Model", "Seeds", "Val macro-F1 (selection)", "Test macro-F1", "Test bal. acc."], rows,
        f"Classification, mean $\\pm$ SD over training seeds ({seeds_note}); fixed split, "
        "checkpoint selected on validation macro-F1 only.", "tab:classification"))

    # Faithfulness (macro over predicted classes, as in the paper)
    rows, report["faithfulness"] = [], {}
    for model, dirs in runs.items():
        for ordering, name in FAITH_ORDERINGS.items():
            entry = {k: mean_sd(seed_values(dirs, lambda s, k=k, o=ordering: s["faithfulness"][o][k]))
                     for k in ("deletion_auc_macro", "insertion_auc_macro")}
            report["faithfulness"][f"{model}/{ordering}"] = entry
            rows.append([MODELS[model], name or NATIVE[model], fmt(*entry["deletion_auc_macro"]),
                         fmt(*entry["insertion_auc_macro"])])
    (out / "faithfulness.tex").write_text(latex_table(
        ["Model", "Ordering", "Deletion AUC $\\downarrow$", "Insertion AUC $\\uparrow$"], rows,
        "Deletion/insertion AUC of P(predicted class), macro-averaged over predicted classes; mean $\\pm$ SD "
        "over training seeds. Random and center-prior orderings use no explanation.", "tab:faithfulness"))

    # Localization
    rows, report["localization"] = [], {}
    for model, dirs in runs.items():
        for source, name in LOC_SOURCES.items():
            entry = {k: mean_sd(seed_values(dirs, lambda s, k=k, src=source: s["localization"][src][k]))
                     for k in ("iou_mean", "pointing_tol0", "pointing_tol15")}
            report["localization"][f"{model}/{source}"] = entry
            label = name or NATIVE[model]
            rows.append([MODELS[model], label] + [fmt(*entry[k]) for k in entry])
    (out / "localization.tex").write_text(latex_table(
        ["Model", "Heatmap", "IoU (top 20\\%)", "PG exact", "PG 15 px"], rows,
        "Localization against lesion masks (224$\\times$224, nearest-neighbor resize). PG exact: peak pixel "
        "inside the mask (primary); PG 15 px: within 15 px (secondary). Mean $\\pm$ SD over training seeds; "
        "the center prior is model-free, so its SD is zero.", "tab:localization"))

    # Paired image-level contrasts
    contrasts = []

    def add(name: str, a: pd.Series, b: pd.Series) -> None:
        contrasts.append({"contrast": name, **paired_bootstrap(a, b, rng)})

    for model, dirs in runs.items():
        for metric in ("iou", "pointing_hit_tol0"):
            center = per_image(dirs, "localization_center.csv", metric)
            add(f"{MODELS[model]} {NATIVE[model]} - center prior, {metric}",
                per_image(dirs, "localization_native.csv", metric), center)
            add(f"{MODELS[model]} IG - center prior, {metric}", per_image(dirs, "localization_ig.csv", metric), center)
        for curve in ("deletion_auc", "insertion_auc"):
            native = per_image(dirs, "faithfulness_native.csv", curve)
            ig = per_image(dirs, "faithfulness_ig.csv", curve)
            for control in ("random", "center"):
                ref = per_image(dirs, f"faithfulness_{control}.csv", curve)
                add(f"{MODELS[model]} {NATIVE[model]} - {control}, {curve}", native, ref)
                add(f"{MODELS[model]} IG - {control}, {curve}", ig, ref)
    if {"efficientnet", "vit"} <= set(runs):
        for filename, metric in (("localization_ig.csv", "iou"), ("localization_ig.csv", "pointing_hit_tol0"),
                                 ("faithfulness_ig.csv", "deletion_auc"), ("faithfulness_ig.csv", "insertion_auc")):
            add(f"IG EfficientNet - IG ViT, {metric}", per_image(runs["efficientnet"], filename, metric),
                per_image(runs["vit"], filename, metric))
    with (out / "contrasts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrasts[0]))
        writer.writeheader()
        writer.writerows(contrasts)
    rows = [[c["contrast"].split(",")[0], c["contrast"].split(", ")[1].replace("_", "\\_"),
             f"{c['diff']:+.3f}", f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"] for c in contrasts]
    (out / "contrasts.tex").write_text(latex_table(
        ["Contrast", "Metric", "Mean diff.", "95\\% CI"], rows,
        f"Paired differences over the same test images (per-image values averaged over seeds; plain mean over "
        f"images), paired bootstrap with {DRAWS:,} draws, percentile 95\\% CI.", "tab:contrasts"))
    report["contrasts"] = contrasts
    (out / "aggregate.json").write_text(json.dumps(report, indent=2))

    if args.paper_tables:
        paper_tables = Path(args.paper_tables)
        paper_tables.mkdir(parents=True, exist_ok=True)
        for table in out.glob("*.tex"):
            (paper_tables / table.name).write_text(table.read_text())
    write_provenance(args.provenance, runs, out)
    print(f"tables in {out}; provenance in {args.provenance}")


def write_provenance(path: Path, runs: dict[str, list[Path]], tables: Path) -> None:
    def rel(p: Path) -> str:
        return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)

    lines = ["# Provenance of every number in the paper", "",
             "Generated by `scripts/aggregate.py`. Do not edit the run table by hand.", "",
             "## Final runs (`scripts/run_seed.py`)", "",
             "| Run | Model id | Train seed | Split seed | Git commit | Checkpoint sha256 | Config | Smoke |",
             "|---|---|---|---|---|---|---|---|"]
    for model, dirs in runs.items():
        for d in dirs:
            s = json.loads((d / "summary.json").read_text())
            lines.append(f"| `{rel(d)}` | {s['model_id']} | {s['training_seed']} | {s['split_seed']} | "
                         f"`{s['git_commit'][:12]}` | `{s['checkpoint_sha256'][:16]}` | `{rel(d / 'config_used.yaml')}` | "
                         f"{s['smoke_test']} |")
    lines += ["", "Per-run files: `classification.json` (classification), `faithfulness_{native,ig,random,center}.csv` "
              "(faithfulness), `localization_{native,ig,center}.csv` (localization), `summary.json` (all).", "",
              "## Aggregated tables", "",
              f"`{rel(tables)}/{{classification,faithfulness,localization,contrasts}}.tex`, `aggregate.json`, "
              "`contrasts.csv`.", "",
              "## Other sources", "",
              "- `results/localization_center_prior.csv`: `scripts/center_prior_baseline.py` on the 1,002 ids of "
              "`results/localization_efficientnet_best.csv` (must match `localization_center.csv` of any run).",
              "- Legacy single-seed files in `results/*.csv` (Sep 21 paper): reference only, not cited in the paper.",
              "- ViT phase-1 lr 5.0e-2 (`configs/vit.yaml`): from team repo commit 4b5a8f8 (Jul 30). Confirmed only "
              "if the ViT seed-42 rerun reproduces the paper's 0.748 test macro-F1 within seed noise. TODO: record "
              "outcome here.", ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
