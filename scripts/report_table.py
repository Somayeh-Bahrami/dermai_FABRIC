"""Builds a single report-ready Markdown table of faithfulness results from a
saved run, per class and macro-averaged, showing both the raw RISE AUC (Petsiuk
et al., 2018) and the AOPC relative to the random-ordering control (Samek et al.,
2017).

Usage:
    python scripts/report_table.py \
        --results EfficientNet-B0=results/faithfulness_efficientnet \
                  ViT-B/16=results/faithfulness_vit \
        --title "deletion=blur, insertion=mean" \
        --output results/faithfulness_table.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dermai.data import CLASSES
from dermai.faithfulness import aopc_aucs, load_run, macro_average, raw_aucs

HEADER = "| Model | Class | n | Deletion AUC (lower) | Deletion AOPC-random (+faithful) | Insertion AUC (higher) | Insertion AOPC-random (+faithful) |"
DIVIDER = "|---|---|---:|---:|---:|---:|---:|"


def build_rows(prefix: str) -> list[dict]:
    target_classes, data = load_run(prefix)
    del_raw, ins_raw = raw_aucs(data["deletion"]), raw_aucs(data["insertion"])
    del_aopc = aopc_aucs(data["deletion"], "deletion") - aopc_aucs(data["deletion_random"], "deletion")
    ins_aopc = aopc_aucs(data["insertion"], "insertion") - aopc_aucs(data["insertion_random"], "insertion")

    def row(label, n, mask):
        return {"label": label, "n": n,
                "del_raw": del_raw[mask].mean(), "del_aopc": del_aopc[mask].mean(),
                "ins_raw": ins_raw[mask].mean(), "ins_aopc": ins_aopc[mask].mean()}

    rows = [row(CLASSES[c], int((target_classes == c).sum()), target_classes == c)
            for c in range(len(CLASSES)) if (target_classes == c).any()]
    rows.append({"label": "**macro**", "n": len(target_classes),
                 "del_raw": macro_average(del_raw, target_classes), "del_aopc": macro_average(del_aopc, target_classes),
                 "ins_raw": macro_average(ins_raw, target_classes), "ins_aopc": macro_average(ins_aopc, target_classes)})
    return rows


def format_table(model_rows: dict[str, list[dict]], title: str) -> str:
    lines = [f"# Faithfulness results ({title})", "",
             "Raw AUC follows RISE (Petsiuk et al., 2018); AOPC-random is the AOPC "
             "(Samek et al., 2017) of the saliency ordering minus the random ordering, "
             "positive means more faithful than chance. Macro is the mean over the 7 classes.",
             "", HEADER, DIVIDER]
    for model, rows in model_rows.items():
        for i, r in enumerate(rows):
            model_cell = f"**{model}**" if i == 0 else ""
            lines.append(f"| {model_cell} | {r['label']} | {r['n']} | {r['del_raw']:.3f} | "
                         f"{r['del_aopc']:+.3f} | {r['ins_raw']:.3f} | {r['ins_aopc']:+.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report-ready faithfulness table")
    parser.add_argument("--results", nargs="+", required=True, help="LABEL=path-prefix entries (no extension)")
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    model_rows = {item.split("=", 1)[0]: build_rows(item.split("=", 1)[1]) for item in args.results}
    table = format_table(model_rows, args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table)
    print(table)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
