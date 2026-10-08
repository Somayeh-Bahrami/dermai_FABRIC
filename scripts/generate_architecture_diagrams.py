from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

FILL_CONV = "#cfe2f3"
FILL_ATTENTION = "#d9ead3"
FILL_MLP = "#fce5cd"
FILL_NORM = "#d9d2e9"
FILL_NEUTRAL = "#f3f3f3"
FILL_HEAD = "#fff2cc"

BORDER_FROZEN = "#999999"
BORDER_UNFROZEN = "#1155cc"
BORDER_HEAD = "#38761d"
BORDER_NEUTRAL = "#555555"
RED = "#cc0000"


def layout_centers(heights: list[float], gap: float, y_top: float) -> list[float]:
    centers = []
    y = y_top - heights[0] / 2
    centers.append(y)
    for i in range(1, len(heights)):
        y -= heights[i - 1] / 2 + gap + heights[i] / 2
        centers.append(y)
    return centers


def draw_box(ax, cx, cy, width, height, label, fill, border_color, lw=2.2, fontsize=9.5, linestyle="-", zorder=2):
    box = FancyBboxPatch(
        (cx - width / 2, cy - height / 2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw, edgecolor=border_color, facecolor=fill,
        linestyle=linestyle, zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize, zorder=zorder + 1)


def draw_arrow(ax, x, y_top, y_bottom, color="#333333", lw=1.6, style="-|>"):
    ax.annotate("", xy=(x, y_bottom), xytext=(x, y_top),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def draw_note(ax, x, y, text, arrow_to=None, fontsize=9.2, box_edge="#333333", box_fill="#ffffff", text_color="black"):
    ax.text(x, y, text, fontsize=fontsize, color=text_color, ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc=box_fill, ec=box_edge, lw=1.2), zorder=5)
    if arrow_to is not None:
        ax.annotate("", xy=arrow_to, xytext=(x, y),
                    arrowprops=dict(arrowstyle="-", color=box_edge, lw=1.0, linestyle="dashed"), zorder=4)


def draw_role_legend_entries(ax, x, y_start, dy, fill_entries, border_entries, red_note):
    y = y_start
    ax.text(x, y, "Legend", fontsize=12, fontweight="bold")
    y -= dy
    for color, label in border_entries:
        draw_box(ax, x + 0.55, y, 1.0, 0.55, "", "#ffffff", color, lw=2.6, zorder=2)
        ax.text(x + 1.25, y, label, fontsize=9, va="center")
        y -= dy
    y -= dy * 0.25
    for color, label in fill_entries:
        draw_box(ax, x + 0.55, y, 1.0, 0.55, "", color, "#777777", lw=1.2, zorder=2)
        ax.text(x + 1.25, y, label, fontsize=9, va="center")
        y -= dy
    y -= dy * 0.25
    ax.text(x, y, red_note, fontsize=9, color=RED, fontweight="bold", va="center")


def stage_header(ax, cx, y, title, subtitle, color):
    ax.text(cx, y, title, ha="center", va="bottom", fontsize=13, fontweight="bold", color=color)
    ax.text(cx, y - 0.55, subtitle, ha="center", va="bottom", fontsize=9.5, color="#333333")


def build_efficientnet_diagram(path: str) -> None:
    rows = [
        ("Input\n224×224×3", FILL_NEUTRAL, "neutral"),
        ("Stem Conv\n3×3, stride 2", FILL_CONV, "backbone"),
        ("MBConv Stage 1\nk3, ×1 block, 16ch", FILL_CONV, "backbone"),
        ("MBConv Stage 2\nk3, ×2 blocks, 24ch", FILL_CONV, "backbone"),
        ("MBConv Stage 3\nk5, ×2 blocks, 40ch", FILL_CONV, "backbone"),
        ("MBConv Stage 4\nk3, ×3 blocks, 80ch", FILL_CONV, "backbone"),
        ("MBConv Stage 5\nk5, ×3 blocks, 112ch", FILL_CONV, "backbone"),
        ("MBConv Stage 6\nk5, ×4 blocks, 192ch", FILL_CONV, "backbone"),
        ("MBConv Stage 7\nk3, ×1 block, 320ch", FILL_CONV, "backbone"),
        ("Global Average\nPooling", FILL_NORM, "backbone"),
        ("Dropout\np=0.2", FILL_HEAD, "head"),
        ("Linear\n1280 → 7 classes", FILL_HEAD, "head"),
    ]
    heights = [1.0] * len(rows)
    gap = 0.32
    y_top = sum(heights) + gap * (len(rows) - 1)
    centers = layout_centers(heights, gap, y_top)

    col1_x, col2_x = 3.0, 9.4
    box_w = 4.6
    ann_x = 15.6

    fig, ax = plt.subplots(figsize=(19, 21), dpi=150)
    ax.set_xlim(-0.5, 21.5)
    ax.set_ylim(-8.5, y_top + 3.2)
    ax.axis("off")
    ax.set_title("EfficientNet-B0 — Fine-Tuning Architecture (HAM10000, 7-class skin lesion classification)",
                 fontsize=15, fontweight="bold", pad=18)

    stage_header(ax, col1_x, y_top + 1.1, "STAGE 1 — Head-only", "backbone frozen, head trains", BORDER_FROZEN)
    stage_header(ax, col2_x, y_top + 1.1, "STAGE 2 — Full fine-tune", "backbone unfrozen, head keeps training", BORDER_UNFROZEN)

    for col_x, role_border in [(col1_x, BORDER_FROZEN), (col2_x, BORDER_UNFROZEN)]:
        for i, ((label, fill, kind), cy) in enumerate(zip(rows, centers)):
            if kind == "neutral":
                border = BORDER_NEUTRAL
            elif kind == "head":
                border = BORDER_HEAD
            else:
                border = role_border
            draw_box(ax, col_x, cy, box_w, heights[i] * 0.92, label, fill, border, lw=2.4 if kind != "neutral" else 1.4)
            if i > 0:
                prev_h = heights[i - 1]
                draw_arrow(ax, col_x, centers[i - 1] - prev_h * 0.46, cy + heights[i] * 0.46)

    dropout_idx = len(rows) - 2
    linear_idx = len(rows) - 1
    draw_note(ax, ann_x, centers[dropout_idx] + 0.35,
              "Dropout p=0.2 →\nconsider 0.3-0.4\nif overfitting",
              arrow_to=(col2_x + box_w / 2, centers[dropout_idx]))
    draw_note(ax, ann_x, centers[linear_idx] - 0.15,
              "Linear head:\nFreshly initialized\noutput = 7 classes",
              arrow_to=(col2_x + box_w / 2, centers[linear_idx]))
    draw_note(ax, ann_x, y_top - 1.3,
              "Stage 2 LR:\nbackbone = 1e-4\nhead = 1e-3",
              arrow_to=(col2_x + box_w / 2, centers[1]),
              box_edge=BORDER_UNFROZEN)

    fill_entries = [
        (FILL_CONV, "Conv / MBConv block"),
        (FILL_NORM, "Normalization (GAP)"),
        (FILL_HEAD, "Head (Dropout / Linear)"),
    ]
    border_entries = [
        (BORDER_FROZEN, "Frozen (Stage 1)"),
        (BORDER_UNFROZEN, "Unfrozen (Stage 2)"),
        (BORDER_HEAD, "Head — always trainable"),
    ]
    draw_role_legend_entries(ax, 15.2, -0.8, 0.85, fill_entries, border_entries,
                              "Red text = critical fix needed\nbefore next training run")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_transformer_block(ax, cx, cy_top, width, role_border, red_annotation_x=None):
    sub_rows = [
        ("LayerNorm", FILL_NORM, False),
        ("Multi-Head Self-Attention\n12 heads × 64 dim/head", FILL_ATTENTION, False),
        ("Dropout (attn_probs_dropout_prob)", FILL_NEUTRAL, True),
        ("Residual Add  (+)", FILL_NEUTRAL, False),
        ("LayerNorm", FILL_NORM, False),
        ("MLP  768 → 3072 → 768\nGELU activation", FILL_MLP, False),
        ("Dropout (hidden_dropout_prob)", FILL_NEUTRAL, True),
        ("Residual Add  (+)", FILL_NEUTRAL, False),
    ]
    sub_h = 0.62
    sub_gap = 0.14
    inner_w = width - 0.6
    pad = 0.35
    total_inner_h = len(sub_rows) * sub_h + (len(sub_rows) - 1) * sub_gap
    outer_h = total_inner_h + pad * 2

    outer_box = FancyBboxPatch(
        (cx - width / 2, cy_top - outer_h), width, outer_h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=3.0, edgecolor=role_border, facecolor="#fbfbfb", linestyle="-", zorder=1,
    )
    ax.add_patch(outer_box)
    ax.text(cx, cy_top + 0.18, "Transformer Encoder Block  ×12 (identical, repeated)",
            ha="center", va="bottom", fontsize=9.3, fontweight="bold", color="#222222")

    y = cy_top - pad - sub_h / 2
    sub_centers = []
    for label, fill, is_critical in sub_rows:
        border = RED if is_critical else "#666666"
        lw = 2.0 if is_critical else 1.1
        draw_box(ax, cx, y, inner_w, sub_h * 0.9, label, fill, border, lw=lw, fontsize=8.3)
        sub_centers.append((y, is_critical))
        y -= sub_h + sub_gap

    for i in range(len(sub_rows) - 1):
        y_from = sub_centers[i][0] - sub_h * 0.45
        y_to = sub_centers[i + 1][0] + sub_h * 0.45
        draw_arrow(ax, cx, y_from, y_to, lw=1.1)

    if red_annotation_x is not None:
        for y, is_critical in sub_centers:
            if is_critical:
                draw_note(ax, red_annotation_x, y,
                          "CRITICAL: currently 0.0\n→ set to 0.1",
                          arrow_to=(cx + inner_w / 2, y),
                          box_edge=RED, text_color=RED, fontsize=8.6)

    return cy_top - outer_h


def build_vit_diagram(path: str) -> None:
    col1_x, col2_x = 3.2, 10.2
    box_w = 5.2
    gap = 0.34

    fig, ax = plt.subplots(figsize=(22, 27), dpi=150)
    ax.axis("off")
    ax.set_title("ViT-B/16 — Fine-Tuning Architecture (HAM10000, 7-class skin lesion classification)",
                 fontsize=15, fontweight="bold", pad=18)

    top_rows = [
        ("Input\n224×224×3", FILL_NEUTRAL, "neutral", 1.0),
        ("Patch Embedding\n16×16 patches → 196 tokens, dim=768", FILL_CONV, "backbone", 1.0),
        ("CLS Token Prepended\n→ sequence length 197", FILL_NEUTRAL, "backbone", 1.0),
        ("Positional Embedding\n(learned, 1D)", FILL_NORM, "backbone", 1.0),
    ]
    bottom_rows = [
        ("CLS Token Extracted", FILL_NEUTRAL, "backbone", 1.0),
        ("Linear\n768 → 7 classes", FILL_HEAD, "head", 1.0),
    ]

    heights_top = [r[3] for r in top_rows]
    y_top = sum(heights_top) + gap * (len(top_rows) - 1) + 8.5
    centers_top = layout_centers(heights_top, gap, y_top)

    stage_header(ax, col1_x, y_top + 1.1, "STAGE 1 — Head-only", "backbone frozen, head trains", BORDER_FROZEN)
    stage_header(ax, col2_x, y_top + 1.1, "STAGE 2 — Full fine-tune", "backbone unfrozen, head keeps training", BORDER_UNFROZEN)

    for col_x, role_border in [(col1_x, BORDER_FROZEN), (col2_x, BORDER_UNFROZEN)]:
        for i, ((label, fill, kind, h), cy) in enumerate(zip(top_rows, centers_top)):
            border = BORDER_NEUTRAL if kind == "neutral" and i == 0 else role_border
            draw_box(ax, col_x, cy, box_w, h * 0.92, label, fill, border, lw=2.4 if i > 0 else 1.4, fontsize=9.2)
            if i > 0:
                draw_arrow(ax, col_x, centers_top[i - 1] - heights_top[i - 1] * 0.46, cy + h * 0.46)

    block_top_y = centers_top[-1] - heights_top[-1] * 0.46 - 0.5
    block_bottom_stage1 = build_transformer_block(ax, col1_x, block_top_y, box_w, BORDER_FROZEN)
    block_bottom_stage2 = build_transformer_block(ax, col2_x, block_top_y, box_w, BORDER_UNFROZEN, red_annotation_x=17.4)
    block_bottom = min(block_bottom_stage1, block_bottom_stage2)

    draw_arrow(ax, col1_x, centers_top[-1] - heights_top[-1] * 0.46, block_top_y + 0.15)
    draw_arrow(ax, col2_x, centers_top[-1] - heights_top[-1] * 0.46, block_top_y + 0.15)

    heights_bottom = [r[3] for r in bottom_rows]
    centers_bottom = layout_centers(heights_bottom, gap, block_bottom - gap)
    for col_x, role_border in [(col1_x, BORDER_FROZEN), (col2_x, BORDER_UNFROZEN)]:
        for i, ((label, fill, kind, h), cy) in enumerate(zip(bottom_rows, centers_bottom)):
            border = BORDER_HEAD if kind == "head" else role_border
            draw_box(ax, col_x, cy, box_w, h * 0.92, label, fill, border, lw=2.4, fontsize=9.2)
            if i == 0:
                draw_arrow(ax, col_x, block_bottom, cy + h * 0.46)
            else:
                draw_arrow(ax, col_x, centers_bottom[i - 1] - heights_bottom[i - 1] * 0.46, cy + h * 0.46)

    ann_x = 17.4
    draw_note(ax, ann_x, centers_bottom[-1],
              "Linear head:\nFreshly initialized\noutput = 7 classes",
              arrow_to=(col2_x + box_w / 2, centers_bottom[-1]))
    draw_note(ax, ann_x, y_top + 0.2,
              "Stage 2 LR:\nbackbone = 2e-5\nhead = 2e-4 (backbone ×10)",
              box_edge=BORDER_UNFROZEN)
    draw_note(ax, ann_x, y_top - 1.6,
              "AdamW + linear warmup\n(10% of steps)\nweight_decay = 0.01",
              box_edge=BORDER_UNFROZEN)

    fill_entries = [
        (FILL_CONV, "Conv / patch embed"),
        (FILL_ATTENTION, "Attention (MHSA)"),
        (FILL_MLP, "MLP / feedforward"),
        (FILL_NORM, "Normalization / embedding"),
        (FILL_HEAD, "Head (Linear)"),
    ]
    border_entries = [
        (BORDER_FROZEN, "Frozen (Stage 1)"),
        (BORDER_UNFROZEN, "Unfrozen (Stage 2)"),
        (BORDER_HEAD, "Head — always trainable"),
    ]
    draw_role_legend_entries(ax, 17.0, centers_bottom[-1] - 2.2, 0.85, fill_entries, border_entries,
                              "Red border/text = critical fix\nneeded before next training run")

    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(centers_bottom[-1] - 11.5, y_top + 3.2)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    build_efficientnet_diagram("efficientnet_b0_finetuning.png")
    build_vit_diagram("vit_b16_finetuning.png")
    print("wrote efficientnet_b0_finetuning.png and vit_b16_finetuning.png")
