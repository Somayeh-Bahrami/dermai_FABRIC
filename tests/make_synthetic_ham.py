"""Builds a tiny HAM10000-shaped tree (metadata, 600x450 JPGs, lesion masks) for CPU
smoke tests when the real dataset is unavailable. Never used for paper numbers.

    python tests/make_synthetic_ham.py <out_dir> [--lesions-per-class 30]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--lesions-per-class", type=int, default=30)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    image_dir = args.out_dir / "HAM10000_images_part_1"
    mask_dir = args.out_dir / "HAM10000_segmentations_lesion_tschandl"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = ["lesion_id,image_id,dx,dx_type,age,sex,localization"]
    index = 0
    for class_index, dx in enumerate(CLASSES):
        for lesion in range(args.lesions_per_class):
            lesion_id = f"HAM_{class_index:02d}{lesion:05d}"
            for _ in range(int(rng.integers(1, 3))):
                image_id = f"ISIC_{index:07d}"
                index += 1
                cx, cy = rng.integers(200, 400), rng.integers(150, 300)
                rx, ry = rng.integers(60, 150), rng.integers(50, 120)
                box = [cx - rx, cy - ry, cx + rx, cy + ry]
                background = rng.integers(150, 230, size=(450, 600, 3), dtype=np.uint8)
                image = Image.fromarray(background)
                ImageDraw.Draw(image).ellipse(box, fill=tuple(int(v) for v in rng.integers(30, 120, 3)))
                image.save(image_dir / f"{image_id}.jpg", quality=85)
                mask = Image.new("L", (600, 450), 0)
                ImageDraw.Draw(mask).ellipse(box, fill=255)
                mask.save(mask_dir / f"{image_id}_segmentation.png")
                rows.append(f"{lesion_id},{image_id},{dx},histo,50.0,female,back")
    (args.out_dir / "HAM10000_metadata").write_text("\n".join(rows) + "\n")
    print(f"wrote {index} synthetic images to {args.out_dir}")


if __name__ == "__main__":
    main()
