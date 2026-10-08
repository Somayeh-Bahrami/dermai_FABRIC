#!/usr/bin/env python3
"""Download and unpack the HAM10000 dataset into data/.

Fetches the files from the Harvard Dataverse (DOI 10.7910/DVN/DBW86T),
unzips the image and segmentation archives, and lays everything out under
data/ so the rest of the project can run without manual setup.

Usage:
    python scripts/download_data.py            # core files (~2.8 GB)
    python scripts/download_data.py --isic      # also fetch the ISIC 2018 test set
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CORE = {
    "HAM10000_images_part_1.zip": 3172585,
    "HAM10000_images_part_2.zip": 3172584,
    "HAM10000_metadata": 4338392,
    "HAM10000_segmentations_lesion_tschandl.zip": 3838943,
}
ISIC = {
    "ISIC2018_Task3_Test_Images.zip": 3855824,
    "ISIC2018_Task3_Test_GroundTruth.csv": 6924466,
}


def download(name: str, file_id: int, dest: Path) -> None:
    if dest.exists():
        print(f"  skip {name} (already present)")
        return
    url = f"{DATAVERSE}/{file_id}"
    print(f"  get  {name}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            read += len(chunk)
            if total:
                pct = read * 100 // total
                print(f"\r       {pct:3d}%  {read >> 20} / {total >> 20} MB", end="")
        print()


def unzip(name: str, dest_dir: Path) -> None:
    archive = DATA_DIR / name
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.namelist() if not m.startswith("__MACOSX")]
        if all((dest_dir / m).exists() for m in members):
            print(f"  skip unzip {name} (already extracted)")
            return
        print(f"  unzip {name}")
        z.extractall(dest_dir, members)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isic", action="store_true", help="also download the ISIC 2018 test set")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    targets = {**CORE, **(ISIC if args.isic else {})}

    print(f"Downloading into {DATA_DIR}")
    for name, file_id in targets.items():
        download(name, file_id, DATA_DIR / name)

    print("Extracting archives")
    for name in targets:
        if name.endswith(".zip"):
            unzip(name, DATA_DIR)

    print("Done. Dataset is ready under data/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
