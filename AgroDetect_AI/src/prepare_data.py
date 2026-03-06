"""
AgroDetect AI - Plant Disease Classification Engine
====================================================
Step 0: Download & prepare the PlantVillage dataset
Command: python src/prepare_data.py
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

RAW_DIR  = Path("data/raw")
PROC_DIR = Path("data/processed")

# ─────────────────────────────────────────────
# OPTION A – Use Kaggle CLI (recommended)
# ─────────────────────────────────────────────
def download_via_kaggle():
    """
    Requires:  pip install kaggle
               ~/.kaggle/kaggle.json  (your API key)
    """
    print("\n[Kaggle] Downloading PlantVillage dataset...")
    os.makedirs(RAW_DIR, exist_ok=True)
    os.system(
        f"kaggle datasets download -d emmarex/plantdisease "
        f"-p {RAW_DIR} --unzip"
    )
    print("[Kaggle] Download complete.")

# ─────────────────────────────────────────────
# OPTION B – Use a local zip you already have
# ─────────────────────────────────────────────
def extract_local_zip(zip_path: str):
    print(f"\n[Local] Extracting {zip_path} → {RAW_DIR}")
    os.makedirs(RAW_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(RAW_DIR)
    print("[Local] Extraction complete.")

# ─────────────────────────────────────────────
# RESIZE & ORGANISE
# ─────────────────────────────────────────────
def preprocess_images(src_root: Path, dst_root: Path,
                      img_size=(224, 224)):
    """
    Walk src_root/<class_name>/*.jpg  →  dst_root/<class_name>/*.jpg
    Resizes every image to img_size.
    """
    print(f"\n[Preprocess] Resizing images to {img_size}...")
    class_dirs = sorted([d for d in src_root.iterdir() if d.is_dir()])
    total = sum(len(list(d.glob("*"))) for d in class_dirs)

    with tqdm(total=total, unit="img") as pbar:
        for cls_dir in class_dirs:
            out_dir = dst_root / cls_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    pbar.update(1)
                    continue
                img_r = cv2.resize(img, img_size)
                cv2.imwrite(str(out_dir / img_path.name), img_r)
                pbar.update(1)

    print(f"[Preprocess] Done → {dst_root}")

# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────
def print_dataset_stats(root: Path):
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    print(f"\n{'─'*50}")
    print(f"  Dataset Statistics  ({root})")
    print(f"{'─'*50}")
    total = 0
    for cls in classes:
        n = len(list((root / cls).glob("*")))
        total += n
        print(f"  {cls:<45} {n:>5} images")
    print(f"{'─'*50}")
    print(f"  {'TOTAL':<45} {total:>5} images")
    print(f"{'─'*50}\n")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle",   action="store_true", help="Download via Kaggle CLI")
    parser.add_argument("--zip",      default="",          help="Path to local dataset zip")
    parser.add_argument("--raw_dir",  default=str(RAW_DIR), help="Raw data directory")
    args = parser.parse_args()

    if args.kaggle:
        download_via_kaggle()
    elif args.zip:
        extract_local_zip(args.zip)
    else:
        print("\n⚠️  No download flag provided.")
        print("   Use --kaggle  OR  --zip path/to/plantvillage.zip")
        print("   Assuming data already in data/raw/ ...\n")

    raw = Path(args.raw_dir)
    if raw.exists():
        preprocess_images(raw, PROC_DIR)
        print_dataset_stats(PROC_DIR)
    else:
        print(f"❌ Raw directory not found: {raw}")
        print("   Please download the dataset first.")
