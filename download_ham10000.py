"""
Download HAM10000 from Hugging Face and save to data/ham10000/.

Usage:
    python download_ham10000.py

Requirements:
    pip install datasets pillow tqdm

Dataset: marmal88/skin_cancer  (public, no auth required)
  - 13,354 dermoscopy images at 600×600 px
  - Columns: image_id, lesion_id, dx, dx_type, age, sex, localization
  - dx values: akiec, bcc, bkl, df, mel, nv, vasc  (maps to classes 0–6)
"""

import os
import sys
import csv
from pathlib import Path
from tqdm import tqdm

# ── Output directory ──────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
OUTPUT_ROOT = SCRIPT_DIR / "data" / "ham10000"
IMAGE_DIR   = OUTPUT_ROOT / "HAM10000_images"
META_CSV    = OUTPUT_ROOT / "HAM10000_metadata.csv"

# ── HuggingFace dataset ───────────────────────────────────────────────────────
HF_DATASET  = "marmal88/skin_cancer"

METADATA_COLS = ["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"]


def main():
    try:
        from datasets import load_dataset, DownloadConfig
    except ImportError:
        sys.exit(
            "ERROR: 'datasets' package not found.\n"
            "Install it with:  pip install datasets\n"
        )

    print(f"Target directory : {OUTPUT_ROOT}")
    print(f"HuggingFace ID   : {HF_DATASET}")
    print()

    if META_CSV.exists():
        existing = sum(1 for _ in IMAGE_DIR.glob("*.jpg")) if IMAGE_DIR.exists() else 0
        print(f"Found existing download: {existing} images in {IMAGE_DIR}")
        answer = input("Re-download? [y/N]: ").strip().lower()
        if answer != "y":
            print("Skipping download.")
            _verify(existing)
            return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset from HuggingFace (this streams metadata first)...")
    print("Full download is ~3.7 GB — this may take 10–30 minutes on a typical connection.\n")

    # Load all three splits so we have the full dataset
    ds = load_dataset(
        HF_DATASET,
        trust_remote_code=True,
    )

    # Combine all splits into a single list for our own stratified split
    all_splits = []
    for split_name, split_ds in ds.items():
        print(f"  Split '{split_name}': {len(split_ds)} samples")
        all_splits.append(split_ds)

    # Merge
    from datasets import concatenate_datasets
    full_ds = concatenate_datasets(all_splits)
    print(f"\nTotal samples: {len(full_ds)}")
    print(f"Saving images to: {IMAGE_DIR}")

    # ── Write images + CSV ────────────────────────────────────────────────────
    rows = []
    failed = 0

    for sample in tqdm(full_ds, desc="Saving", unit="img"):
        img_id  = sample.get("image_id", "")
        pil_img = sample.get("image")

        if not img_id or pil_img is None:
            failed += 1
            continue

        out_path = IMAGE_DIR / f"{img_id}.jpg"
        if not out_path.exists():
            pil_img.convert("RGB").save(out_path, format="JPEG", quality=95)

        rows.append({
            "lesion_id":    sample.get("lesion_id", ""),
            "image_id":     img_id,
            "dx":           sample.get("dx", ""),
            "dx_type":      sample.get("dx_type", ""),
            "age":          sample.get("age", ""),
            "sex":          sample.get("sex", ""),
            "localization": sample.get("localization", ""),
        })

    # Write CSV
    with open(META_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLS)
        writer.writeheader()
        writer.writerows(rows)

    n_saved = len(rows)
    print(f"\nDone. {n_saved} images saved, {failed} skipped.")
    print(f"Metadata CSV : {META_CSV}")
    _verify(n_saved)


def _verify(n_images: int):
    print("\n── Verification ──────────────────────────────────────────────────")
    if not META_CSV.exists():
        print("  ✗ HAM10000_metadata.csv missing")
        return
    with open(META_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    dx_counts = {}
    for row in rows:
        dx_counts[row["dx"]] = dx_counts.get(row["dx"], 0) + 1

    label_map = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6}
    print(f"  Total rows in CSV : {len(rows)}")
    print(f"  Images on disk    : {n_images}")
    print()
    print("  Class distribution:")
    for dx, label_int in label_map.items():
        count = dx_counts.get(dx, 0)
        bar   = "█" * (count // 100)
        print(f"    [{label_int}] {dx:<8} {count:>5}  {bar}")

    unknown = [dx for dx in dx_counts if dx not in label_map]
    if unknown:
        print(f"\n  WARNING: unknown dx values: {unknown}")
    else:
        print(f"\n  ✓ All dx values recognised")
        print(f"  ✓ Ready for training — run: python src/training/train_advanced.py")


if __name__ == "__main__":
    main()
