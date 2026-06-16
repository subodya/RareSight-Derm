"""Curate real dermoscopy images for the Disease Library from local HAM10000.

Picks N histopathology-confirmed images per class (deterministic: sorted by
image_id) and copies them to src/app/assets/library_images/cls_{id}_{n}.jpg,
where the backend serves them statically at /api/images/library_images/.

Run from repo root:  python scripts/build_library_images.py
"""

import csv
import os
import shutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HAM_DIR = os.path.join(REPO_ROOT, "data", "ham10000")
IMAGES_DIR = os.path.join(HAM_DIR, "HAM10000_images")
METADATA_CSV = os.path.join(HAM_DIR, "HAM10000_metadata.csv")
OUT_DIR = os.path.join(REPO_ROOT, "src", "app", "assets", "library_images")

PER_CLASS = 8

# dx string in local metadata CSV -> RareSight class id
DX_TO_CLASS = {
    "actinic_keratoses": 0,
    "basal_cell_carcinoma": 1,
    "benign_keratosis-like_lesions": 2,
    "dermatofibroma": 3,
    "melanoma": 4,
    "melanocytic_Nevi": 5,
    "vascular_lesions": 6,
}


def main():
    by_class: dict[int, list[dict]] = {i: [] for i in DX_TO_CLASS.values()}
    with open(METADATA_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls = DX_TO_CLASS.get(row["dx"])
            if cls is not None:
                by_class[cls].append(row)

    os.makedirs(OUT_DIR, exist_ok=True)
    for cls, rows in sorted(by_class.items()):
        # Prefer histopathology-confirmed diagnoses; deterministic order.
        rows.sort(key=lambda r: (r["dx_type"] != "histo", r["image_id"]))
        copied = 0
        for row in rows:
            src = os.path.join(IMAGES_DIR, row["image_id"] + ".jpg")
            if not os.path.exists(src):
                continue
            dst = os.path.join(OUT_DIR, f"cls_{cls}_{copied}.jpg")
            shutil.copyfile(src, dst)
            copied += 1
            if copied == PER_CLASS:
                break
        print(f"class {cls}: copied {copied} images")

    print(f"\nDone -> {OUT_DIR}")


if __name__ == "__main__":
    main()
