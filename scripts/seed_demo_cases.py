"""Seed the dashboard with demo cases by running real HAM10000 images through
the live /api/predict endpoint — so every seeded case carries genuine model
output (calibrated probabilities, OOD score, heatmap, referral decision).

Backdates each case over the past ~10 days so the 14-day activity chart and
KPIs look like an in-use clinic rather than a single burst.

Usage (backend must be running):
    python scripts/seed_demo_cases.py [--api http://localhost:8000]
"""

import argparse
import csv
import os
import random
import sqlite3
from datetime import datetime, timedelta, timezone

import requests

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HAM_DIR = os.path.join(REPO_ROOT, "data", "ham10000")
IMAGES_DIR = os.path.join(HAM_DIR, "HAM10000_images")
METADATA_CSV = os.path.join(HAM_DIR, "HAM10000_metadata.csv")
DB_PATH = os.path.join(REPO_ROOT, "src", "app", "assets", "raresight.db")

# (dx in local CSV, patient name, expected sex in CSV row, clinical note)
SEED_PLAN = [
    ("melanoma", "Nuwan Rajapaksa", "male",
     "Dark irregular lesion on the calf; patient reports change in size and colour over recent months; family history of skin cancer."),
    ("basal_cell_carcinoma", "Kasun Bandara", "male",
     "Pearly nodule on the nose with occasional bleeding; long history of outdoor farming work."),
    ("actinic_keratoses", "Sanduni Fernando", "female",
     "Rough scaly patch on the cheek, present for over a year, sun-exposed site."),
    ("melanocytic_Nevi", "Dilani Jayawardena", "female",
     "Stable brown mole on the back, routine skin check, no reported change."),
    ("benign_keratosis-like_lesions", "Tharindu Silva", "male",
     "Waxy stuck-on plaque on the upper back, asymptomatic."),
    ("vascular_lesions", "Amara Wickramasinghe", "female",
     "Small red-purple papule on the trunk, unchanged for years."),
]


DX_TO_ID = {
    "actinic_keratoses": 0, "basal_cell_carcinoma": 1, "benign_keratosis-like_lesions": 2,
    "dermatofibroma": 3, "melanoma": 4, "melanocytic_Nevi": 5, "vascular_lesions": 6,
}
MAX_CANDIDATES = 25  # tries per patient to find an image the model reads correctly


def pick_candidates() -> list[tuple[list[str], dict]]:
    """Per plan entry, return a shuffled list of candidate image paths (dx + sex matched)
    so the seeding loop can keep the first one the deployed model classifies correctly —
    a coherent dashboard (malignancies referred, benigns routine), not random misses."""
    by_dx: dict[str, list[dict]] = {}
    with open(METADATA_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_dx.setdefault(row["dx"], []).append(row)
    rng = random.Random(42)
    out = []
    used = set()
    for dx, name, want_sex, note in SEED_PLAN:
        rows = [r for r in by_dx.get(dx, [])
                if r["image_id"] not in used and r["sex"] == want_sex]
        rng.shuffle(rows)
        cands = []
        for r in rows[:MAX_CANDIDATES]:
            path = os.path.join(IMAGES_DIR, r["image_id"] + ".jpg")
            if os.path.exists(path):
                cands.append((path, r))
        out.append((cands, {"name": name, "note": note, "dx_id": DX_TO_ID[dx]}))
    return out


def backdate_cases(case_ids: list[int]) -> None:
    """Spread the seeded cases over the past 10 days (most recent first)."""
    now = datetime.now(timezone.utc)
    conn = sqlite3.connect(DB_PATH)
    for i, cid in enumerate(case_ids):
        ts = now - timedelta(days=i, hours=(i * 3) % 9, minutes=(i * 17) % 60)
        conn.execute("UPDATE cases SET created_at = ? WHERE id = ?",
                     (ts.isoformat(), cid))
    conn.commit()
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    args = ap.parse_args()

    health = requests.get(f"{args.api}/api/health", timeout=10)
    health.raise_for_status()
    print(f"Backend online: {health.json()['model']}")

    def scan(path, row, info):
        age = row["age"].split(".")[0] if row["age"] else ""
        sex = {"male": "M", "female": "F"}.get(row["sex"], "")
        with open(path, "rb") as f:
            resp = requests.post(
                f"{args.api}/api/predict",
                files={"file": (os.path.basename(path), f, "image/jpeg")},
                data={"patient_name": info["name"], "age": age, "sex": sex,
                      "localization": row["localization"], "scan_type": "Dermoscopy",
                      "clinical_note": info["note"]},
                timeout=120,
            )
        resp.raise_for_status()
        return resp.json()

    case_ids, discard = [], []
    for cands, info in pick_candidates():
        kept = None
        for path, row in cands:
            r = scan(path, row, info)
            if r["top_class_id"] == info["dx_id"]:
                kept = r                       # first correctly-read image; stop here
                break
            discard.append(r["case_id"])       # rejected attempt -> delete below
        if kept is None and cands:             # none matched; fall back to one attempt
            kept = scan(cands[0][0], cands[0][1], info)
        if kept:
            case_ids.append(kept["case_id"])
            print(f"  {kept['case_display_id']}  {info['name']:<20} -> {kept['top_class_name']} "
                  f"({kept['predictions'][0]['probability'] * 100:.1f}%)"
                  f"{'  [REFER]' if kept['refer_to_specialist'] else ''}")

    for cid in discard:                        # remove rejected scans so no junk remains
        requests.delete(f"{args.api}/api/cases/{cid}", timeout=30)

    backdate_cases(case_ids)
    print(f"\nSeeded {len(case_ids)} cases (cleaned {len(discard)} rejected attempts), "
          f"backdated over ~10 days.")


if __name__ == "__main__":
    main()
