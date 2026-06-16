"""
PAD-UFES-20 loader + label/metadata harmonization — the SINGLE place that knows the
PAD-UFES-20 schema. Every downstream script (cross-dataset eval, clinical prototypes, OOD
recalibration, metadata fusion) consumes the clean harmonized frame this returns, so a
column/path mismatch can only ever surface HERE (and it asserts loudly if so).

PAD-UFES-20 (Pacheco et al., 2020, Mendeley Data — search "PAD-UFES-20"): ~2,298 CLINICAL
smartphone photos (NOT dermoscopy) of ~1,373 patients, with rich per-lesion metadata.
This is the off-modality / domain-shift test set for the patient-upload path — the exact
modality real users upload (vs the dermoscopy the model trained on).

DESIGN (why "ironclad"):
- ONE patient-grouped split, up front (GroupShuffleSplit on patient_id). Multiple images
  per lesion and multiple lesions per patient → a random split leaks the same patient
  across train/test (the lesion_id leakage we police in HAM). PAD-TRAIN feeds the
  adaptation steps; PAD-TEST is reserved for ALL reporting and never seen in training.
- Label harmonization to the 7 HAM ints. Shared classes only: BCC, MEL, NEV, ACK->akiec,
  SEK->bkl. SCC has NO HAM class -> DROPPED. HAM's df/vasc have no PAD examples.
- Metadata harmonization (gender->HAM sex vocab, region->HAM localization vocab) so the
  metadata-fusion tables fit on HAM transfer cleanly.

Expected layout (override via env PAD_ROOT):  data/pad_ufes20/
    metadata.csv                      (or any *.csv with the expected columns)
    imgs_part_1/ imgs_part_2/ ...      (images discovered by globbing img_id, layout-agnostic)
"""
import os, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# PAD diagnostic label -> HAM 7-way int. SCC intentionally absent (no HAM equivalent).
PAD_TO_HAM = {"ACK": 0, "BCC": 1, "SEK": 2, "MEL": 4, "NEV": 5}
HAM_NAMES = {0: "Actinic keratoses", 1: "Basal cell carcinoma", 2: "Benign keratosis",
             3: "Dermatofibroma", 4: "Melanoma", 5: "Melanocytic nevi", 6: "Vascular lesions"}
SHARED_HAM_CLASSES = sorted(set(PAD_TO_HAM.values()))   # [0,1,2,4,5]

# PAD `region` (anatomical site) -> HAM `localization` vocab. Best-effort; unmapped -> unknown.
REGION_TO_HAM_SITE = {
    "FACE": "face", "NOSE": "face", "FOREHEAD": "face", "CHEEK": "face", "LIP": "face",
    "EAR": "ear", "SCALP": "scalp", "NECK": "neck", "CHEST": "chest", "BACK": "back",
    "ABDOMEN": "abdomen", "ARM": "upper extremity", "FOREARM": "upper extremity",
    "HAND": "hand", "THIGH": "lower extremity", "SHIN": "lower extremity",
    "LOWER LEG": "lower extremity", "CALF": "lower extremity", "FOOT": "foot",
}

REQUIRED_COLS = ["img_id", "diagnostic", "patient_id"]
# Accepted aliases (case-insensitive) -> canonical name used internally.
COL_ALIASES = {
    "img_id": ["img_id", "image_id", "imageid"],
    "diagnostic": ["diagnostic", "diagnostic_1", "dx", "diagnosis"],
    "patient_id": ["patient_id", "patientid", "patient"],
    "lesion_id": ["lesion_id", "lesionid", "lesion"],
    "age": ["age", "age_years"],
    "gender": ["gender", "sex"],
    "region": ["region", "localization", "anatom_site", "anatomical_site", "site"],
}


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def _resolve_root(data_root=None):
    if data_root is None:
        data_root = os.environ.get("PAD_ROOT", os.path.join(_project_root(), "data", "pad_ufes20"))
    return data_root


def _find_csv(data_root):
    direct = os.path.join(data_root, "metadata.csv")
    if os.path.exists(direct):
        return direct
    cands = glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True)
    if not cands:
        raise FileNotFoundError(
            f"No CSV under {data_root}. Download PAD-UFES-20 (Mendeley Data) and extract to "
            f"{data_root}/ (expecting metadata.csv + imgs_part_*/).")
    return cands[0]


def _canonicalize_columns(df):
    """Map the CSV's actual columns to our canonical names (case-insensitive, alias-aware)."""
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for canon, aliases in COL_ALIASES.items():
        for a in aliases:
            if a in lower:
                rename[lower[a]] = canon
                break
    df = df.rename(columns=rename)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"PAD metadata is missing required columns {missing} after alias mapping.\n"
            f"  CSV header was: {list(df.columns)}\n"
            f"  Add the real names to COL_ALIASES in src/data/pad_ufes.py.")
    return df


def _norm_sex(g):
    if not isinstance(g, str):
        return "unknown"
    g = g.strip().lower()
    if g in ("m", "male"):
        return "male"
    if g in ("f", "female"):
        return "female"
    return "unknown"


def _norm_site(r):
    if not isinstance(r, str):
        return "unknown"
    return REGION_TO_HAM_SITE.get(r.strip().upper(), "unknown")


def _discover_images(data_root):
    """{img_id: abspath} by globbing all images under data_root (layout-agnostic)."""
    idx = {}
    for ext in ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG"):
        for p in glob.glob(os.path.join(data_root, "**", f"*.{ext}"), recursive=True):
            idx[os.path.basename(p)] = p
    return idx


def load_pad_ufes(data_root=None, test_size=0.30, seed=42, verbose=True):
    """Return dict with 'train'/'test' DataFrames (patient-grouped split) and harmonized
    columns: path, label(int 0-6), diagnostic(raw), patient_id, lesion_id, age, sex, site.
    Only rows whose PAD diagnostic maps to a shared HAM class are kept (SCC etc. dropped)."""
    data_root = _resolve_root(data_root)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"PAD-UFES-20 not found at {data_root}. Download from Mendeley Data "
            f"(search 'PAD-UFES-20') and extract metadata.csv + imgs_part_*/ there, "
            f"or set env PAD_ROOT to its location.")

    csv_path = _find_csv(data_root)
    df = pd.read_csv(csv_path)
    if verbose:
        print(f"PAD CSV: {csv_path}  rows={len(df)}")
        print(f"  raw header: {list(df.columns)}")
    df = _canonicalize_columns(df)

    # image paths
    img_index = _discover_images(data_root)
    if verbose:
        print(f"  discovered {len(img_index)} image files under {data_root}")
    df["path"] = df["img_id"].astype(str).map(img_index)
    n_missing_img = int(df["path"].isna().sum())
    df = df[df["path"].notna()].copy()

    # label harmonization (drop unmapped: SCC, etc.)
    df["diagnostic"] = df["diagnostic"].astype(str).str.strip().str.upper()
    n_before = len(df)
    df["label"] = df["diagnostic"].map(PAD_TO_HAM)
    dropped = df[df["label"].isna()]
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    # metadata harmonization
    df["sex"] = df["gender"].map(_norm_sex) if "gender" in df.columns else "unknown"
    df["site"] = df["region"].map(_norm_site) if "region" in df.columns else "unknown"
    if "age" not in df.columns:
        df["age"] = np.nan
    if "lesion_id" not in df.columns:
        df["lesion_id"] = df["patient_id"].astype(str) + "_" + df["img_id"].astype(str)

    keep = ["path", "label", "diagnostic", "patient_id", "lesion_id", "age", "sex", "site"]
    df = df[keep].reset_index(drop=True)

    if verbose:
        print(f"  dropped {n_missing_img} rows w/o image file; "
              f"{n_before - len(dropped) - len(df) if False else len(dropped)} rows dropped as unmapped dx "
              f"(e.g. SCC: {sorted(dropped['diagnostic'].unique().tolist())})")
        counts = df["label"].value_counts().sort_index()
        print("  kept per HAM class:")
        for c in SHARED_HAM_CLASSES:
            print(f"    {c} {HAM_NAMES[c]:22s}: {int(counts.get(c, 0))}")
        print(f"  total kept: {len(df)}  patients: {df['patient_id'].nunique()}")

    # ONE patient-grouped split (reserve test for all reporting)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(gss.split(df, df["label"], groups=df["patient_id"]))
    train, test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    # sanity: no patient leaks across the split
    overlap = set(train["patient_id"]) & set(test["patient_id"])
    assert not overlap, f"patient leakage across split: {len(overlap)} shared patients"
    if verbose:
        print(f"  split (grouped by patient): train {len(train)} / test {len(test)}  "
              f"(no patient overlap: OK)")
    return {"train": train, "test": test, "shared_classes": SHARED_HAM_CLASSES}


if __name__ == "__main__":
    # Smoke test the loader against whatever is on disk.
    d = load_pad_ufes()
    print("\nOK — train/test frames built.")
