"""
Build a frontend demo test-set: pick real held-out images that the DEPLOYED predict()
pipeline routes + classifies correctly, copy them into demo_test_samples/, and emit a
manifest with the exact UI settings + the actual app output per sample.

- 5 clinical classes (akiec,bcc,bkl,mel,nv): PAD-UFES-20 TEST (held-out, patient-grouped;
  bcc/mel biopsy-proven). Routed to the clinical path. Metadata is IGNORED by the app here.
- df, vasc: HAM10000 dermoscopy (no PAD clinical data exists). Dermoscopy path.
- metadata-demo: HAM dermoscopy cases where age/sex/site fusion actually fires (the only
  path where the app applies metadata).

Validated through src/app/backend/inference.predict() == what the frontend calls.
Run:  python scripts/build_demo_samples.py
"""
import sys, os, json, shutil, glob, numpy as np, pandas as pd
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src/app/backend"))
from src.app.backend.inference import load_resources, predict
from src.data.pad_ufes import load_pad_ufes

OUT_DIR = os.path.join(ROOT, "demo_test_samples")
MANIFEST = os.path.join(OUT_DIR, "manifest.json")

# UI dropdown anatomical sites (lowercase, must match Scan.jsx ANATOMICAL_SITES)
UI_SITES = {"abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital", "hand",
            "lower extremity", "neck", "scalp", "trunk", "upper extremity"}
SHORT = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}
NOTES = {
    0: "Rough scaly hyperkeratotic patch on sun-damaged skin; strawberry pattern of dotted vessels; adherent white scale.",
    1: "Pearly translucent papule with arborizing branching vessels; blue-grey ovoid nests; shiny white streaks.",
    2: "Stuck-on waxy seborrheic keratosis; cerebriform surface; milia-like cysts and comedo-like openings.",
    3: "Firm dermal papule on the lower leg; central white scar-like patch; peripheral pigment network; positive dimple sign.",
    4: "Asymmetric lesion with atypical irregular pigment network; blue-white veil; regression structures; multiple colours; ABCDE positive.",
    5: "Symmetric mole with regular uniform pigment network; homogeneous brown colour; smooth regular borders.",
    6: "Cherry angioma with well-defined red to purple lacunae; homogeneous red area; sharp borders; no pigment network; blanches with pressure.",
}


def run(res, path, age, sex, site, note):
    img = Image.open(path).convert("RGB")
    meta = {"age": (str(age) if age == age and age is not None else None),
            "sex": sex, "localization": site}
    r = predict(img, res, metadata=meta, clinical_note=note)
    top = r["predictions"][0]
    return {"top_id": r["top_class_id"], "top_name": r["top_class_name"],
            "prob": round(float(top["probability"]), 3), "modality": r["modality"],
            "p_clinical": r.get("p_clinical"), "metadata_used": r["metadata_used"],
            "metadata_changed_ranking": r["metadata_changed_ranking"],
            "note_used": r["note_used"], "note_supports": r["note_supports"],
            "refer": r["refer_to_specialist"], "is_unknown": r["is_unknown"],
            "native_res": r["native_resolution"]}


def main():
    print("Loading resources (deployed model + artifacts)...")
    res = load_resources()
    os.makedirs(OUT_DIR, exist_ok=True)
    for sub in ("clinical_pad", "dermoscopy_ham"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    manifest = []

    # ---------- PAD clinical: akiec, bcc, bkl, mel, nv ----------
    pad = load_pad_ufes(verbose=False)["test"]
    meta_csv = pd.read_csv(os.path.join(ROOT, "data/pad_ufes20/metadata.csv"))
    biopsed = {str(r.img_id): bool(r.biopsed) for r in meta_csv.itertuples()}
    pad["img_id"] = pad["path"].map(lambda p: os.path.basename(p))
    pad["biopsed"] = pad["img_id"].map(lambda x: biopsed.get(x, False))

    for cls in [0, 1, 2, 4, 5]:
        cand = pad[(pad["label"] == cls) & (pad["site"].isin(UI_SITES))].copy()
        cand = cand.sort_values("biopsed", ascending=False)  # biopsy-proven first
        best = None
        for i, row in enumerate(cand.itertuples()):
            if i >= 30:
                break
            out = run(res, row.path, row.age, row.sex, row.site, NOTES[cls])
            ok = out["modality"] == "clinical" and out["top_id"] == cls
            score = out["prob"] + (0.2 if getattr(row, "biopsed") else 0)
            if ok and (best is None or score > best["_score"]):
                best = {**out, "_score": score, "_path": row.path, "_age": row.age,
                        "_sex": row.sex, "_site": row.site, "_biopsed": bool(row.biopsed)}
                if out["prob"] >= 0.6:
                    break
        if best is None:
            print(f"  [PAD {SHORT[cls]}] no clean clinical-routed sample found in 30 tries")
            continue
        dst = os.path.join(OUT_DIR, "clinical_pad", f"{SHORT[cls]}_clinical.jpg")
        Image.open(best["_path"]).convert("RGB").save(dst, "JPEG", quality=92)
        sx = {"male": "M", "female": "F"}.get(str(best["_sex"]).lower(), "O")
        manifest.append({
            "file": os.path.relpath(dst, ROOT).replace("\\", "/"), "track": "clinical_pad",
            "true_class": SHORT[cls], "source": "PAD-UFES-20 test (held-out)",
            "biopsy_proven": best["_biopsed"],
            "ui": {"age": None if best["_age"] != best["_age"] else int(best["_age"]),
                   "sex": sx, "anatomical_site": best["_site"],
                   "scan_modality": "Clinical Photo", "clinical_note": NOTES[cls]},
            "app_result": {k: best[k] for k in best if not k.startswith("_")},
        })
        print(f"  [PAD {SHORT[cls]}] -> {best['top_name']} {best['prob']} "
              f"(modality={best['modality']}, refer={best['refer']}, biopsy={best['_biopsed']})")

    # ---------- HAM dermoscopy: df, vasc + metadata-demo ----------
    ham = pd.read_csv(os.path.join(ROOT, "data/ham10000/HAM10000_metadata.csv"))
    img_idx = {os.path.splitext(os.path.basename(p))[0]: p
               for p in glob.glob(os.path.join(ROOT, "data/ham10000/**/*.jpg"), recursive=True)}
    # HAM dx column uses full names (case as in the CSV).
    cls2dx = {0: "actinic_keratoses", 1: "basal_cell_carcinoma", 2: "benign_keratosis-like_lesions",
              3: "dermatofibroma", 4: "melanoma", 5: "melanocytic_Nevi", 6: "vascular_lesions"}
    ham = ham[ham["image_id"].isin(img_idx)].copy()

    def ham_pick(cls, need_meta=False, need_changed=False, max_try=40):
        sub = ham[ham["dx"] == cls2dx[cls]].copy()
        sub = sub[sub["localization"].isin(UI_SITES)]
        sub = sub.sample(frac=1.0, random_state=42)
        best = None
        for i, row in enumerate(sub.itertuples()):
            if i >= max_try:
                break
            p = img_idx[row.image_id]
            sx = {"male": "M", "female": "F"}.get(str(row.sex).lower(), "O")
            out = run(res, p, row.age, sx, row.localization, NOTES[cls])
            ok = out["modality"] == "dermoscopy" and out["top_id"] == cls
            if need_meta and not out["metadata_used"]:
                ok = False
            if need_changed and not out["metadata_changed_ranking"]:
                ok = False
            score = out["prob"] + (0.5 if out["metadata_changed_ranking"] else 0)
            if ok and (best is None or score > best["_score"]):
                best = {**out, "_score": score, "_path": p, "_age": row.age, "_sex": sx,
                        "_site": row.localization}
                if not need_changed and out["prob"] >= 0.6:
                    break
        return best

    for cls in [3, 6]:
        b = ham_pick(cls)
        if b is None:
            print(f"  [HAM {SHORT[cls]}] no clean sample"); continue
        dst = os.path.join(OUT_DIR, "dermoscopy_ham", f"{SHORT[cls]}_dermoscopy.jpg")
        Image.open(b["_path"]).convert("RGB").save(dst, "JPEG", quality=92)
        manifest.append({
            "file": os.path.relpath(dst, ROOT).replace("\\", "/"), "track": "dermoscopy_ham",
            "true_class": SHORT[cls], "source": "HAM10000 dermoscopy (model domain; no PAD clinical data for this class)",
            "biopsy_proven": None,
            "ui": {"age": None if b["_age"] != b["_age"] else int(b["_age"]), "sex": b["_sex"],
                   "anatomical_site": b["_site"], "scan_modality": "Dermoscopy", "clinical_note": NOTES[cls]},
            "app_result": {k: b[k] for k in b if not k.startswith("_")},
        })
        print(f"  [HAM {SHORT[cls]}] -> {b['top_name']} {b['prob']} (meta_used={b['metadata_used']})")

    # metadata-demo: akiec + mel where fusion fires (prefer changed ranking)
    for cls in [0, 4]:
        b = ham_pick(cls, need_meta=True, need_changed=True) or ham_pick(cls, need_meta=True)
        if b is None:
            print(f"  [HAM meta-demo {SHORT[cls]}] none"); continue
        dst = os.path.join(OUT_DIR, "dermoscopy_ham", f"metademo_{SHORT[cls]}_dermoscopy.jpg")
        Image.open(b["_path"]).convert("RGB").save(dst, "JPEG", quality=92)
        manifest.append({
            "file": os.path.relpath(dst, ROOT).replace("\\", "/"), "track": "metadata_demo",
            "true_class": SHORT[cls], "source": "HAM10000 dermoscopy (metadata-fusion showcase)",
            "biopsy_proven": None,
            "ui": {"age": None if b["_age"] != b["_age"] else int(b["_age"]), "sex": b["_sex"],
                   "anatomical_site": b["_site"], "scan_modality": "Dermoscopy", "clinical_note": NOTES[cls]},
            "app_result": {k: b[k] for k in b if not k.startswith("_")},
        })
        print(f"  [HAM meta-demo {SHORT[cls]}] -> {b['top_name']} {b['prob']} "
              f"(meta_used={b['metadata_used']}, changed={b['metadata_changed_ranking']})")

    json.dump(manifest, open(MANIFEST, "w"), indent=2)
    print(f"\nWrote {len(manifest)} samples -> {os.path.relpath(MANIFEST, ROOT)}")


if __name__ == "__main__":
    main()
