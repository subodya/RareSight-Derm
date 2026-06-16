"""Honest hit-rate on RANDOM (un-cherry-picked) images through the deployed predict(),
mirroring frontend usage. Shows what a user actually experiences uploading arbitrary images,
vs the cherry-picked demo set. Reports per-class hit-rate + mean top-confidence + mean rank
of the true class, for both the clinical (PAD) and dermoscopy (HAM) paths."""
import sys, os, glob, random, numpy as np, pandas as pd
from collections import defaultdict
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.app.backend.inference import load_resources, predict
from src.data.pad_ufes import load_pad_ufes

SHORT = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}
NOTES = {
    0: "Rough scaly hyperkeratotic patch on sun-damaged skin; strawberry pattern of dotted vessels; adherent white scale.",
    1: "Pearly translucent papule with arborizing branching vessels; blue-grey ovoid nests; shiny white streaks.",
    2: "Stuck-on waxy seborrheic keratosis; cerebriform surface; milia-like cysts and comedo-like openings.",
    3: "Firm dermal papule on the lower leg; central white scar-like patch; peripheral pigment network; positive dimple sign.",
    4: "Asymmetric lesion with atypical irregular pigment network; blue-white veil; regression structures; multiple colours; ABCDE positive.",
    5: "Symmetric mole with regular uniform pigment network; homogeneous brown colour; smooth regular borders.",
    6: "Cherry angioma with well-defined red to purple lacunae; homogeneous red area; sharp borders; no pigment network; blanches.",
}
PER_CLASS = 25
random.seed(0)


def run(res, path, age, sex, site, modality, note=None):
    img = Image.open(path).convert("RGB")
    meta = {"age": (str(age) if age == age and age is not None else None), "sex": sex, "localization": site}
    r = predict(img, res, metadata=meta, clinical_note=note)
    order = [p["class_id"] for p in r["predictions"]]
    return r["top_class_id"], float(r["predictions"][0]["probability"]), r["modality"], r["refer_to_specialist"]


def report(title, rows):
    # rows: list of (true_cls, top_cls, prob, modality)
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)
    by = defaultdict(list)
    for t, p, pr, mod in rows:
        by[t].append((p == t, pr, mod))
    allhit, allconf = [], []
    print(f"  {'class':<8}{'n':>4}{'hit-rate':>10}{'mean conf':>11}{'routed':>16}")
    for c in sorted(by):
        hits = [h for h, _, _ in by[c]]
        confs = [pr for _, pr, _ in by[c]]
        mods = defaultdict(int)
        for _, _, m in by[c]:
            mods[m] += 1
        allhit += hits; allconf += confs
        modstr = ",".join(f"{k}:{v}" for k, v in mods.items())
        print(f"  {SHORT[c]:<8}{len(hits):>4}{np.mean(hits)*100:>9.0f}%{np.mean(confs):>11.3f}   {modstr:<16}")
    print(f"  {'-'*50}")
    print(f"  {'OVERALL':<8}{len(allhit):>4}{np.mean(allhit)*100:>9.0f}%{np.mean(allconf):>11.3f}")


def main():
    print("Loading deployed model...")
    res = load_resources()

    # ---- Clinical path: random PAD-test images (no selection) ----
    pad = load_pad_ufes(verbose=False)["test"]
    rows_nonote, rows_note = [], []
    for c in [0, 1, 2, 4, 5]:
        sub = pad[pad["label"] == c]
        idx = list(sub.index)
        random.shuffle(idx)
        for i in idx[:PER_CLASS]:
            r = sub.loc[i]
            sx = {"male": "M", "female": "F"}.get(str(r["sex"]).lower(), "O")
            top, prob, mod, _ = run(res, r["path"], r["age"], sx, r["site"], "Clinical Photo")
            rows_nonote.append((c, top, prob, mod))
            top2, prob2, mod2, _ = run(res, r["path"], r["age"], sx, r["site"], "Clinical Photo", NOTES[c])
            rows_note.append((c, top2, prob2, mod2))
    report(f"CLINICAL — random PAD-test, {PER_CLASS}/class, NO note (image+routing only)", rows_nonote)
    report(f"CLINICAL — same images, WITH class-matching clinical note", rows_note)

    # ---- Dermoscopy path: random HAM images (no selection) ----
    ham = pd.read_csv(os.path.join(ROOT, "data/ham10000/HAM10000_metadata.csv"))
    img_idx = {os.path.splitext(os.path.basename(p))[0]: p
               for p in glob.glob(os.path.join(ROOT, "data/ham10000/**/*.jpg"), recursive=True)}
    ham = ham[ham["image_id"].isin(img_idx)]
    cls2dx = {0: "actinic_keratoses", 1: "basal_cell_carcinoma", 2: "benign_keratosis-like_lesions",
              3: "dermatofibroma", 4: "melanoma", 5: "melanocytic_Nevi", 6: "vascular_lesions"}
    rows_nonote, rows_note = [], []
    for c in range(7):
        sub = ham[ham["dx"] == cls2dx[c]].sample(frac=1.0, random_state=1)
        for r in list(sub.itertuples())[:PER_CLASS]:
            sx = {"male": "M", "female": "F"}.get(str(r.sex).lower(), "O")
            top, prob, mod, _ = run(res, img_idx[r.image_id], r.age, sx, r.localization, "Dermoscopy")
            rows_nonote.append((c, top, prob, mod))
            top2, prob2, mod2, _ = run(res, img_idx[r.image_id], r.age, sx, r.localization, "Dermoscopy", NOTES[c])
            rows_note.append((c, top2, prob2, mod2))
    report(f"DERMOSCOPY — random HAM, {PER_CLASS}/class balanced, metadata, NO note", rows_nonote)
    report(f"DERMOSCOPY — same images, WITH class-matching clinical note", rows_note)


if __name__ == "__main__":
    main()
