"""Find a compelling few-shot weak->strong case over the REAL HTTP path.

For each target class: reserve K held-out HAM images as the few-shot SUPPORT set,
then for a disjoint pool of QUERY images:
  1. baseline  POST /api/predict  (Dermoscopy, no note, no session)
  2. refine    POST /api/refine   (support set, class_id)
  3. after     POST /api/predict  (same query, WITH session_id)
Report queries that flip wrong->correct and are not referred/unknown after.

Backend on :8044. Support and query sets are disjoint (no leakage).
"""
import os, sys, glob, json
import pandas as pd
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE = "http://127.0.0.1:8044"
SHORT = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}
CLS2DX = {0: "actinic_keratoses", 1: "basal_cell_carcinoma", 2: "benign_keratosis-like_lesions",
          3: "dermatofibroma", 4: "melanoma", 5: "melanocytic_Nevi", 6: "vascular_lesions"}

K_SUPPORT = 5
N_QUERY = 20
TARGETS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1", "3", "2"])]

ham = pd.read_csv(os.path.join(ROOT, "data/ham10000/HAM10000_metadata.csv"))
img_idx = {os.path.splitext(os.path.basename(p))[0]: p
           for p in glob.glob(os.path.join(ROOT, "data/ham10000/**/*.jpg"), recursive=True)}
ham = ham[ham["image_id"].isin(img_idx)].copy()


def predict(path, session_id=None):
    data = {"scan_type": "Dermoscopy"}
    if session_id:
        data["session_id"] = session_id
    with open(path, "rb") as f:
        r = requests.post(BASE + "/api/predict", files={"file": f}, data=data, timeout=120)
    r.raise_for_status()
    j = r.json()
    top = j["predictions"][0]
    return {"id": j["top_class_id"], "name": top["class_name"], "prob": top["probability"],
            "modality": j["modality"], "refer": j["refer_to_specialist"], "unknown": j["is_unknown"]}


def refine(paths, cls_id, session_id=None):
    files = [("files", (os.path.basename(p), open(p, "rb"), "image/jpeg")) for p in paths]
    data = {"class_id": str(cls_id)}
    if session_id:
        data["session_id"] = session_id
    r = requests.post(BASE + "/api/refine", files=files, data=data, timeout=120)
    for _, fh in files:
        fh[1].close()
    r.raise_for_status()
    return r.json()["session_id"]


for cls in TARGETS:
    sub = ham[ham["dx"] == CLS2DX[cls]].sample(frac=1.0, random_state=42)
    paths = [img_idx[r.image_id] for r in sub.itertuples()]
    support, queries = paths[:K_SUPPORT], paths[K_SUPPORT:K_SUPPORT + N_QUERY]
    print(f"\n===== {SHORT[cls]} (class {cls}) — {len(support)} support, {len(queries)} queries =====")

    sess = refine(support, cls)
    flips = []
    for q in queries:
        b = predict(q)
        a = predict(q, session_id=sess)
        flag = ""
        if b["id"] != cls and a["id"] == cls and not a["refer"] and not a["unknown"]:
            flag = "  <== FLIP wrong->correct, not referred"
            flips.append((q, b, a))
        elif b["id"] != cls and a["id"] == cls:
            flag = "  (flip but referred/unknown after)"
        print(f"  {os.path.basename(q):<20} before={b['name'][:18]:<18}{b['prob']:.2f} "
              f"-> after={a['name'][:18]:<18}{a['prob']:.2f} "
              f"refer={a['refer']} unk={a['unknown']}{flag}")
    print(f"  CLEAN FLIPS for {SHORT[cls]}: {len(flips)}")
    for q, b, a in flips[:3]:
        print(f"    BEST: {os.path.basename(q)}  {b['name']} {b['prob']:.2f} -> {a['name']} {a['prob']:.2f}")
        print(f"          support: {[os.path.basename(s) for s in support]}")
