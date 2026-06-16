"""
Quick diagnostic: run the DEPLOYED backend predict() path over derma_samples/
(28x28 dermoscopy images, one folder per class) and report top-1 / top-3 accuracy,
per-class accuracy, confusion, OOD-flag rate, and the raw image-only argmax accuracy
(before metadata/calibration) so we can separate "model is wrong" from "fusion/OOD is wrong".
"""
import os, sys, glob
import numpy as np
from PIL import Image

HERE = os.path.dirname(__file__)
BACKEND = os.path.abspath(os.path.join(HERE, "..", "backend"))
sys.path.insert(0, BACKEND)
import inference as inf

ROOT = inf.PROJECT_ROOT
SAMPLES = os.path.join(ROOT, "derma_samples")

# folder index -> class id (they already match the model's class order)
folders = sorted(os.listdir(SAMPLES))
print("Folders:", folders)

res = inf.load_resources()
print("device:", res["device"])

n_classes = 7
conf = np.zeros((n_classes, n_classes), dtype=int)      # true x pred (deployed top1)
conf_img = np.zeros((n_classes, n_classes), dtype=int)  # true x pred (raw image-only argmax)
ood_flags = 0
total = 0
top3_correct = 0

for fi, folder in enumerate(folders):
    true_cls = int(folder.split("_")[0])
    paths = sorted(glob.glob(os.path.join(SAMPLES, folder, "*.png")))
    for p in paths:
        img = Image.open(p).convert("RGB")
        out = inf.predict(img, res)  # no metadata, no note -> pure image-only deployed path
        pred = out["top_class_id"]
        conf[true_cls, pred] += 1
        if out["is_unknown"]:
            ood_flags += 1
        top3 = [d["class_id"] for d in out["predictions"]]
        if true_cls in top3:
            top3_correct += 1
        total += 1

acc = np.trace(conf) / total
print(f"\n=== DEPLOYED image-only path (no metadata) ===")
print(f"Top-1 accuracy: {np.trace(conf)}/{total} = {acc*100:.2f}%")
print(f"Top-3 accuracy: {top3_correct}/{total} = {top3_correct/total*100:.2f}%")
print(f"OOD-flagged (is_unknown): {ood_flags}/{total}")
print("\nPer-class top-1 acc:")
for c in range(n_classes):
    n = conf[c].sum()
    print(f"  cls {c} {res['metadata'][str(c)]['name']:22s}: {conf[c,c]}/{n}")
print("\nConfusion (rows=true, cols=pred):")
print("      " + " ".join(f"{c:3d}" for c in range(n_classes)))
for c in range(n_classes):
    print(f"true{c} " + " ".join(f"{conf[c,j]:3d}" for j in range(n_classes)))
