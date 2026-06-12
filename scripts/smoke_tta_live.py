"""Smoke-test the LIVE inference.predict() after the TTA clinical wiring.
Confirms: (1) clinical PAD images route to the clinical path and are TTA-encoded, accuracy
sane; (2) a HAM dermoscopy image still routes dermoscopy (TTA NOT applied). No files written."""
import sys, os, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src/app/backend"))
from PIL import Image
import inference
from src.data.pad_ufes import load_pad_ufes

R = inference.load_resources()
print("clinical_serving encoding flag:", R["clinical_serving"].get("encoding"))

pad = load_pad_ufes(verbose=False)
test = pad["test"]; classes = [0, 1, 2, 4, 5]
test = test[test["label"].isin(classes)].reset_index(drop=True)
rng = np.random.RandomState(0)
idx = rng.choice(len(test), size=40, replace=False)

n_clin = n_correct = 0
for i in idx:
    row = test.iloc[i]
    img = Image.open(row["path"]).convert("RGB")
    out = inference.predict(img, R)  # image-only (no metadata) — honest path
    if out["modality"] == "clinical":
        n_clin += 1
        n_correct += int(out["top_class_id"] == int(row["label"]))
print(f"\nPAD-test sample (n=40): routed clinical={n_clin}/40, "
      f"clinical-path acc={100*n_correct/max(n_clin,1):.1f}% (image-only, ~66% expected)")

# dermoscopy: a HAM image should route dermoscopy (TTA not applied there)
ham = None
for d in ("data/ham10000/images", "data/ham10000"):
    p = os.path.join(ROOT, d)
    if os.path.isdir(p):
        for f in os.listdir(p):
            if f.lower().endswith((".jpg", ".png")):
                ham = os.path.join(p, f); break
    if ham: break
if ham:
    o = inference.predict(Image.open(ham).convert("RGB"), R)
    print(f"HAM dermoscopy sample: modality={o['modality']} band={o['resolution_band']} "
          f"top={o['top_class_name']}  (expect dermoscopy)")
else:
    print("HAM sample not found — skipped dermoscopy check")
