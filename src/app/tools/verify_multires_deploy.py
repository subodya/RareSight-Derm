"""
End-to-end verification of the DEPLOYED resolution-routing path (inference.predict).
Drives the real backend predict() — which now auto-routes a query to its resolution band —
on HAM test images downsampled to each band, plus the user's derma_samples. Reports per
resolution: top-1 accuracy AND the OOD-abstain rate (must NOT be ~100% on valid images).
Confirms (a) low-res recovers, (b) full-res does NOT regress.
"""
import os, sys, glob
import numpy as np
from PIL import Image

HERE = os.path.dirname(__file__)
BACKEND = os.path.abspath(os.path.join(HERE, "..", "backend"))
sys.path.insert(0, BACKEND)
import inference as inf
sys.path.insert(0, inf.PROJECT_ROOT)
from src.data.preprocessing import load_ham10000

res = inf.load_resources()
print("multi-band loaded:", res.get("serving_multi") is not None,
      "| bands:", res["serving_multi"]["bands"] if res.get("serving_multi") else None)

HAM = os.path.join(inf.PROJECT_ROOT, "data", "ham10000")
te_paths, te_y = load_ham10000(HAM, split="test", seed=42); te_y = np.array(te_y)
rng = np.random.RandomState(0)
keep = []
for c in range(7):
    idx = np.where(te_y == c)[0]
    keep += rng.choice(idx, size=min(30, len(idx)), replace=False).tolist()
keep = np.array(keep)
imgs = [Image.open(te_paths[i]).convert("RGB") for i in keep]
ys = te_y[keep]

def downs(p, r):
    return p if r >= min(p.size) else p.resize((r, r), Image.BILINEAR)

print("\nHAM test (30/class), through deployed predict() with routing:")
print(" res | band | acc   | abstain%")
for r in [28, 56, 112, 224, 450]:
    correct = abstain = 0
    bandsel = None
    for img, y in zip(imgs, ys):
        out = inf.predict(downs(img, r), res)
        bandsel = out["resolution_band"]
        if out["top_class_id"] == y:
            correct += 1
        if out["is_unknown"]:
            abstain += 1
    n = len(ys)
    print(f" {r:3d} | {str(bandsel):>4} | {correct/n*100:5.1f} | {abstain/n*100:5.1f}")

# derma_samples (real 28x28)
folders = sorted(os.listdir(os.path.join(inf.PROJECT_ROOT, "derma_samples")))
sc = sa = st = 0
for folder in folders:
    c = int(folder.split("_")[0])
    for p in sorted(glob.glob(os.path.join(inf.PROJECT_ROOT, "derma_samples", folder, "*.png"))):
        out = inf.predict(Image.open(p).convert("RGB"), res)
        st += 1
        if out["top_class_id"] == c: sc += 1
        if out["is_unknown"]: sa += 1
print(f"\nderma_samples (real 28x28): acc={sc}/{st}={sc/st*100:.1f}%  abstain={sa}/{st}={sa/st*100:.1f}%")
