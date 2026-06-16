"""
End-to-end verification of the FULL deployed routing (resolution band x modality) through
inference.predict(). Confirms:
  (a) dermoscopy (HAM multi-res + derma_samples) routes DERMOSCOPY, accuracy preserved;
  (b) smartphone (PAD-TEST) routes CLINICAL, accuracy recovered (~62% vs 23% always-derm);
  (c) CHECK #3: far-OOD / garbage uploads still ABSTAIN even when routed clinical (looser tau).
"""
import os, sys, glob
import numpy as np
from PIL import Image

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "backend")))
import inference as inf
sys.path.insert(0, inf.PROJECT_ROOT)
from src.data.preprocessing import load_ham10000
from src.data.pad_ufes import load_pad_ufes

res = inf.load_resources()
print("probe:", res.get("modality_probe") is not None,
      "| clinical:", res.get("clinical_protos") is not None,
      "| T:", res["modality_probe"]["T"] if res.get("modality_probe") else None)

def downs(p, r):
    return p if r >= min(p.size) else p.resize((r, r), Image.BILINEAR)

# ---- (a) HAM dermoscopy multi-res ----
hp, hy = load_ham10000(os.path.join(inf.PROJECT_ROOT, "data", "ham10000"), split="test", seed=42); hy = np.array(hy)
rng = np.random.RandomState(0); sel = []
for c in range(7): sel += rng.choice(np.where(hy == c)[0], size=min(25, (hy == c).sum()), replace=False).tolist()
imgs = [Image.open(hp[i]).convert("RGB") for i in sel]; ys = hy[sel]
print("\n(a) HAM dermoscopy through deployed predict():")
print(" res | acc  | %routed-clinical | abstain%")
for r in [28, 112, 450]:
    cor = clin = ab = 0
    for img, y in zip(imgs, ys):
        o = inf.predict(downs(img, r), res)
        cor += (o["top_class_id"] == y); clin += (o["modality"] == "clinical"); ab += o["is_unknown"]
    nn = len(ys)
    print(f" {r:3d} | {cor/nn*100:4.1f} | {clin/nn*100:5.1f}            | {ab/nn*100:4.1f}")

# ---- (b) PAD smartphone ----
pad = load_pad_ufes(verbose=False)["test"]; pp = pad["path"].tolist(); yp = pad["label"].values.astype(int)
cor = clin = ab = 0
for p, y in zip(pp, yp):
    o = inf.predict(Image.open(p).convert("RGB"), res)
    cor += (o["top_class_id"] == y); clin += (o["modality"] == "clinical"); ab += o["is_unknown"]
nP = len(yp)
print(f"\n(b) PAD smartphone (n={nP}): acc={cor/nP*100:.1f}%  routed-clinical={clin/nP*100:.1f}%  abstain={ab/nP*100:.1f}%")

# ---- derma_samples ----
sc = sclin = st = 0
for f in sorted(os.listdir(os.path.join(inf.PROJECT_ROOT, "derma_samples"))):
    c = int(f.split("_")[0])
    for p in sorted(glob.glob(os.path.join(inf.PROJECT_ROOT, "derma_samples", f, "*.png"))):
        o = inf.predict(Image.open(p).convert("RGB"), res); st += 1
        sc += (o["top_class_id"] == c); sclin += (o["modality"] == "clinical")
print(f"    derma_samples(28px): acc={sc/st*100:.1f}%  routed-clinical={sclin/st*100:.1f}%")

# ---- (c) CHECK #3: far-OOD must abstain regardless of routed path ----
print("\n(c) FAR-OOD / garbage (must abstain):")
samples = []
webp = os.path.join(inf.PROJECT_ROOT, "dermafibroma-sample.webp")
if os.path.exists(webp): samples.append(("wide-field webp", Image.open(webp).convert("RGB")))
samples.append(("random RGB noise", Image.fromarray(np.random.randint(0, 256, (224, 224, 3), np.uint8))))
samples.append(("solid grey", Image.new("RGB", (224, 224), (127, 127, 127))))
for name, im in samples:
    o = inf.predict(im, res)
    print(f"   {name:18s}: modality={o['modality']:10s} p_clin={o['p_clinical']} "
          f"ood={o['ood_score']:.2f} -> {'ABSTAIN' if o['is_unknown'] else 'ACCEPTED (!)'}")
