"""
RESOLUTION SWEEP + ROBUST-PROTOTYPE FIX.

Q1 (characterise the cliff): take HAM test images (full-res 600x450), downsample each to
    {28,56,112,224,450}px, push through the DEPLOYED full-res prototypes, plot accuracy.
Q2 (test the fix): build RESOLUTION-ROBUST prototypes — support images encoded at a MIX of
    resolutions (28..450) so the prototype manifold covers blur — and re-run the same sweep.
    Goal: a flat, high curve = "recognise the disease at any level" (user's requirement).

All training-free. HAM test split == the one used for the 57% headline (load_ham10000 seed 42).
"""
import os, sys
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from src.models.raresight_net import RareSight
from src.data.preprocessing import load_ham10000

device = "cuda" if torch.cuda.is_available() else "cpu"
model = RareSight(device=device)
model.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "raresight_nblk4mix.pth"),
                                 map_location=device), strict=False)
model.eval()
HAM = os.path.join(ROOT, "data", "ham10000")
RES_LEVELS = [28, 56, 112, 224, 450]
rng = np.random.RandomState(42)

def downsample(pil, res):
    """Simulate a low-res capture: shrink to res x res then the model.preprocess upscales to 224."""
    if res >= min(pil.size):
        return pil
    return pil.resize((res, res), Image.BILINEAR)

def encode_pils(pils, bs=64):
    out = []
    for i in range(0, len(pils), bs):
        t = torch.stack([model.preprocess(p) for p in pils[i:i+bs]]).to(device)
        with torch.no_grad():
            e = model.backbone.encode_image(t); e = e / e.norm(dim=-1, keepdim=True)
        out.append(e.cpu())
    return torch.cat(out)

def macro_f1(pred, y):
    f1 = []
    for c in range(7):
        tp=((pred==c)&(y==c)).sum(); fp=((pred==c)&(y!=c)).sum(); fn=((pred!=c)&(y==c)).sum()
        p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); f1.append(2*p*r/(p+r+1e-9))
    return np.mean(f1)

# ---- load test images (cap per class for CPU speed) ----
te_paths, te_y = load_ham10000(HAM, split="test", seed=42)
te_y = np.array(te_y)
PER_CLASS = 40
keep = []
for c in range(7):
    idx = np.where(te_y == c)[0]
    keep.extend(rng.choice(idx, size=min(PER_CLASS, len(idx)), replace=False).tolist())
keep = np.array(keep)
test_pils_full = [Image.open(te_paths[i]).convert("RGB") for i in keep]
test_y = te_y[keep]
print(f"Test imgs: {len(test_y)} (<= {PER_CLASS}/class)")

# ---- prototypes A: deployed full-res ----
dep = torch.load(os.path.join(ROOT, "src","app","assets","disease_prototypes.pt"), map_location="cpu")
Pdep = torch.stack([dep[c] if c in dep else dep[str(c)] for c in range(7)]).float()
Pdep = Pdep / Pdep.norm(dim=-1, keepdim=True)

# ---- prototypes B: resolution-robust (multi-res support from TRAIN) ----
tr_paths, tr_y = load_ham10000(HAM, split="train", seed=42)
tr_y = np.array(tr_y)
K = 20
robust = []
for c in range(7):
    idx = rng.choice(np.where(tr_y==c)[0], size=min(K, (tr_y==c).sum()), replace=False)
    pils = []
    for j, i in enumerate(idx):
        p = Image.open(tr_paths[i]).convert("RGB")
        # cycle each support image through the resolution levels -> manifold covers blur
        r = RES_LEVELS[j % len(RES_LEVELS)]
        pils.append(downsample(p, r))
    e = encode_pils(pils); proto = e.mean(0); proto = proto/proto.norm()
    robust.append(proto)
Probust = torch.stack(robust)

# ---- sweep ----
print("\nres |  deployed full-res protos  |  resolution-robust protos")
print("    |   acc      macroF1         |   acc      macroF1")
for res in RES_LEVELS:
    pils = [downsample(p, res) for p in test_pils_full]
    E = encode_pils(pils)
    pa = (E @ Pdep.T).argmax(1).numpy()
    pb = (E @ Probust.T).argmax(1).numpy()
    print(f"{res:3d} |  {(pa==test_y).mean()*100:5.1f}%   {macro_f1(pa,test_y)*100:5.1f}%        "
          f"|  {(pb==test_y).mean()*100:5.1f}%   {macro_f1(pb,test_y)*100:5.1f}%")
