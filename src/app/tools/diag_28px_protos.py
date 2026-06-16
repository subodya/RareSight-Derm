"""
DECISIVE DIAGNOSTIC: does matching the prototype domain to the query domain (28x28)
recover accuracy? Builds IMAGE-ONLY prototypes from DermaMNIST TRAIN at 28x28 (upscaled
through the SAME BiomedCLIP preprocess as queries) and evaluates on:
  (a) DermaMNIST TEST  (2005 imgs - full statistical power)
  (b) derma_samples/   (35 imgs - the user's manual test set)
Compares against the DEPLOYED full-res prototypes on the same 28x28 queries.
"""
import os, sys, glob
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from src.models.raresight_net import RareSight

device = "cuda" if torch.cuda.is_available() else "cpu"
model = RareSight(device=device)
wp = os.path.join(ROOT, "checkpoints", "raresight_nblk4mix.pth")
model.load_state_dict(torch.load(wp, map_location=device), strict=False)
model.eval()

def encode_arr(imgs_uint8, bs=128):
    """imgs_uint8: (N,28,28,3) -> normalized embeddings (N,512) via BiomedCLIP preprocess."""
    embs = []
    for i in range(0, len(imgs_uint8), bs):
        batch = imgs_uint8[i:i+bs]
        tens = torch.stack([model.preprocess(Image.fromarray(a)) for a in batch]).to(device)
        with torch.no_grad():
            e = model.backbone.encode_image(tens)
            e = e / e.norm(dim=-1, keepdim=True)
        embs.append(e.cpu())
    return torch.cat(embs)

d = np.load(os.path.join(ROOT, "data", "raw", "dermamnist.npz"))
tr_x, tr_y = d["train_images"], d["train_labels"].ravel()
te_x, te_y = d["test_images"], d["test_labels"].ravel()

# Build 28x28 image-only prototypes from TRAIN, K=20 per class (matches deployed K).
rng = np.random.RandomState(42)
protos = []
for c in range(7):
    idx = np.where(tr_y == c)[0]
    sel = rng.choice(idx, size=min(20, len(idx)), replace=False)
    e = encode_arr(tr_x[sel])
    p = e.mean(0); p = p / p.norm()
    protos.append(p)
P28 = torch.stack(protos)  # (7,512)

def evaluate(P, X, Y, name):
    E = encode_arr(X)
    cos = E @ P.T
    pred = cos.argmax(1).numpy()
    acc = (pred == Y).mean()
    # macro-F1
    f1s = []
    for c in range(7):
        tp = ((pred == c) & (Y == c)).sum()
        fp = ((pred == c) & (Y != c)).sum()
        fn = ((pred != c) & (Y == c)).sum()
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1s.append(2*prec*rec/(prec+rec+1e-9))
    print(f"\n[{name}] acc={acc*100:.2f}%  macroF1={np.mean(f1s)*100:.2f}%  (N={len(Y)})")
    for c in range(7):
        n = (Y==c).sum()
        print(f"   cls{c}: {((pred==c)&(Y==c)).sum()}/{n}  f1={f1s[c]*100:.1f}")
    return acc

# Deployed full-res prototypes for comparison
dep = torch.load(os.path.join(ROOT, "src", "app", "assets", "disease_prototypes.pt"), map_location="cpu")
Pdep = torch.stack([dep[c] if c in dep else dep[str(c)] for c in range(7)]).float()
Pdep = Pdep / Pdep.norm(dim=-1, keepdim=True)

print("="*60)
print("DermaMNIST TEST (28x28 queries, 2005 imgs):")
evaluate(Pdep, te_x, te_y, "DEPLOYED full-res protos")
evaluate(P28,  te_x, te_y, "NEW 28x28 protos (domain-matched)")

# derma_samples
folders = sorted(os.listdir(os.path.join(ROOT, "derma_samples")))
sx, sy = [], []
for folder in folders:
    c = int(folder.split("_")[0])
    for p in sorted(glob.glob(os.path.join(ROOT, "derma_samples", folder, "*.png"))):
        sx.append(np.array(Image.open(p).convert("RGB"))); sy.append(c)
sx = np.stack(sx); sy = np.array(sy)
print("\n" + "="*60)
print("derma_samples (28x28, 35 imgs):")
evaluate(Pdep, sx, sy, "DEPLOYED full-res protos")
evaluate(P28,  sx, sy, "NEW 28x28 protos (domain-matched)")
