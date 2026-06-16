"""
Build the MODALITY ROUTER: a logistic-regression probe (dermoscopy=0, clinical/smartphone=1)
on BiomedCLIP embeddings, used to route an upload to the dermoscopy band-path or the clinical
path. Orthogonal to the (training-free) aligned-space classification — it only picks the path.

Key fix (advisor check #1): a full-res-trained probe mis-routes LOW-RES dermoscopy to clinical
(derma_samples routed clinical 57% -> would bypass the band-28 fix). So the dermoscopy class is
trained MULTI-RESOLUTION: HAM at {28,56,112,224,450} + genuine DermaMNIST-28. Clinical = PAD-TRAIN
(native + a downsampled share for low-res smartphones).

Biased operating point (advisor check #2): default dermoscopy; route clinical only when
P(clinical) > T. T is picked on a held-out VAL (not test) as the smallest threshold keeping
dermoscopy-val false-route <= TARGET_FALSE_ROUTE, so the primary domain is protected.

Output: src/app/assets/modality_probe.pt = {coef, intercept, T, classes, meta}
Run: python src/app/tools/build_modality_probe.py
"""
import sys, os, numpy as np, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.models.raresight_net import RareSight
from src.data.preprocessing import load_ham10000
from src.data.pad_ufes import load_pad_ufes

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
_ASSET = os.environ.get("RS_OUT_DIR", os.path.join(ROOT, "src", "app", "assets"))
os.makedirs(_ASSET, exist_ok=True)
OUT = os.path.join(_ASSET, "modality_probe.pt")
BANDS = [28, 56, 112, 224, 450]
SEED = 42
N_HAM, N_DERMA, N_PAD = 700, 350, 900
TARGET_FALSE_ROUTE = 0.01     # max dermoscopy-val images allowed to route clinical
rng = np.random.RandomState(SEED)

dev = "cuda" if torch.cuda.is_available() else "cpu"
_CKPT = os.environ.get("RS_CKPT", os.path.join(ROOT, "checkpoints", "raresight_nblk4mix.pth"))
m = RareSight(device=dev); m.load_state_dict(torch.load(_CKPT, map_location=dev), strict=False); m.eval()

def _norm(x): return x / x.norm(dim=-1, keepdim=True)
def enc(pils, bs=64):
    out=[]
    with torch.no_grad():
        for i in range(0,len(pils),bs):
            t=torch.stack([m.preprocess(p) for p in pils[i:i+bs]]).to(dev)
            out.append(_norm(m.backbone.encode_image(t)).cpu())
    return torch.cat(out).numpy()
def ham_at(path, r):
    im=Image.open(path).convert("RGB")
    return im.resize((r,r), Image.BILINEAR) if r < min(im.size) else im

# ---- dermoscopy class (multi-res HAM + DermaMNIST-28) ----
hp, hy = load_ham10000(os.path.join(ROOT,"data","ham10000"), split="train", seed=SEED)
hsel = rng.choice(len(hp), size=N_HAM, replace=False)
derm_pils = [ham_at(hp[i], BANDS[j % len(BANDS)]) for j,i in enumerate(hsel)]
d = np.load(os.path.join(ROOT,"data","raw","dermamnist.npz"))
dtr = d["train_images"]; dsel = rng.choice(len(dtr), size=N_DERMA, replace=False)
derm_pils += [Image.fromarray(dtr[i]) for i in dsel]

# ---- clinical class (PAD-train, native + downsampled share) ----
pad_tr = load_pad_ufes(verbose=False)["train"]
psel = pad_tr.sample(n=min(N_PAD, len(pad_tr)), random_state=SEED).reset_index(drop=True)
clin_pils=[]
for k,(_,r) in enumerate(psel.iterrows()):
    im=Image.open(r["path"]).convert("RGB")
    if k % 3 == 0:  # 1/3 downsampled to a low band -> low-res smartphone robustness
        b=BANDS[k % 3]  # 28 or 56
        im=im.resize((b,b), Image.BILINEAR)
    clin_pils.append(im)

Xderm = enc(derm_pils); Xclin = enc(clin_pils)
X = np.concatenate([Xderm, Xclin]); y = np.concatenate([np.zeros(len(Xderm)), np.ones(len(Xclin))])

# train/val split for clean T selection (check #2)
idx = rng.permutation(len(X)); ntr = int(0.8*len(X))
tri, vai = idx[:ntr], idx[ntr:]
probe = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(X[tri], y[tri])

# pick T on VAL: smallest threshold keeping dermoscopy-val false-route <= TARGET
pv = probe.predict_proba(X[vai])[:,1]; yv = y[vai]
derm_v = pv[yv==0]
T = 0.5
for cand in np.linspace(0.5, 0.99, 50):
    if (derm_v > cand).mean() <= TARGET_FALSE_ROUTE:
        T = float(cand); break
val_clin_catch = (pv[yv==1] > T).mean()*100
val_derm_falseroute = (derm_v > T).mean()*100
print(f"VAL: T={T:.3f}  dermoscopy false-route={val_derm_falseroute:.1f}%  clinical catch={val_clin_catch:.1f}%")

torch.save({
    "coef": probe.coef_.astype(np.float32), "intercept": float(probe.intercept_[0]),
    "T": round(T,4), "classes": {"dermoscopy":0, "clinical":1},
    "meta": {"target_false_route": TARGET_FALSE_ROUTE, "n_derm": len(Xderm), "n_clin": len(Xclin),
             "val_clinical_catch": round(val_clin_catch,1), "val_derm_falseroute": round(val_derm_falseroute,1)},
}, OUT)
print(f"saved -> {OUT}")

# ---- verify low-res dermoscopy now STAYS dermoscopy (the bug we are fixing) ----
import glob
def proba(pils):
    Z = enc(pils); return 1/(1+np.exp(-(Z@probe.coef_[0]+probe.intercept_[0])))
hte, hyy = load_ham10000(os.path.join(ROOT,"data","ham10000"), split="test", seed=SEED)
tsel = rng.choice(len(hte), size=200, replace=False)
print("\n[verify] low-res dermoscopy routing (want routed-clinical LOW):")
for r in [28,56,112,450]:
    pc = proba([ham_at(hte[i], r) for i in tsel])
    print(f"  HAM@{r:3d}px: mean P(clin)={pc.mean():.2f}  routed-clinical@T={ (pc>T).mean()*100:4.1f}%")
ds=[Image.fromarray(np.array(Image.open(p).convert('RGB'))) for f in sorted(os.listdir(os.path.join(ROOT,'derma_samples'))) for p in sorted(glob.glob(os.path.join(ROOT,'derma_samples',f,'*.png')))]
pcd=proba(ds)
print(f"  derma_samples(28px real): mean P(clin)={pcd.mean():.2f}  routed-clinical@T={(pcd>T).mean()*100:4.1f}%")
