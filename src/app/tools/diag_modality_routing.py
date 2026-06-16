"""
Validate DUAL-OOD MODALITY ROUTING before wiring it into the app.

Premise: a query's own embedding is in-distribution for its TRUE modality's OOD model and
out-of-distribution for the other. So we can route without a modality classifier:
  margin_path = ood_score_path - tau_path ;  pick argmax margin ; both<0 -> truly unknown.

Tests on:
  - HAM-TEST (dermoscopy, full-res -> band 450)  -> should route DERMOSCOPY
  - PAD-TEST (smartphone)                          -> should route CLINICAL
Reports routing accuracy + the resulting classification accuracy vs always-dermoscopy.
"""
import sys, os, numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.models.raresight_net import RareSight
from src.data.preprocessing import load_ham10000
from src.data.pad_ufes import load_pad_ufes, SHARED_HAM_CLASSES

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = RareSight(device=dev); m.load_state_dict(torch.load(os.path.join(ROOT,"checkpoints","raresight_nblk4mix.pth"), map_location=dev), strict=False); m.eval()

def _norm(x): return x / x.norm(dim=-1, keepdim=True)
def enc(paths_or_arrs, bs=48):
    out=[]
    with torch.no_grad():
        for i in range(0,len(paths_or_arrs),bs):
            chunk=paths_or_arrs[i:i+bs]
            pils=[Image.open(p).convert("RGB") if isinstance(p,str) else Image.fromarray(p) for p in chunk]
            t=torch.stack([m.preprocess(p) for p in pils]).to(dev)
            out.append(_norm(m.backbone.encode_image(t)).cpu())
    return torch.cat(out).numpy()

def maha_score(X, means, inv):
    diff = X[:,None,:]-means[None]
    return -np.einsum("nci,ij,ncj->nc", diff, inv, diff).min(1)

# dermoscopy path = band-450 of the multi artifact ; clinical path = checkpoints
sm = torch.load(os.path.join(ROOT,"src","app","assets","serving_params_multi.pt"), map_location="cpu", weights_only=False)
pm = torch.load(os.path.join(ROOT,"src","app","assets","disease_prototypes_multi.pt"), map_location="cpu")
temp = sm["shared"]["temp_metric"]
def derm_path(native):  # pick band like deployment
    bands=sm["bands"]; b=min(bands,key=lambda x:abs(np.log(x)-np.log(max(native,1))))
    pb=sm["per_band"][b]
    P=torch.stack([_norm(pm[b][c]) for c in range(7)]).numpy()
    return P, np.array(pb["maha_means"]), np.array(pb["maha_inv_cov"]), float(pb["ood_tau"])

clin_p = torch.load(os.path.join(ROOT,"checkpoints","clinical_prototypes.pt"), map_location="cpu", weights_only=False)
clin_s = torch.load(os.path.join(ROOT,"checkpoints","clinical_serving_params.pt"), map_location="cpu", weights_only=False)
clin_classes = clin_s["classes"]
Pc = {c: _norm(clin_p[c]).numpy() for c in clin_classes}
clin_means=np.array(clin_s["maha_means"]); clin_inv=np.array(clin_s["maha_inv_cov"]); clin_tau=float(clin_s["ood_tau"])
clin_means_classes = clin_s.get("classes", clin_classes)
# clinical prototype matrix + class mask on LOGITS (defined early; used by router + classifier)
Pc_mat=np.zeros((7,512)); valid=np.full(7,-1e9)
for c in clin_classes:
    Pc_mat[c]=Pc[c]; valid[c]=0.0

# ---- HAM dermoscopy test (full-res) ----
hp, hy = load_ham10000(os.path.join(ROOT,"data","ham10000"), split="test", seed=42); hy=np.array(hy)
rng=np.random.RandomState(0); sel=[]
for c in range(7): sel+=rng.choice(np.where(hy==c)[0],size=min(40,(hy==c).sum()),replace=False).tolist()
Xh=enc([hp[i] for i in sel]); yh=hy[sel]
Pd,dm,di,dt = derm_path(450)
md_h = maha_score(Xh, dm, di) - dt
mc_h = maha_score(Xh, clin_means, clin_inv) - clin_tau
route_h_derm = (md_h >= mc_h).mean()*100

# ---- PAD smartphone test ----
pad=load_pad_ufes(verbose=False)["test"]; pp=pad["path"].tolist(); yp=pad["label"].values.astype(int)
Xp=enc(pp)
# native res per PAD image
pad_native=np.array([min(Image.open(p).size) for p in pp])
md_p=np.array([maha_score(Xp[i:i+1], *derm_path(pad_native[i])[1:3] if False else (dm,di))[0] for i in range(len(Xp))]) - dt
mc_p = maha_score(Xp, clin_means, clin_inv) - clin_tau
route_p_clin = (mc_p > md_p).mean()*100

print("=== ROUTING SIGNAL COMPARISON (HAM should->DERM, PAD should->CLIN) ===")
print(f"[margin=ood-tau]  HAM->derm {route_h_derm:5.1f}% | PAD->clin {route_p_clin:5.1f}%")

# raw Mahalanobis DISTANCE (smaller = closer to that modality's manifold)
def maha_dist(X, means, inv):
    diff = X[:,None,:]-means[None]
    return np.einsum("nci,ij,ncj->nc", diff, inv, diff).min(1)
dd_h, dc_h = maha_dist(Xh,dm,di), maha_dist(Xh,clin_means,clin_inv)
dd_p, dc_p = maha_dist(Xp,dm,di), maha_dist(Xp,clin_means,clin_inv)
print(f"[raw maha dist]   HAM->derm {(dd_h<dc_h).mean()*100:5.1f}% | PAD->clin {(dc_p<dd_p).mean()*100:5.1f}%")

# max cosine to each path's prototypes (higher = better class+modality match)
def maxcos_derm(X): return (X@Pd.T).max(1)
def maxcos_clin(X):
    s=X@Pc_mat.T + valid[None,:]; return s.max(1)
cd_h, cc_h = maxcos_derm(Xh), maxcos_clin(Xh)
cd_p, cc_p = maxcos_derm(Xp), maxcos_clin(Xp)
print(f"[max cosine]      HAM->derm {(cd_h>=cc_h).mean()*100:5.1f}% | PAD->clin {(cc_p>cd_p).mean()*100:5.1f}%")
# z-normalised margin: standardise each OOD score by its own in-dist mean/std on the path's own data
print(f"\n  (HAM n={len(yh)}, PAD n={len(yp)})")

# classification under routing for PAD (5 shared classes); mask on LOGITS (Pc_mat/valid set above)
def clf_clin(X):
    s=X@Pc_mat.T + valid[None,:]   # cosine to clinical protos, invalid classes masked to -inf
    return s.argmax(1)
def clf_derm(X): return (X@Pd.T).argmax(1)

# ROUTER = raw Mahalanobis distance (route clinical when closer to clinical manifold).
route_clin_p = dc_p < dd_p
pred_p = np.where(route_clin_p, clf_clin(Xp), clf_derm(Xp))
acc_routed = (pred_p==yp).mean()*100
acc_alwaysderm = (clf_derm(Xp)==yp).mean()*100
acc_allclin = (clf_clin(Xp)==yp).mean()*100
print(f"\nPAD-TEST (smartphone) classification (n={len(yp)}, 5 shared classes):")
print(f"  always-dermoscopy : {acc_alwaysderm:.1f}%")
print(f"  always-clinical   : {acc_allclin:.1f}%")
print(f"  DUAL-OOD routed   : {acc_routed:.1f}%  ({route_clin_p.mean()*100:.0f}% routed clinical)")

# HAM-TEST (dermoscopy) under the SAME router — must NOT regress vs always-dermoscopy.
route_clin_h = dc_h < dd_h
pred_h = np.where(route_clin_h, clf_clin(Xh), clf_derm(Xh))
acc_ham_routed = (pred_h==yh).mean()*100
acc_ham_derm = (clf_derm(Xh)==yh).mean()*100
print(f"\nHAM-TEST (dermoscopy) classification (n={len(yh)}, 7-way):")
print(f"  always-dermoscopy : {acc_ham_derm:.1f}%")
print(f"  DUAL-OOD routed   : {acc_ham_routed:.1f}%  ({route_clin_h.mean()*100:.0f}% mis-routed clinical)")

# ---- DEDICATED MODALITY PROBE (logistic regression: dermoscopy=0, clinical=1) ----
from sklearn.linear_model import LogisticRegression
htr_p, htr_y = load_ham10000(os.path.join(ROOT,"data","ham10000"), split="train", seed=42)
ham_tr = rng.choice(len(htr_p), size=600, replace=False)
pad_tr_df = load_pad_ufes(verbose=False)["train"]
pad_tr = pad_tr_df.sample(n=min(600,len(pad_tr_df)), random_state=42)
Xtr_h = enc([htr_p[i] for i in ham_tr])
Xtr_p = enc(pad_tr["path"].tolist())
Xtr = np.concatenate([Xtr_h, Xtr_p]); ytr = np.concatenate([np.zeros(len(Xtr_h)), np.ones(len(Xtr_p))])
probe = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
# test separability
ph = probe.predict(Xh); pp = probe.predict(Xp)
print(f"\n[MODALITY PROBE]  HAM->derm {(ph==0).mean()*100:5.1f}% | PAD->clin {(pp==1).mean()*100:5.1f}%")
# BIASED routing: default dermoscopy; switch to clinical only when P(clinical) > T.
prob_clin_h = probe.predict_proba(Xh)[:,1]
prob_clin_p = probe.predict_proba(Xp)[:,1]
print("\n[BIASED probe routing — protect dermoscopy primary domain]")
print("  T    HAM mis-route%  HAM acc   PAD catch%  PAD acc")
for T in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
    rh = prob_clin_h > T; rp = prob_clin_p > T
    ah = (np.where(rh, clf_clin(Xh), clf_derm(Xh))==yh).mean()*100
    ap = (np.where(rp, clf_clin(Xp), clf_derm(Xp))==yp).mean()*100
    print(f"  {T:4.2f}   {rh.mean()*100:6.1f}       {ah:5.1f}     {rp.mean()*100:6.1f}     {ap:5.1f}")
print(f"  (baselines: HAM always-derm {acc_ham_derm:.1f}% | PAD always-clin {acc_allclin:.1f}%)")

# CHECK #1 (critical): do LOW-RES dermoscopy images stay dermoscopy under this full-res-trained probe?
import glob as _glob
def downs(arr_or_path, r):
    im = Image.open(arr_or_path).convert("RGB") if isinstance(arr_or_path,str) else Image.fromarray(arr_or_path)
    return np.array(im.resize((r,r), Image.BILINEAR)) if r < min(im.size) else np.array(im)
print("\n[CHECK#1] low-res DERMOSCOPY routing (want P(clinical) LOW -> stays dermoscopy):")
for r in [28, 56, 112]:
    Xr = enc([downs(hp[i], r) for i in sel])
    pc = probe.predict_proba(Xr)[:,1]
    print(f"  HAM@{r:3d}px: mean P(clin)={pc.mean():.2f}  routed-clinical@T0.7={ (pc>0.7).mean()*100:4.1f}%")
ds_imgs=[]
for folder in sorted(os.listdir(os.path.join(ROOT,"derma_samples"))):
    for p in sorted(_glob.glob(os.path.join(ROOT,"derma_samples",folder,"*.png"))):
        ds_imgs.append(np.array(Image.open(p).convert("RGB")))
Xds = enc(ds_imgs); pcd = probe.predict_proba(Xds)[:,1]
print(f"  derma_samples(28px real): mean P(clin)={pcd.mean():.2f}  routed-clinical@T0.7={(pcd>0.7).mean()*100:4.1f}%")
