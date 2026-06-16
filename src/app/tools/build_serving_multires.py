"""
Multi-resolution serving builder. Produces resolution-BANDED deployment artifacts so the
app recognises lesions at any input size (the 600x450-only prototypes rejected/garbled
28-112px inputs; see thesis/SESSION_resolution_diagnosis.md).

For each band in {28,56,112,224,450}: downsample train-support / train-maha / val images to
that band (then the BiomedCLIP preprocess upscales to 224, exactly mirroring an upload of
that native size), encode, and fit:
  - M3/CoOp-blended prototypes (same beta/lam/gap/text as deployed)
  - Mahalanobis OOD means + shared inv-cov + tau (10% in-dist false-abstain on val)
  - calibration temperatures T_img / T_meta
  - metadata-fusion alpha (tuned per band on val macro-F1)
Metadata log-likelihood TABLES (age/sex/site | class) are resolution-independent -> built once.

Outputs:
  src/app/assets/disease_prototypes_multi.pt  -> {band: {cls_id: tensor(512)}}
  src/app/assets/serving_params_multi.pt       -> {bands, shared{...}, per_band{band:{...}}}

CPU-conscious sample sizes (MAHA_PER_CLASS, VAL cap) keep the 5-band sweep tractable.
Run: python src/app/tools/build_serving_multires.py
"""
import sys, os, numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.models.raresight_net import RareSight
import src.app.tools.build_serving_artifacts as bsa

BANDS = [28, 56, 112, 224, 450]
N_CLASSES = 7
SEED = 42
ROOT = bsa.os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_ROOT = os.path.join(ROOT, "data", "ham10000")
# env overrides (default = deployed paths -> behaviour unchanged unless RS_* set)
CKPT = os.environ.get("RS_CKPT", os.path.join(ROOT, "checkpoints", "raresight_nblk4mix.pth"))
_ASSET = os.environ.get("RS_OUT_DIR", os.path.join(ROOT, "src", "app", "assets"))
os.makedirs(_ASSET, exist_ok=True)
BLEND_PATH = os.environ.get("RS_BLEND", os.path.join(ROOT, "src", "app", "assets", "blend_params.pt"))
OUT_PROTO = os.path.join(_ASSET, "disease_prototypes_multi.pt")
OUT_SERVE = os.path.join(_ASSET, "serving_params_multi.pt")

K_SHOT = 20
MAHA_PER_CLASS = 150     # reduced from 400 (CPU); 5 bands x 7 classes
VAL_CAP = 700            # cap val images encoded per band (calibration + tau)
ALPHA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
FALSE_ABSTAIN = 0.02


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def downsample(pil, res):
    if res >= min(pil.size):
        return pil
    return pil.resize((res, res), Image.BILINEAR)


def encode_paths(model, dev, paths, band, bs=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            ims = [downsample(Image.open(p).convert("RGB"), band) for p in paths[i:i+bs]]
            b = torch.stack([model.preprocess(im) for im in ims]).to(dev)
            out.append(_norm(model.backbone.encode_image(b)).cpu())
    return torch.cat(out)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")
    m = RareSight(device=dev)
    m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    temp = m.temperature.item()

    blend = torch.load(BLEND_PATH, map_location=dev, weights_only=False)
    BETA, LAM, GAP, TXT = blend["beta"], blend["lam"], blend["gap"].to(dev), blend["text_embs"]

    def apply_blend(img_proto, c):
        ts = TXT[c].to(dev) + LAM * GAP; ts = ts / ts.norm()
        p = BETA * img_proto + (1.0 - BETA) * ts
        return (p / p.norm()).cpu()

    sp = bsa.load_split_meta(DATA_ROOT, seed=SEED)
    logtab, si, xi = bsa.fit_meta_logtab(sp["train"])   # resolution-independent

    # fixed selections (shared across bands so only resolution differs)
    rng = np.random.RandomState(SEED)
    tr = sp["train"].reset_index(drop=True)
    supp_idx, maha_idx = {}, {}
    for c in range(N_CLASSES):
        idx = tr.index[tr["label"] == c].to_numpy()
        supp_idx[c] = rng.choice(idx, size=min(K_SHOT, len(idx)), replace=False)
        maha_idx[c] = rng.choice(idx, size=min(MAHA_PER_CLASS, len(idx)), replace=False)

    val = sp["val"].reset_index(drop=True)
    if len(val) > VAL_CAP:
        vsel = rng.choice(val.index.to_numpy(), size=VAL_CAP, replace=False)
        val = val.iloc[vsel].reset_index(drop=True)
    yv = val["label"].values
    mll_val = bsa.meta_loglik(val, logtab, si, xi)      # resolution-independent

    protos_multi, per_band = {}, {}
    for band in BANDS:
        print(f"\n=== band {band}px ===")
        # prototypes
        pb = {}
        for c in range(N_CLASSES):
            e = encode_paths(m, dev, tr.loc[supp_idx[c], "path"].tolist(), band)
            ip = _norm(e.mean(0, keepdim=True))[0].to(dev)
            pb[c] = apply_blend(ip, c)
        protos_multi[band] = pb
        P = torch.stack([_norm(pb[c]) for c in range(N_CLASSES)]).to(dev)

        # maha fit on band-downsampled train
        mp, my = [], []
        for c in range(N_CLASSES):
            mp += tr.loc[maha_idx[c], "path"].tolist(); my += [c] * len(maha_idx[c])
        Ftr = encode_paths(m, dev, mp, band).numpy(); my = np.array(my)
        means = np.stack([Ftr[my == c].mean(0) for c in range(N_CLASSES)])
        centered = np.concatenate([Ftr[my == c] - means[c] for c in range(N_CLASSES)])
        cov = np.cov(centered, rowvar=False) + 1e-3 * np.eye(Ftr.shape[1])
        inv_cov = np.linalg.inv(cov)

        # val encode -> logits, cos, meta fusion, calibration, tau
        Qv = encode_paths(m, dev, val["path"].tolist(), band).to(dev)
        z = (-torch.cdist(Qv, P) * temp).cpu().numpy()
        # tune alpha per band on val macro-F1
        best_a, best_f1 = 0.0, -1
        for a in ALPHA_GRID:
            f1 = bsa.macroF1(yv, (z + a * mll_val).argmax(1))
            if f1 > best_f1:
                best_f1, best_a = f1, a
        fused = z + best_a * mll_val

        def fit_T(lg):
            bT, bn = 1.0, 1e9
            for T in np.concatenate([np.linspace(0.05, 1.0, 96), np.linspace(1.05, 10, 90)]):
                p = bsa.softmax_np(lg / T)
                nll = -np.mean(np.log(p[np.arange(len(yv)), yv] + 1e-12))
                if nll < bn:
                    bn, bT = nll, float(T)
            return bT
        T_img, T_meta = fit_T(z), fit_T(fused)

        # maha tau: 10% in-dist false-abstain on val
        diff = Qv.cpu().numpy()[:, None, :] - means[None, :, :]   # (Nv, C, D)
        d = np.einsum("nci,ij,ncj->nc", diff, inv_cov, diff)
        val_score = -d.min(1)
        tau = float(np.quantile(val_score, FALSE_ABSTAIN))

        per_band[band] = {
            "maha_means": means.tolist(), "maha_inv_cov": inv_cov.tolist(),
            "ood_method": "maha", "ood_tau": round(tau, 4),
            "calib_T_img": round(T_img, 4), "calib_T_meta": round(T_meta, 4),
            "meta_alpha": best_a, "meta_val_macroF1": round(best_f1 * 100, 2),
            "ece_after_img_test": round(bsa.ece(bsa.softmax_np(z / T_img), yv), 4),
            "ece_after_meta_test": round(bsa.ece(bsa.softmax_np(fused / T_meta), yv), 4),
        }
        print(f"  alpha*={best_a} valF1={best_f1*100:.1f}  T_img={T_img:.3f} T_meta={T_meta:.3f}  tau={tau:.3f}")

    serving = {
        "bands": BANDS,
        "shared": {
            "meta_logtab": {k: v.tolist() for k, v in logtab.items()},
            "meta_site_index": si, "meta_sex_index": xi, "temp_metric": round(temp, 4),
            "ood_false_abstain": FALSE_ABSTAIN,
        },
        "per_band": per_band,
    }
    torch.save(protos_multi, OUT_PROTO)
    torch.save(serving, OUT_SERVE)
    print(f"\nSaved -> {OUT_PROTO}\nSaved -> {OUT_SERVE}")


if __name__ == "__main__":
    main()
