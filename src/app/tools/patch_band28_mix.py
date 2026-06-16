"""
Patch ONLY the 28px band of the multi-res artifacts to a MIXED low-res domain:
HAM-downsampled-28 (BILINEAR) + genuine DermaMNIST-train-28 (the MedMNIST crop/resize domain).

Rationale (advisor): both share native min-side 28 -> same band, so they can't be routed;
DermaMNIST is the only REAL 28px data, while HAM-downsampled-28 is synthetic. A within-resolution
mix covers both crop framings (dilution is mild at fixed resolution). Re-fits the band-28
prototypes + Mahalanobis OOD + tau + calibration on the mixed distribution, and reports the
generalizing DermaMNIST-TEST (2005-img) number, not just the 35-img derma_samples spot-check.
Bands >=56 are untouched (DermaMNIST only exists at 28px).
"""
import sys, os, numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.models.raresight_net import RareSight
import src.app.tools.build_serving_artifacts as bsa

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
CKPT = os.environ.get("RS_CKPT", os.path.join(ROOT, "checkpoints", "raresight_nblk4mix.pth"))
_ASSET = os.environ.get("RS_OUT_DIR", os.path.join(ROOT, "src", "app", "assets"))
BLEND = os.environ.get("RS_BLEND", os.path.join(ROOT, "src", "app", "assets", "blend_params.pt"))
NPZ = os.path.join(ROOT, "data", "raw", "dermamnist.npz")
OUT_PROTO = os.path.join(_ASSET, "disease_prototypes_multi.pt")
OUT_SERVE = os.path.join(_ASSET, "serving_params_multi.pt")
N, SEED, BAND = 7, 42, 28
K_HAM, K_DERMA = 20, 20
MAHA_HAM, MAHA_DERMA = 100, 100
ALPHA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
FALSE_ABSTAIN = 0.02


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    temp = m.temperature.item()
    blend = torch.load(BLEND, map_location=dev, weights_only=False)
    BETA, LAM, GAP, TXT = blend["beta"], blend["lam"], blend["gap"].to(dev), blend["text_embs"]

    def apply_blend(ip, c):
        ts = TXT[c].to(dev) + LAM * GAP; ts = ts / ts.norm()
        p = BETA * ip + (1.0 - BETA) * ts
        return (p / p.norm()).cpu()

    def enc_pils(pils, bs=64):
        out = []
        with torch.no_grad():
            for i in range(0, len(pils), bs):
                t = torch.stack([m.preprocess(p) for p in pils[i:i+bs]]).to(dev)
                out.append(_norm(m.backbone.encode_image(t)).cpu())
        return torch.cat(out)

    def ham28(p):  # HAM full-res -> 28px BILINEAR (synthetic low-res)
        im = Image.open(p).convert("RGB")
        return im.resize((28, 28), Image.BILINEAR)

    # data sources
    sp = bsa.load_split_meta(os.path.join(ROOT, "data", "ham10000"), seed=SEED)
    logtab, si, xi = bsa.fit_meta_logtab(sp["train"])
    tr = sp["train"].reset_index(drop=True)
    d = np.load(NPZ)
    dtr_x, dtr_y = d["train_images"], d["train_labels"].ravel()
    dva_x, dva_y = d["val_images"], d["val_labels"].ravel()
    dte_x, dte_y = d["test_images"], d["test_labels"].ravel()
    rng = np.random.RandomState(SEED)

    # ---- mixed prototypes ----
    protos, maha_emb, maha_y = {}, [], []
    for c in range(N):
        ham_idx = rng.choice(tr.index[tr["label"] == c].to_numpy(), size=K_HAM, replace=False)
        der_idx = rng.choice(np.where(dtr_y == c)[0], size=min(K_DERMA, (dtr_y == c).sum()), replace=False)
        pils = [ham28(tr.loc[i, "path"]) for i in ham_idx] + [Image.fromarray(dtr_x[i]) for i in der_idx]
        e = enc_pils(pils); ip = _norm(e.mean(0, keepdim=True))[0].to(dev)
        protos[c] = apply_blend(ip, c)
        # maha support (separate, larger)
        mham = rng.choice(tr.index[tr["label"] == c].to_numpy(), size=MAHA_HAM, replace=False)
        mder = rng.choice(np.where(dtr_y == c)[0], size=min(MAHA_DERMA, (dtr_y == c).sum()), replace=False)
        mp = [ham28(tr.loc[i, "path"]) for i in mham] + [Image.fromarray(dtr_x[i]) for i in mder]
        me = enc_pils(mp).numpy(); maha_emb.append(me); maha_y += [c] * len(me)
    P = torch.stack([_norm(protos[c]) for c in range(N)]).to(dev)
    F = np.concatenate(maha_emb); maha_y = np.array(maha_y)
    means = np.stack([F[maha_y == c].mean(0) for c in range(N)])
    centered = np.concatenate([F[maha_y == c] - means[c] for c in range(N)])
    inv_cov = np.linalg.inv(np.cov(centered, rowvar=False) + 1e-3 * np.eye(F.shape[1]))

    # ---- val for alpha/calibration/tau: DermaMNIST val + HAM-downsampled-28 val ----
    hv = sp["val"].reset_index(drop=True)
    hv_sel = rng.choice(hv.index.to_numpy(), size=min(400, len(hv)), replace=False)
    val_pils = [ham28(hv.loc[i, "path"]) for i in hv_sel] + [Image.fromarray(dva_x[i]) for i in range(len(dva_x))]
    yv = np.concatenate([hv.loc[hv_sel, "label"].values, dva_y])
    # metadata only available for HAM val rows; DermaMNIST has none -> meta term zeros there
    mll = np.zeros((len(yv), N))
    mll[:len(hv_sel)] = bsa.meta_loglik(hv.loc[hv_sel], logtab, si, xi)
    Qv = enc_pils(val_pils).to(dev)
    z = (-torch.cdist(Qv, P) * temp).cpu().numpy()
    best_a, best_f1 = 0.0, -1
    for a in ALPHA_GRID:
        f1 = bsa.macroF1(yv, (z + a * mll).argmax(1))
        if f1 > best_f1:
            best_f1, best_a = f1, a
    fused = z + best_a * mll

    def fit_T(lg):
        bT, bn = 1.0, 1e9
        for T in np.concatenate([np.linspace(0.05, 1.0, 96), np.linspace(1.05, 10, 90)]):
            p = bsa.softmax_np(lg / T)
            nll = -np.mean(np.log(p[np.arange(len(yv)), yv] + 1e-12))
            if nll < bn:
                bn, bT = nll, float(T)
        return bT
    T_img, T_meta = fit_T(z), fit_T(fused)
    diff = Qv.cpu().numpy()[:, None, :] - means[None]
    val_score = -np.einsum("nci,ij,ncj->nc", diff, inv_cov, diff).min(1)
    tau = float(np.quantile(val_score, FALSE_ABSTAIN))

    # ---- DermaMNIST TEST (generalizing 2005-img metric) ----
    Qt = enc_pils([Image.fromarray(x) for x in dte_x]).to(dev)
    pred = (Qt @ P.T).argmax(1).cpu().numpy()
    derma_test_acc = float((pred == dte_y).mean() * 100)
    print(f"band28 MIX: alpha*={best_a} valF1={best_f1*100:.1f} T_img={T_img:.3f} tau={tau:.3f}")
    print(f"DermaMNIST-TEST (2005 imgs) acc = {derma_test_acc:.2f}%")

    # ---- patch artifacts in place ----
    pm = torch.load(OUT_PROTO, map_location="cpu"); pm[BAND] = protos; torch.save(pm, OUT_PROTO)
    sm = torch.load(OUT_SERVE, map_location="cpu", weights_only=False)
    sm["per_band"][BAND] = {
        "maha_means": means.tolist(), "maha_inv_cov": inv_cov.tolist(),
        "ood_method": "maha", "ood_tau": round(tau, 4),
        "calib_T_img": round(T_img, 4), "calib_T_meta": round(T_meta, 4),
        "meta_alpha": best_a, "meta_val_macroF1": round(best_f1 * 100, 2),
        "ece_after_img_test": round(bsa.ece(bsa.softmax_np(z / T_img), yv), 4),
        "ece_after_meta_test": round(bsa.ece(bsa.softmax_np(fused / T_meta), yv), 4),
        "derma_mnist_test_acc": round(derma_test_acc, 2), "domain": "HAM28+DermaMNIST mix",
    }
    torch.save(sm, OUT_SERVE)
    print(f"patched band {BAND} -> {OUT_PROTO}, {OUT_SERVE}")


if __name__ == "__main__":
    main()
