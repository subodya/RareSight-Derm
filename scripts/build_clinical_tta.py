"""
Rebuild the deployed CLINICAL path under TTA (the only Tier-1 keeper — see
thesis/CLINICAL_improvement_options.md §3b). Clinical path ONLY; dermoscopy untouched.

What changes (all under TTA flip/rotation-averaged encoding of PAD-train):
  - clinical_prototypes.pt        : M3 prototypes rebuilt on TTA features
  - clinical_serving_params.pt    : maha_means / maha_inv_cov / ood_tau / temp_metric / calib_T
                                    recomputed on TTA features
What is PRESERVED unchanged (encoding-independent class-conditional priors):
  - meta_logtab, meta_sex_index, meta_site_index, meta_alpha, meta_source, calib_T_meta

Backs up nothing here (caller already backed up *_backup_20260612_pretta.pt). Verifies the
rebuilt path reproduces the experiment's 66.61% on PAD-TEST before saving.

Run:  python scripts/build_clinical_tta.py
"""
import sys, os, json, numpy as np, torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.models.raresight_net import RareSight
from src.data.pad_ufes import load_pad_ufes, HAM_NAMES

CKPT = os.path.join(ROOT, "checkpoints", "raresight_nblk4mix.pth")
BLEND = os.path.join(ROOT, "src/app/assets/blend_params.pt")
OUT_PROTO = os.path.join(ROOT, "src/app/assets/clinical_prototypes.pt")
OUT_SERVING = os.path.join(ROOT, "src/app/assets/clinical_serving_params.pt")
BETA, LAM, FALSE_ABSTAIN = 0.75, 1.0, 0.02


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def _tta_views(img):
    return [img, img.transpose(Image.FLIP_LEFT_RIGHT), img.transpose(Image.FLIP_TOP_BOTTOM),
            img.transpose(Image.ROTATE_90), img.transpose(Image.ROTATE_180),
            img.transpose(Image.ROTATE_270)]


def encode_tta(model, dev, paths):
    out = []
    with torch.no_grad():
        for p in paths:
            views = _tta_views(Image.open(p).convert("RGB"))
            b = torch.cat([model.preprocess(v).unsqueeze(0) for v in views]).to(dev)
            e = _norm(model.backbone.encode_image(b)).mean(0)
            out.append(_norm(e).cpu())
    return torch.stack(out).numpy().astype(np.float32)


def maha_score(X, means, inv):
    diff = X[:, None, :] - means[None]
    return -np.einsum("nci,ij,ncj->nc", diff, inv, diff).min(1)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}  (TTA clinical rebuild)")
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    temp = float(m.temperature.item())
    blend = torch.load(BLEND, map_location="cpu", weights_only=False)
    text_embs = blend["text_embs"]
    prev = torch.load(OUT_SERVING, map_location="cpu", weights_only=False)  # for preserved meta tables

    pad = load_pad_ufes(verbose=False)
    train, test = pad["train"], pad["test"]
    classes = sorted(set(int(c) for c in train["label"].unique()))
    Xtr = encode_tta(m, dev, train["path"].tolist()); ytr = train["label"].values.astype(int)
    print(f"TTA-encoded PAD-train ({len(ytr)}), classes={classes}")

    # M3 prototypes on TTA features
    img_protos = {c: _norm(torch.tensor(Xtr[ytr == c].mean(0))) for c in classes}
    img_mean = torch.stack([img_protos[c] for c in classes]).mean(0)
    txt_mean = torch.stack([_norm(text_embs[c]) for c in classes]).mean(0)
    gap = _norm(img_mean - txt_mean)
    protos = {c: _norm(BETA * img_protos[c] + (1.0 - BETA) * _norm(text_embs[c] + LAM * gap)) for c in classes}

    # Mahalanobis OOD + tau on TTA features
    means = np.stack([Xtr[ytr == c].mean(0) for c in classes])
    centered = np.concatenate([Xtr[ytr == c] - Xtr[ytr == c].mean(0) for c in classes])
    cov = np.cov(centered, rowvar=False) + 1e-3 * np.eye(Xtr.shape[1])
    inv = np.linalg.inv(cov)
    tau = float(np.quantile(maha_score(Xtr, means, inv), FALSE_ABSTAIN))

    # calibration T on TTA train prototype logits (same fit as build_clinical_path)
    P = torch.stack([protos[c] for c in classes]).numpy()
    N = 7
    def proto_logits(X):
        z = -np.linalg.norm(X[:, None, :] - P[None], axis=2) * temp
        full = np.full((len(X), N), -1e9); full[:, classes] = z
        return full
    ztr = proto_logits(Xtr)
    bT, bn = 1.0, 1e9
    for T in np.concatenate([np.linspace(0.05, 1.0, 96), np.linspace(1.05, 10, 90)]):
        e = np.exp(ztr / T - (ztr / T).max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
        nll = -np.mean(np.log(p[np.arange(len(ytr)), ytr] + 1e-12))
        if nll < bn: bn, bT = nll, float(T)

    # verify on PAD-TEST before saving
    yte = test["label"].values.astype(int)
    keep = np.isin(yte, classes)
    te_paths = [p for p, k in zip(test["path"].tolist(), keep) if k]; yte = yte[keep]
    Xte = encode_tta(m, dev, te_paths)
    Pn = np.stack([_norm(protos[c]).numpy() for c in classes])
    pred = np.array(classes)[(Xte @ Pn.T).argmax(1)]
    acc = float((pred == yte).mean() * 100)
    ab = float((maha_score(Xte, means, inv) < tau).mean() * 100)
    print(f"\nVERIFY PAD-TEST: acc={acc:.2f}%  abstain={ab:.2f}%  (experiment target 66.61)")
    for c in classes:
        if (yte == c).any():
            print(f"  {HAM_NAMES[c]:22s}: {float((pred[yte==c]==c).mean())*100:5.1f}%  (n={(yte==c).sum()})")

    # save: rebuilt image-dependent fields + PRESERVED metadata tables
    serving = {"maha_means": means.tolist(), "maha_inv_cov": inv.tolist(),
               "ood_tau": round(tau, 4), "ood_method": "maha", "temp_metric": round(temp, 4),
               "calib_T": round(bT, 4), "calib_T_img": round(bT, 4), "classes": classes,
               "encoding": "tta_flip_rot6"}
    for k in ("meta_logtab", "meta_sex_index", "meta_site_index", "meta_alpha",
              "meta_source", "calib_T_meta"):
        if k in prev:
            serving[k] = prev[k]
    torch.save(protos, OUT_PROTO)
    torch.save(serving, OUT_SERVING)
    print(f"\nSaved TTA clinical artifacts -> {os.path.relpath(OUT_PROTO, ROOT)} + serving "
          f"(meta tables preserved: {'meta_logtab' in serving})")


if __name__ == "__main__":
    main()
