"""
Re-tune the M3 blend beta PER RESOLUTION BAND (training-free diagnostic).

Deployed prototypes are   normalize( beta * img_proto + (1-beta) * txt_shift_c )
with a SINGLE global beta (blend_params.pt). The resolution-diagnosis session found
that at 28px, pure image protos (beta=1) scored 45.3% vs 38.8% for the blended
deployed beta -> the text blend that helps at full-res HURTS at low-res. This script
quantifies that across all bands and finds the best beta per band.

Method (faithful to build_serving_multires.py, no re-embedding per beta):
  * img_proto[c]  = mean of K=20 band-downsampled support embeddings (same seed/selection)
  * txt_shift[c]  = normalize(text_emb_c + lam * gap)        (band-independent)
  * for each beta in a grid: blend, classify eval by cosine-nearest prototype.
  * SELECT beta on the VAL split, REPORT on TEST (no model-selection leak).

Bands evaluated on HAM (dermoscopy, the deployed band source). Band 28 ALSO evaluated
on REAL DermaMNIST-test@28 (the citable low-res number) using DermaMNIST-train protos
-- this is the exact 45.3-vs-38.8 axis.

This script only DIAGNOSES + recommends a per-band beta schedule. It does NOT modify
deployment; wiring a per-band beta in means rebuilding disease_prototypes_multi.pt
(build_serving_multires) with band-specific beta, then re-verifying OOD/calibration.

Run:  python src/app/tools/retune_blend_beta.py
"""
import sys, os, json
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.models.raresight_net import RareSight
import src.app.tools.build_serving_artifacts as bsa

BANDS      = [28, 56, 112, 224, 450]
N_CLASSES  = 7
SEED       = 42
K_SHOT     = 20
EVAL_CAP   = 700
BETA_GRID  = np.round(np.linspace(0.0, 1.0, 21), 2)   # 0.00 .. 1.00 step 0.05

ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_ROOT  = os.path.join(ROOT, "data", "ham10000")
DM_ROOT    = os.path.join(ROOT, "data", "raw")
CKPT       = os.path.join(ROOT, "checkpoints", "raresight_nblk4mix.pth")
BLEND_PATH = os.path.join(ROOT, "src", "app", "assets", "blend_params.pt")
OUT_JSON   = os.path.join(ROOT, "thesis", "beta_retune_results.json")

dev = "cuda" if torch.cuda.is_available() else "cpu"


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def downsample(pil, res):
    return pil if res >= min(pil.size) else pil.resize((res, res), Image.BILINEAR)


def encode_paths(model, paths, band, bs=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            ims = [downsample(Image.open(p).convert("RGB"), band) for p in paths[i:i + bs]]
            b = torch.stack([model.preprocess(im) for im in ims]).to(dev)
            out.append(_norm(model.backbone.encode_image(b)).cpu())
    return torch.cat(out)


def encode_arrays(model, arrs, band, bs=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(arrs), bs):
            ims = [downsample(Image.fromarray(a).convert("RGB"), band) for a in arrs[i:i + bs]]
            b = torch.stack([model.preprocess(im) for im in ims]).to(dev)
            out.append(_norm(model.backbone.encode_image(b)).cpu())
    return torch.cat(out)


def acc(Q, protos, y):
    pred = (Q @ protos.t()).argmax(1).numpy()
    return 100.0 * (pred == y).mean()


def sweep_beta(img_proto, txt_shift, Qv, yv, Qt, yt):
    """Return (best_beta_by_val, val@best, test@best, val_curve, test_curve)."""
    val_curve, test_curve = {}, {}
    for beta in BETA_GRID:
        protos = torch.stack([
            _norm(beta * img_proto[c] + (1.0 - beta) * txt_shift[c]) for c in range(N_CLASSES)
        ])
        val_curve[float(beta)] = acc(Qv, protos, yv)
        test_curve[float(beta)] = acc(Qt, protos, yt)
    best_beta = max(val_curve, key=val_curve.get)
    return best_beta, val_curve[best_beta], test_curve[best_beta], val_curve, test_curve


def main():
    print(f"device={dev}")
    m = RareSight(device=dev)
    m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()

    blend = torch.load(BLEND_PATH, map_location=dev, weights_only=False)
    BETA_G, LAM, GAP, TXT = blend["beta"], blend["lam"], blend["gap"].to(dev), blend["text_embs"]
    print(f"Deployed GLOBAL beta = {BETA_G:.3f}  lam = {LAM:.3f}")

    # band-independent text shift per class
    txt_shift = {c: _norm(TXT[c].to(dev) + LAM * GAP).cpu() for c in range(N_CLASSES)}

    sp = bsa.load_split_meta(DATA_ROOT, seed=SEED)
    tr, va, te = sp["train"].reset_index(drop=True), sp["val"].reset_index(drop=True), sp["test"].reset_index(drop=True)
    rng = np.random.RandomState(SEED)
    supp_idx = {c: rng.choice(tr.index[tr["label"] == c].to_numpy(),
                              size=min(K_SHOT, int((tr["label"] == c).sum())), replace=False)
                for c in range(N_CLASSES)}

    def cap(df):
        if len(df) <= EVAL_CAP:
            return df
        sel = rng.choice(df.index.to_numpy(), size=EVAL_CAP, replace=False)
        return df.iloc[sel].reset_index(drop=True)
    va_c, te_c = cap(va), cap(te)
    yv, yt = va_c["label"].values, te_c["label"].values

    results = {"global_beta": float(BETA_G), "bands": {}}
    print(f"\n{'band':>6} {'beta*':>6} {'val@b*':>7} {'test@b*':>8} "
          f"{'test@global':>12} {'gain':>6}")
    for band in BANDS:
        img_proto = {}
        for c in range(N_CLASSES):
            e = encode_paths(m, tr.loc[supp_idx[c], "path"].tolist(), band)
            img_proto[c] = _norm(e.mean(0, keepdim=True))[0]
        Qv = encode_paths(m, va_c["path"].tolist(), band)
        Qt = encode_paths(m, te_c["path"].tolist(), band)
        bb, va_b, te_b, vc, tc = sweep_beta(img_proto, txt_shift, Qv, yv, Qt, yt)
        te_global = tc[min(tc, key=lambda b: abs(b - BETA_G))]   # nearest grid beta to global
        results["bands"][band] = {"best_beta": bb, "val_at_best": round(va_b, 2),
                                  "test_at_best": round(te_b, 2),
                                  "test_at_global": round(te_global, 2),
                                  "gain": round(te_b - te_global, 2),
                                  "test_curve": {str(k): round(v, 2) for k, v in tc.items()}}
        print(f"{band:>6} {bb:>6.2f} {va_b:>7.2f} {te_b:>8.2f} {te_global:>12.2f} "
              f"{te_b - te_global:>+6.2f}")

    # ── REAL DermaMNIST-28 (the citable low-res number, reproduces 45.3-vs-38.8 axis) ──
    print("\nDermaMNIST-28 (real low-res capture)...")
    import medmnist
    from medmnist import INFO
    DC = getattr(medmnist, INFO["dermamnist"]["python_class"])
    dm_tr = DC(split="train", download=True, root=DM_ROOT)
    dm_va = DC(split="val",   download=True, root=DM_ROOT)
    dm_te = DC(split="test",  download=True, root=DM_ROOT)
    dm_tr_i, dm_tr_y = dm_tr.imgs, dm_tr.labels.flatten()
    rng2 = np.random.RandomState(SEED)
    dm_sup = {c: rng2.choice(np.where(dm_tr_y == c)[0],
                             size=min(K_SHOT, int((dm_tr_y == c).sum())), replace=False)
              for c in range(N_CLASSES) if (dm_tr_y == c).any()}
    dm_img_proto = {}
    for c in range(N_CLASSES):
        if c in dm_sup:
            e = encode_arrays(m, [dm_tr_i[i] for i in dm_sup[c]], 28)
            dm_img_proto[c] = _norm(e.mean(0, keepdim=True))[0]
        else:
            dm_img_proto[c] = txt_shift[c]   # no DM data -> text only
    Qv_dm = encode_arrays(m, list(dm_va.imgs), 28); yv_dm = dm_va.labels.flatten()
    Qt_dm = encode_arrays(m, list(dm_te.imgs), 28); yt_dm = dm_te.labels.flatten()
    bb, va_b, te_b, vc, tc = sweep_beta(dm_img_proto, txt_shift, Qv_dm, yv_dm, Qt_dm, yt_dm)
    te_global = tc[min(tc, key=lambda b: abs(b - BETA_G))]
    results["dermamnist28"] = {"best_beta": bb, "val_at_best": round(va_b, 2),
                               "test_at_best": round(te_b, 2),
                               "test_at_global": round(te_global, 2),
                               "gain": round(te_b - te_global, 2),
                               "test_curve": {str(k): round(v, 2) for k, v in tc.items()}}
    print(f"{'DM-28':>6} {bb:>6.2f} {va_b:>7.2f} {te_b:>8.2f} {te_global:>12.2f} "
          f"{te_b - te_global:>+6.2f}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # ── verdict ──
    print(f"\n{'='*60}\nVERDICT\n{'='*60}")
    print(f"Deployed global beta = {BETA_G:.2f}")
    print("Per-band best beta (higher beta = more IMAGE weight):")
    for band in BANDS:
        b = results["bands"][band]
        print(f"  {band:>4}px: beta* {b['best_beta']:.2f}  test {b['test_at_global']:.2f} "
              f"-> {b['test_at_best']:.2f}  ({b['gain']:+.2f}pp)")
    d = results["dermamnist28"]
    print(f"  DM-28 : beta* {d['best_beta']:.2f}  test {d['test_at_global']:.2f} "
          f"-> {d['test_at_best']:.2f}  ({d['gain']:+.2f}pp)")
    print(f"\nSaved -> {OUT_JSON}")
    print("\nNOTE: diagnostic only. To deploy a per-band beta, rebuild "
          "disease_prototypes_multi.pt with band-specific beta, then re-verify OOD/calibration.")


if __name__ == "__main__":
    main()
