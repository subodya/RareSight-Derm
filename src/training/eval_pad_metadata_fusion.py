"""
STEP 4 — METADATA FUSION on PAD-UFES-20 (independent clinical dataset).

Reinforces the metadata-fusion contribution on a SECOND, genuinely independent dataset: fits
class-conditional log-likelihoods log P(age|c), P(sex|c), P(site|c) on PAD-TRAIN (harmonized to
HAM vocab by src/data/pad_ufes.py), tunes the fusion weight alpha on a patient-grouped val carved
from PAD-TRAIN, and reports the macro-F1 / accuracy gain on the reserved PAD-TEST using the
clinical prototype logits from build_clinical_path.py (Step 2/3).

Run AFTER build_clinical_path.py:
    conda run -n raresight python src/training/eval_pad_metadata_fusion.py
"""
import sys, os, json, numpy as np, torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.data.pad_ufes import load_pad_ufes, HAM_NAMES, SHARED_HAM_CLASSES

N = 7
CKPT = "checkpoints/raresight_nblk4mix.pth"
CLIN_PROTO = "checkpoints/clinical_prototypes.pt"
CLIN_SERVING = "checkpoints/clinical_serving_params.pt"
OUT = "checkpoints/pad_metadata_fusion_results.json"
SEXES = ["male", "female", "unknown"]
ALPHA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
SMOOTH = 1.0


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def encode(model, dev, paths, bs=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            b = torch.cat([model.preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                           for p in paths[i:i+bs]]).to(dev)
            out.append(_norm(model.backbone.encode_image(b)).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def age_bin(a):
    try:
        a = float(a)
    except (TypeError, ValueError):
        return None
    if a != a:
        return None
    return int(min(max(a, 0), 89) // 10)


def fit_logtab(df, sites):
    si = {s: i for i, s in enumerate(sites)}
    xi = {s: i for i, s in enumerate(SEXES)}
    tab = {"age": np.full((N, 9), SMOOTH), "sex": np.full((N, len(SEXES)), SMOOTH),
           "site": np.full((N, len(sites)), SMOOTH)}
    for _, r in df.iterrows():
        c = int(r["label"]); ab = age_bin(r["age"])
        if ab is not None:
            tab["age"][c, ab] += 1
        tab["sex"][c, xi.get(r["sex"], xi["unknown"])] += 1
        tab["site"][c, si.get(r["site"], si["unknown"])] += 1
    logtab = {k: np.log(v / v.sum(1, keepdims=True)) for k, v in tab.items()}
    return logtab, si, xi


def loglik(df, logtab, si, xi):
    out = np.zeros((len(df), N))
    for i, (_, r) in enumerate(df.iterrows()):
        ab = age_bin(r["age"])
        if ab is not None:
            out[i] += logtab["age"][:, ab]
        out[i] += logtab["sex"][:, xi.get(r["sex"], xi["unknown"])]
        out[i] += logtab["site"][:, si.get(r["site"], si["unknown"])]
    return out


def main():
    if not (os.path.exists(CLIN_PROTO) and os.path.exists(CLIN_SERVING)):
        raise FileNotFoundError("Run src/training/build_clinical_path.py first (needs clinical prototypes).")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    cs = torch.load(CLIN_SERVING, map_location="cpu", weights_only=False)
    temp = float(cs["temp_metric"]); classes = cs["classes"]
    protos = torch.load(CLIN_PROTO, map_location="cpu")
    P = torch.stack([protos[c] for c in classes]).numpy()

    pad = load_pad_ufes()
    train_all, test = pad["train"], pad["test"]
    # patient-grouped val carved from PAD-TRAIN (for alpha tuning)
    gss = GroupShuffleSplit(1, test_size=0.25, random_state=42)
    tr_i, va_i = next(gss.split(train_all, train_all["label"], groups=train_all["patient_id"]))
    train, val = train_all.iloc[tr_i].reset_index(drop=True), train_all.iloc[va_i].reset_index(drop=True)

    sites = sorted(set(train_all["site"]) | {"unknown"})
    logtab, si, xi = fit_logtab(train, sites)

    def proto_logits(X):
        z = -np.linalg.norm(X[:, None, :] - P[None], axis=2) * temp
        full = np.full((len(X), N), -1e9); full[:, classes] = z
        return full

    print(f"\nEncoding PAD val ({len(val)}) + test ({len(test)})...")
    Xv = encode(m, dev, val["path"].tolist()); yv = val["label"].values.astype(int)
    Xt = encode(m, dev, test["path"].tolist()); yt = test["label"].values.astype(int)
    zv, zt = proto_logits(Xv), proto_logits(Xt)
    mv, mt = loglik(val, logtab, si, xi), loglik(test, logtab, si, xi)

    def macroF1(y, p):
        return f1_score(y, p, average="macro", labels=classes, zero_division=0)

    best_a, best_f1 = 0.0, -1
    for a in ALPHA_GRID:
        f1 = macroF1(yv, (zv + a * mv).argmax(1))
        if f1 > best_f1:
            best_f1, best_a = f1, a

    base_acc = float((zt.argmax(1) == yt).mean() * 100)
    base_f1 = macroF1(yt, zt.argmax(1)) * 100
    fused_pred = (zt + best_a * mt).argmax(1)
    fused_acc = float((fused_pred == yt).mean() * 100)
    fused_f1 = macroF1(yt, fused_pred) * 100

    print("\n" + "=" * 60)
    print("PAD METADATA FUSION (clinical path, PAD-TEST)")
    print("=" * 60)
    print(f"  alpha* (tuned on val macro-F1) = {best_a}")
    print(f"  image-only : acc {base_acc:5.2f}%  macroF1 {base_f1:5.2f}")
    print(f"  + metadata : acc {fused_acc:5.2f}%  macroF1 {fused_f1:5.2f}   "
          f"(Δacc {fused_acc-base_acc:+.2f}, ΔmacroF1 {fused_f1-base_f1:+.2f})")
    res = {"alpha": best_a, "n_test": len(yt),
           "image_only": {"acc": round(base_acc, 2), "macroF1": round(base_f1, 2)},
           "with_metadata": {"acc": round(fused_acc, 2), "macroF1": round(fused_f1, 2)},
           "delta_macroF1": round(fused_f1 - base_f1, 2), "delta_acc": round(fused_acc - base_acc, 2)}
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nSaved → {OUT}")
    if best_a == 0.0:
        print("Note: alpha*=0 → metadata adds nothing on top of the clinical prototypes here "
              "(consistent with the HAM finding that the image path can absorb the signal).")


if __name__ == "__main__":
    main()
