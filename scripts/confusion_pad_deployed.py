"""
Confusion matrices for the DEPLOYED clinical path on PAD-UFES-20 PAD-TEST (nblk4mix encoder,
src/app/assets clinical artifacts). Produces image-only and +metadata (alpha tuned on a
patient-grouped val carve of PAD-TRAIN) 5x5 confusion matrices, a PNG figure, and a JSON.

No deployed file is modified. Run:  python scripts/confusion_pad_deployed.py
"""
import sys, os, json, numpy as np, torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.models.raresight_net import RareSight
from src.data.pad_ufes import load_pad_ufes, HAM_NAMES

N = 7
CKPT = os.path.join(ROOT, "checkpoints", "raresight_nblk4mix.pth")
CLIN_PROTO = os.path.join(ROOT, "src/app/assets/clinical_prototypes.pt")
CLIN_SERVING = os.path.join(ROOT, "src/app/assets/clinical_serving_params.pt")
FIG = os.path.join(ROOT, "figures", "pad_confusion_deployed.png")
OUT = os.path.join(ROOT, "checkpoints", "pad_confusion_deployed.json")
SEXES = ["male", "female", "unknown"]
ALPHA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
SMOOTH = 1.0
SHORT = {0: "akiec", 1: "bcc", 2: "bkl", 4: "mel", 5: "nv"}


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def encode(model, dev, paths, bs=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            b = torch.cat([model.preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                           for p in paths[i:i + bs]]).to(dev)
            out.append(_norm(model.backbone.encode_image(b)).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def age_bin(a):
    try:
        a = float(a)
    except (TypeError, ValueError):
        return None
    return None if a != a else int(min(max(a, 0), 89) // 10)


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
    return {k: np.log(v / v.sum(1, keepdims=True)) for k, v in tab.items()}, si, xi


def loglik(df, logtab, si, xi):
    out = np.zeros((len(df), N))
    for i, (_, r) in enumerate(df.iterrows()):
        ab = age_bin(r["age"])
        if ab is not None:
            out[i] += logtab["age"][:, ab]
        out[i] += logtab["sex"][:, xi.get(r["sex"], xi["unknown"])]
        out[i] += logtab["site"][:, si.get(r["site"], si["unknown"])]
    return out


def confmat(y, pred, classes):
    k = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((k, k), int)
    for t, p in zip(y, pred):
        if p in idx:
            M[idx[t], idx[p]] += 1
    return M


def plot(ax, M, classes, title):
    row = M.sum(1, keepdims=True)
    pct = M / np.maximum(row, 1) * 100
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    labs = [SHORT[c] for c in classes]
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(labs)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(labs)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{M[i, j]}\n{pct[i, j]:.0f}%", ha="center", va="center",
                    color="white" if pct[i, j] > 55 else "black", fontsize=8)
    return im


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    cs = torch.load(CLIN_SERVING, map_location="cpu", weights_only=False)
    temp = float(cs["temp_metric"]); classes = [int(c) for c in cs["classes"]]
    protos = torch.load(CLIN_PROTO, map_location="cpu", weights_only=False)
    P = torch.stack([_norm(protos[c]) for c in classes]).numpy()

    pad = load_pad_ufes()
    train_all, test = pad["train"], pad["test"]
    gss = GroupShuffleSplit(1, test_size=0.25, random_state=42)
    tr_i, va_i = next(gss.split(train_all, train_all["label"], groups=train_all["patient_id"]))
    train, val = train_all.iloc[tr_i].reset_index(drop=True), train_all.iloc[va_i].reset_index(drop=True)
    sites = sorted(set(train_all["site"]) | {"unknown"})
    logtab, si, xi = fit_logtab(train, sites)

    def proto_logits(X):
        z = -np.linalg.norm(X[:, None, :] - P[None], axis=2) * temp
        full = np.full((len(X), N), -1e9); full[:, classes] = z
        return full

    print(f"Encoding PAD val ({len(val)}) + test ({len(test)})...")
    Xv = encode(m, dev, val["path"].tolist()); yv = val["label"].values.astype(int)
    Xt = encode(m, dev, test["path"].tolist()); yt = test["label"].values.astype(int)
    zv, zt = proto_logits(Xv), proto_logits(Xt)
    mv, mt = loglik(val, logtab, si, xi), loglik(test, logtab, si, xi)

    mf1 = lambda y, p: f1_score(y, p, average="macro", labels=classes, zero_division=0) * 100
    best_a, best = 0.0, -1
    for a in ALPHA_GRID:
        f = mf1(yv, (zv + a * mv).argmax(1))
        if f > best:
            best, best_a = f, a

    pred_img = zt.argmax(1)
    pred_meta = (zt + best_a * mt).argmax(1)
    M_img = confmat(yt, pred_img, classes)
    M_meta = confmat(yt, pred_meta, classes)

    def summarize(tag, pred, M):
        acc = float((pred == yt).mean() * 100); f1 = mf1(yt, pred)
        print(f"\n{tag}: acc {acc:.2f}%  macroF1 {f1:.2f}")
        print("   true\\pred  " + " ".join(f"{SHORT[c]:>5}" for c in classes) + "   recall")
        for i, c in enumerate(classes):
            rec = M[i, i] / max(M[i].sum(), 1) * 100
            print(f"   {SHORT[c]:>8}  " + " ".join(f"{M[i, j]:5d}" for j in range(len(classes)))
                  + f"   {rec:5.1f}% (n={M[i].sum()})")
        return acc, f1

    print("\n" + "=" * 64)
    print("DEPLOYED CLINICAL PATH on PAD-TEST (n=638) — CONFUSION MATRICES")
    print("=" * 64)
    print(f"alpha* (val-tuned) = {best_a}")
    a_img, f_img = summarize("IMAGE-ONLY", pred_img, M_img)
    a_meta, f_meta = summarize("+ METADATA", pred_meta, M_meta)

    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    im = plot(axes[0], M_img, classes, f"Image-only\nacc {a_img:.1f}%  macro-F1 {f_img:.1f}")
    plot(axes[1], M_meta, classes, f"+ Metadata (age/sex/site)\nacc {a_meta:.1f}%  macro-F1 {f_meta:.1f}")
    fig.suptitle("RareSight deployed clinical path — PAD-UFES-20 test (n=638, patient-grouped)", fontsize=12)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04, label="row-normalized %")
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure -> {os.path.relpath(FIG, ROOT)}")

    json.dump({"classes": [SHORT[c] for c in classes], "n_test": int(len(yt)), "alpha": best_a,
               "image_only": {"acc": round(a_img, 2), "macroF1": round(f_img, 2), "confusion": M_img.tolist()},
               "with_metadata": {"acc": round(a_meta, 2), "macroF1": round(f_meta, 2), "confusion": M_meta.tolist()}},
              open(OUT, "w"), indent=2)
    print(f"Saved data   -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
