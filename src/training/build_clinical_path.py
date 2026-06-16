"""
STEPS 2 + 3 — CLINICAL-MODALITY PATH (separate from the dermoscopy deployment).

The probe-revert lesson: do NOT overwrite the working dermoscopy artifacts. This builds a
PARALLEL clinical path from PAD-UFES-20 PAD-TRAIN clinical photos and saves it to its own
files, leaving disease_prototypes.pt / serving_params.pt untouched:

  STEP 2  clinical reference prototypes (M3 recipe on PAD-TRAIN clinical images, recomputing
          the modality gap from clinical embeddings) -> checkpoints/clinical_prototypes.pt
  STEP 3  clinical OOD + calibration (Mahalanobis on PAD-TRAIN clinical embeddings, tau, and a
          calibration temperature) so genuine clinical photos are IN-distribution for this path
          -> checkpoints/clinical_serving_params.pt

Then evaluates the clinical path on the reserved PAD-TEST and prints the BEFORE/AFTER vs Step 1
(dermoscopy path): clinical adaptation should drop the abstention rate and lift accuracy. Only
the 5 shared classes have PAD data; df/vasc remain dermoscopy-only (reported as such).

Run AFTER eval_cross_dataset_pad.py, with PAD-UFES-20 on disk:
    conda run -n raresight python src/training/build_clinical_path.py
"""
import sys, os, json, numpy as np, torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.data.pad_ufes import load_pad_ufes, HAM_NAMES, SHARED_HAM_CLASSES

N = 7
CKPT = os.environ.get("RS_CKPT", "checkpoints/raresight_nblk4mix.pth")
BLEND = os.environ.get("RS_BLEND", "src/app/assets/blend_params.pt")
SERVING = "src/app/assets/serving_params.pt"
_OUT = os.environ.get("RS_OUT_DIR", "checkpoints")
os.makedirs(_OUT, exist_ok=True)
OUT_PROTO = os.path.join(_OUT, "clinical_prototypes.pt")
OUT_SERVING = os.path.join(_OUT, "clinical_serving_params.pt")
OUT_RESULTS = os.path.join(_OUT, "clinical_path_results.json")
BETA, LAM = 0.75, 1.0          # M3 recipe (same as precompute.py)
FALSE_ABSTAIN = 0.02


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


def maha_score(X, means, inv):
    diff = X[:, None, :] - means[None]
    d = np.einsum("nci,ij,ncj->nc", diff, inv, diff)
    return -d.min(1)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    temp = float(m.temperature.item())

    pad = load_pad_ufes()
    train, test = pad["train"], pad["test"]

    print(f"\nEncoding PAD-TRAIN ({len(train)}) + PAD-TEST ({len(test)}) clinical images...")
    Xtr = encode(m, dev, train["path"].tolist()); ytr = train["label"].values.astype(int)
    Xte = encode(m, dev, test["path"].tolist());  yte = test["label"].values.astype(int)

    # Use only shared classes actually PRESENT in PAD-train (GroupShuffleSplit isn't
    # label-stratified, so a sparse class could land entirely in test). Degrade gracefully
    # to a smaller clinical path + warn, rather than KeyError after the encode.
    classes = [c for c in SHARED_HAM_CLASSES if (ytr == c).any()]
    missing = [c for c in SHARED_HAM_CLASSES if c not in classes]
    if missing:
        print(f"  WARNING: classes {[HAM_NAMES[c] for c in missing]} absent from PAD-train "
              f"→ clinical path is {len(classes)}-way (chance={100/len(classes):.0f}%).")

    # ── STEP 2: clinical prototypes (M3 recipe, gap recomputed from clinical images) ──
    blend = torch.load(BLEND, map_location="cpu", weights_only=False)
    text_embs = blend["text_embs"]                              # CuPL text per class (shared)
    img_protos = {c: _norm(torch.tensor(Xtr[ytr == c].mean(0))) for c in classes if (ytr == c).any()}
    img_mean = torch.stack([img_protos[c] for c in img_protos]).mean(0)
    txt_mean = torch.stack([_norm(text_embs[c]) for c in img_protos]).mean(0)
    gap = _norm(img_mean - txt_mean)
    clinical_protos = {}
    for c in img_protos:
        txt_shift = _norm(text_embs[c] + LAM * gap)
        clinical_protos[c] = _norm(BETA * img_protos[c] + (1.0 - BETA) * txt_shift)
    torch.save(clinical_protos, OUT_PROTO)
    print(f"STEP 2: clinical prototypes for classes {sorted(clinical_protos)} -> {OUT_PROTO}")

    # ── STEP 3: clinical OOD (Mahalanobis on clinical train) + calibration T ──
    means = np.stack([Xtr[ytr == c].mean(0) for c in classes])
    centered = np.concatenate([Xtr[ytr == c] - Xtr[ytr == c].mean(0) for c in classes])
    cov = np.cov(centered, rowvar=False) + 1e-3 * np.eye(Xtr.shape[1])
    inv = np.linalg.inv(cov)
    tau = float(np.quantile(maha_score(Xtr, means, inv), FALSE_ABSTAIN))

    P = torch.stack([clinical_protos[c] for c in classes]).numpy()
    cmask = np.full(N, -1e9)                                    # restrict to clinical classes
    def proto_logits(X):
        z = -np.linalg.norm(X[:, None, :] - P[None], axis=2) * temp   # (n, len(classes))
        full = np.full((len(X), N), -1e9); full[:, classes] = z
        return full
    # calibration T on train prototype logits (note: mildly optimistic; small dataset)
    ztr = proto_logits(Xtr)
    def fit_T(logits, y):
        bT, bn = 1.0, 1e9
        for T in np.concatenate([np.linspace(0.05, 1.0, 96), np.linspace(1.05, 10, 90)]):
            e = np.exp(logits / T - (logits / T).max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
            nll = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
            if nll < bn: bn, bT = nll, float(T)
        return bT
    T = fit_T(ztr, ytr)
    torch.save({"maha_means": means.tolist(), "maha_inv_cov": inv.tolist(), "ood_tau": round(tau, 4),
                "ood_method": "maha", "temp_metric": round(temp, 4), "calib_T": round(T, 4),
                "classes": classes}, OUT_SERVING)
    print(f"STEP 3: clinical OOD+calib (tau={tau:.3f}, T={T:.3f}) -> {OUT_SERVING}")

    # ── evaluate clinical path on PAD-TEST (before/after vs Step 1) ──
    zte = proto_logits(Xte)
    pred = zte.argmax(1)
    acc = float((pred == yte).mean() * 100)
    ab = maha_score(Xte, means, inv) < tau
    ab_rate = float(ab.mean() * 100)
    rec = {HAM_NAMES[c]: (round(float((pred[yte == c] == c).mean()) * 100, 1) if (yte == c).any() else None)
           for c in classes}
    print("\n" + "=" * 60)
    print("CLINICAL PATH on PAD-TEST (5-way clinical, chance=20%)")
    print("=" * 60)
    print(f"  accuracy            : {acc:5.2f}%")
    print(f"  OOD abstention rate : {ab_rate:5.2f}%  (vs Step-1 dermoscopy path — should be much lower)")
    for c in classes:
        print(f"  {HAM_NAMES[c]:22s}: {rec[HAM_NAMES[c]]}")
    res = {"n_train": len(ytr), "n_test": len(yte), "clinical_classes": classes,
           "accuracy": round(acc, 2), "ood_abstention_rate": round(ab_rate, 2),
           "per_class_recall": rec, "calib_T": round(T, 4), "ood_tau": round(tau, 4)}
    json.dump(res, open(OUT_RESULTS, "w"), indent=2)
    print(f"\nSaved → {OUT_RESULTS}")
    print("Note: df/vasc have no PAD clinical data — they stay dermoscopy-only. Dermoscopy "
          "deployment (disease_prototypes.pt/serving_params.pt) is UNCHANGED.")


if __name__ == "__main__":
    main()
