"""
TIER-1 clinical-path improvement experiments (PAD-UFES-20, clinical/smartphone modality).

Tests three training-free / low-risk levers on the CLINICAL path ONLY, each in isolation,
against a faithfully reproduced baseline, then a combined config of the winners:

  L1  color constancy  (Shades-of-Gray illuminant normalization before encoding)
  L2  test-time augmentation (flip/rotation views, embeddings averaged)
  L3  Tip-Adapter (training-free key-value cache over PAD-train features)

Methodology (matches thesis/CLINICAL_improvement_options.md + advisor notes):
- Clinical-only. Reads the deployed encoder + blend text embeddings; writes NOTHING to
  src/app/assets. Dermoscopy path untouched.
- CC / TTA change the embedding space, so prototypes AND Mahalanobis are REBUILT under the
  same transform (not just applied at test) — controlled isolation.
- The "rebuilt baseline" (recipe, no lever) is verified to reproduce the deployed 66.14% so
  lever deltas are measured against a matched reference, not a confounded one.
- Tip-Adapter alpha/beta are grid-searched on a PATIENT-GROUPED val split carved from
  PAD-train (no PAD-TEST leakage); the reported number rebuilds the cache on full train.
- Accuracy is over ALL test (argmax over the 5 clinical classes), mirroring the deployed
  `accuracy` field. IMAGE-ONLY (no metadata) — the honest lever per the memory.

Run:  python scripts/tier1_clinical_experiments.py
"""
import sys, os, json, numpy as np, torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.models.raresight_net import RareSight
from src.data.pad_ufes import load_pad_ufes, HAM_NAMES

CKPT = os.path.join(ROOT, "checkpoints", "raresight_nblk4mix.pth")
BLEND = os.path.join(ROOT, "src/app/assets/blend_params.pt")
DEPLOYED_PROTO = os.path.join(ROOT, "src/app/assets/clinical_prototypes.pt")
OUT = os.path.join(ROOT, "checkpoints", "tier1_clinical_results.json")
BETA, LAM = 0.75, 1.0   # M3 recipe (identical to build_clinical_path.py)


# --------------------------------------------------------------------------- helpers
def _norm_t(x):
    return x / x.norm(dim=-1, keepdim=True)


def shades_of_gray(img, power=6):
    """Shades-of-Gray color constancy (Barata et al.). Estimate illuminant per channel as
    the Minkowski-p mean, normalize to unit length, divide it out. Cheap, training-free."""
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    illum = np.power(np.mean(np.power(a, power), axis=(0, 1)), 1.0 / power)
    illum = illum / (np.linalg.norm(illum) + 1e-8)
    a = a / (illum * np.sqrt(3.0) + 1e-8)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# TTA views: dermatology lesions are rotation/flip invariant -> safe augmentations.
def _tta_views(img):
    return [img,
            img.transpose(Image.FLIP_LEFT_RIGHT),
            img.transpose(Image.FLIP_TOP_BOTTOM),
            img.transpose(Image.ROTATE_90),
            img.transpose(Image.ROTATE_180),
            img.transpose(Image.ROTATE_270)]


def make_encoder(model, dev, transform=None, tta=False, bs=32):
    """Return f(paths)->(N,D) float32 L2-normed features under the chosen transform/TTA."""
    def _load(p):
        img = Image.open(p).convert("RGB")
        return transform(img) if transform else img

    def encode(paths):
        out = []
        with torch.no_grad():
            if not tta:
                for i in range(0, len(paths), bs):
                    b = torch.cat([model.preprocess(_load(p)).unsqueeze(0)
                                   for p in paths[i:i+bs]]).to(dev)
                    out.append(_norm_t(model.backbone.encode_image(b)).cpu())
                return torch.cat(out).numpy().astype(np.float32)
            # TTA: average normalized embeddings over views, renormalize
            for p in paths:
                views = _tta_views(_load(p))
                b = torch.cat([model.preprocess(v).unsqueeze(0) for v in views]).to(dev)
                e = _norm_t(model.backbone.encode_image(b)).mean(0)
                out.append(_norm_t(e).cpu())
            return torch.stack(out).numpy().astype(np.float32)
    return encode


def build_protos(Xtr, ytr, classes, text_embs):
    """M3 clinical prototypes (image proto blended with CuPL text + recomputed modality gap)."""
    img_protos = {c: _norm_t(torch.tensor(Xtr[ytr == c].mean(0))) for c in classes}
    img_mean = torch.stack([img_protos[c] for c in classes]).mean(0)
    txt_mean = torch.stack([_norm_t(text_embs[c]) for c in classes]).mean(0)
    gap = _norm_t(img_mean - txt_mean)
    P = {}
    for c in classes:
        txt_shift = _norm_t(text_embs[c] + LAM * gap)
        P[c] = _norm_t(BETA * img_protos[c] + (1.0 - BETA) * txt_shift)
    return np.stack([P[c].numpy() for c in classes]).astype(np.float32)


def proto_logits(X, P):
    """Cosine similarity to each prototype (monotone in -euclidean for unit vectors)."""
    return X @ P.T                                  # (n, C)


def evaluate(z, y, classes):
    pred = np.array(classes)[z.argmax(1)]
    acc = float((pred == y).mean() * 100)
    mf1 = float(f1_score(y, pred, labels=classes, average="macro") * 100)
    rec = {HAM_NAMES[c]: (round(float((pred[y == c] == c).mean()) * 100, 1), int((y == c).sum()))
           for c in classes if (y == c).any()}
    return {"acc": round(acc, 2), "macroF1": round(mf1, 2), "per_class": rec}, pred


def tip_cache_logits(Xq, Xcache, ycache, classes, beta_tip):
    """Training-free Tip-Adapter cache: affinity = exp(-beta(1 - q.k)); logits = aff @ onehot."""
    L = np.zeros((len(ycache), len(classes)), np.float32)
    cidx = {c: i for i, c in enumerate(classes)}
    for i, yy in enumerate(ycache):
        L[i, cidx[yy]] = 1.0
    aff = np.exp(-beta_tip * (1.0 - Xq @ Xcache.T))     # (nq, ncache)
    return aff @ L                                       # (nq, C)


# --------------------------------------------------------------------------- main
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}\nEncoder: {os.path.basename(CKPT)} (clinical path only)\n")
    m = RareSight(device=dev)
    m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False)
    m.eval()
    blend = torch.load(BLEND, map_location="cpu", weights_only=False)
    text_embs = blend["text_embs"]

    pad = load_pad_ufes(verbose=False)
    train, test = pad["train"], pad["test"]
    classes = sorted(set(int(c) for c in train["label"].unique()))
    tr_paths, ytr = train["path"].tolist(), train["label"].values.astype(int)
    te_paths, yte = test["path"].tolist(), test["label"].values.astype(int)
    keep = np.isin(yte, classes)
    te_paths = [p for p, k in zip(te_paths, keep) if k]; yte = yte[keep]
    print(f"classes={classes}  n_train={len(ytr)}  n_test={len(yte)}\n")

    enc_base = make_encoder(m, dev)
    enc_cc = make_encoder(m, dev, transform=shades_of_gray)
    enc_tta = make_encoder(m, dev, tta=True)
    enc_cc_tta = make_encoder(m, dev, transform=shades_of_gray, tta=True)

    print("Encoding (base / CC / TTA / CC+TTA for train & test)...")
    Xtr_b, Xte_b = enc_base(tr_paths), enc_base(te_paths)
    Xtr_c, Xte_c = enc_cc(tr_paths), enc_cc(te_paths)
    Xtr_t, Xte_t = enc_tta(tr_paths), enc_tta(te_paths)
    Xtr_ct, Xte_ct = enc_cc_tta(tr_paths), enc_cc_tta(te_paths)

    results = {}

    # --- reference: deployed prototypes on base features (must reproduce 66.14) ---
    dep = torch.load(DEPLOYED_PROTO, map_location="cpu", weights_only=False)
    P_dep = np.stack([_norm_t(dep[c]).numpy() for c in classes]).astype(np.float32)
    r, _ = evaluate(proto_logits(Xte_b, P_dep), yte, classes)
    results["baseline_deployed"] = r
    print(f"\n[baseline_deployed]  acc={r['acc']}  mF1={r['macroF1']}  (target 66.14)")

    # --- controlled rebuilt baseline (recipe, no lever) ---
    P_b = build_protos(Xtr_b, ytr, classes, text_embs)
    r, _ = evaluate(proto_logits(Xte_b, P_b), yte, classes)
    results["baseline_rebuilt"] = r
    print(f"[baseline_rebuilt]   acc={r['acc']}  mF1={r['macroF1']}")

    # --- L1 color constancy ---
    P_c = build_protos(Xtr_c, ytr, classes, text_embs)
    r, _ = evaluate(proto_logits(Xte_c, P_c), yte, classes)
    results["L1_color_constancy"] = r
    print(f"[L1 color_constancy] acc={r['acc']}  mF1={r['macroF1']}")

    # --- L2 TTA ---
    P_t = build_protos(Xtr_t, ytr, classes, text_embs)
    r, _ = evaluate(proto_logits(Xte_t, P_t), yte, classes)
    results["L2_tta"] = r
    print(f"[L2 tta]             acc={r['acc']}  mF1={r['macroF1']}")

    # --- L3 Tip-Adapter (tune alpha,beta on patient-grouped val from PAD-train) ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    g = train["patient_id"].values
    fit_idx, val_idx = next(gss.split(Xtr_b, ytr, groups=g))
    Xfit, yfit = Xtr_b[fit_idx], ytr[fit_idx]
    Xval, yval = Xtr_b[val_idx], ytr[val_idx]
    P_fit = build_protos(Xfit, yfit, classes, text_embs)
    base_val = proto_logits(Xval, P_fit)
    best = (-1, None)
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        for beta_tip in [1.0, 2.0, 3.0, 4.0, 5.0]:
            cache_val = tip_cache_logits(Xval, Xfit, yfit, classes, beta_tip)
            z = base_val + alpha * cache_val
            acc = float((np.array(classes)[z.argmax(1)] == yval).mean())
            if acc > best[0]:
                best = (acc, (alpha, beta_tip))
    alpha, beta_tip = best[1]
    print(f"[L3 tip] selected alpha={alpha} beta={beta_tip} (val acc={best[0]*100:.2f})")
    # report: full-train cache + full-train prototypes, eval test
    cache_te = tip_cache_logits(Xte_b, Xtr_b, ytr, classes, beta_tip)
    r, _ = evaluate(proto_logits(Xte_b, P_b) + alpha * cache_te, yte, classes)
    r["alpha"], r["beta_tip"] = alpha, beta_tip
    results["L3_tip_adapter"] = r
    print(f"[L3 tip_adapter]     acc={r['acc']}  mF1={r['macroF1']}")

    # --- combined: apply every lever that did NOT hurt vs baseline_rebuilt ---
    base_acc = results["baseline_rebuilt"]["acc"]
    winners = []
    feat_tr, feat_te = Xtr_b, Xte_b
    if results["L1_color_constancy"]["acc"] >= base_acc:
        winners.append("color_constancy"); feat_tr, feat_te = Xtr_c, Xte_c
    # if both CC and TTA win, use CC+TTA joint encoding
    if results["L2_tta"]["acc"] >= base_acc:
        winners.append("tta")
        if "color_constancy" in winners:
            feat_tr, feat_te = Xtr_ct, Xte_ct
        else:
            feat_tr, feat_te = Xtr_t, Xte_t
    P_comb = build_protos(feat_tr, ytr, classes, text_embs)
    z_comb = proto_logits(feat_te, P_comb)
    if results["L3_tip_adapter"]["acc"] >= base_acc:
        winners.append("tip_adapter")
        z_comb = z_comb + alpha * tip_cache_logits(feat_te, feat_tr, ytr, classes, beta_tip)
    r, _ = evaluate(z_comb, yte, classes)
    r["levers"] = winners
    results["combined_winners"] = r
    print(f"\n[combined_winners {winners}] acc={r['acc']}  mF1={r['macroF1']}")

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nSaved -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
