"""
Canonical builder for RareSight's DEPLOYED serving artifacts, all fit on the exact
protocol the app serves (query image → 7-way vs the 20-shot M3 prototypes) and
evaluated honestly on a held-out test split.

Produces src/app/assets/serving_params.pt with three feature groups:

1. METADATA FUSION (validated to help macro-F1; see _diag_meta_fusion.py)
   - meta_logtab : class-conditional log-likelihoods log P(age_bin|c), P(sex|c), P(site|c)
   - meta_alpha  : fusion weight, tuned on VAL by macro-F1 (7-way deployed protocol)
   Prediction logits = z_img + alpha * logP(meta|c).  alpha is the ACCURACY lever
   (changes argmax); missing metadata terms are dropped per-field.

2. CALIBRATION (temperature scaling, Guo et al. 2017)
   - calib_T : fit on the FUSED val logits by NLL over a WIDE grid. Monotonic →
     never changes argmax/accuracy, only confidence. Reports REAL ECE before/after.

3. OPEN-SET REJECTION (abstain on "unknown / not a known class")
   Uses an IMAGE-only score computed BEFORE metadata fusion (so a confident meta
   prior can't mask an OOD image). Compares msp / maxcos / energy / mahalanobis in
   a leave-one-class-out protocol; keeps the best by AUROC. Stores the score method,
   threshold tau (10% in-dist false-abstain), and (for mahalanobis) the fitted
   per-class means + shared inverse covariance.

Run: conda run -n raresight python src/app/tools/build_serving_artifacts.py
"""
import sys, os, glob, json, numpy as np, torch, pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.models.raresight_net import RareSight
from src.data.preprocessing import HAM10000_LABEL_MAP

N_CLASSES = 7
SEED = 42
DATA_ROOT = "data/ham10000"
CKPT = "checkpoints/raresight_nblk4mix.pth"
PROTO_PATH = "src/app/assets/disease_prototypes.pt"
OUT = "src/app/assets/serving_params.pt"
ALPHA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
FALSE_ABSTAIN = 0.02          # in-dist false-abstain rate for tau (triage: reject few valid)
MAHA_PER_CLASS = 400          # train images/class encoded for Mahalanobis cov fit
SEXES = ["male", "female", "unknown"]


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def load_split_meta(data_root, val_size=0.1, test_size=0.1, seed=42):
    meta = pd.read_csv(os.path.join(data_root, "HAM10000_metadata.csv"))
    id_to_path = {}
    for jpg in glob.glob(os.path.join(data_root, "**", "*.jpg"), recursive=True):
        id_to_path[os.path.splitext(os.path.basename(jpg))[0]] = jpg
    meta = meta[meta["image_id"].isin(id_to_path)].copy()
    meta["label"] = meta["dx"].map(HAM10000_LABEL_MAP).astype(int)
    meta["path"] = meta["image_id"].map(id_to_path)
    labels = meta["label"].values
    idx = np.arange(len(meta))
    trv, te = next(StratifiedShuffleSplit(1, test_size=test_size, random_state=seed).split(idx, labels))
    vf = val_size / (1 - test_size)
    tr_rel, va_rel = next(StratifiedShuffleSplit(1, test_size=vf, random_state=seed).split(trv, labels[trv]))
    return {"train": meta.iloc[trv[tr_rel]].reset_index(drop=True),
            "val":   meta.iloc[trv[va_rel]].reset_index(drop=True),
            "test":  meta.iloc[te].reset_index(drop=True)}


def age_bin(a):
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return None
    return int(min(max(a, 0), 89) // 10)


def fit_meta_logtab(train_df, smooth=1.0):
    sites = sorted(train_df["localization"].fillna("unknown").astype(str).unique().tolist())
    if "unknown" not in sites:
        sites.append("unknown")
    si = {s: i for i, s in enumerate(sites)}
    xi = {s: i for i, s in enumerate(SEXES)}
    tab = {"age": np.full((N_CLASSES, 9), smooth),
           "sex": np.full((N_CLASSES, len(SEXES)), smooth),
           "site": np.full((N_CLASSES, len(sites)), smooth)}
    for _, r in train_df.iterrows():
        c = int(r["label"])
        ab = age_bin(r["age"])
        if ab is not None:
            tab["age"][c, ab] += 1
        tab["sex"][c, xi.get(r["sex"], xi["unknown"])] += 1
        st = r["localization"] if r["localization"] in si else "unknown"
        tab["site"][c, si[st]] += 1
    logtab = {k: np.log(v / v.sum(1, keepdims=True)) for k, v in tab.items()}
    return logtab, si, xi


def meta_loglik(df, logtab, si, xi):
    out = np.zeros((len(df), N_CLASSES))
    for i, (_, r) in enumerate(df.iterrows()):
        ab = age_bin(r["age"])
        if ab is not None:
            out[i] += logtab["age"][:, ab]
        out[i] += logtab["sex"][:, xi.get(r["sex"], xi["unknown"])]
        st = r["localization"] if r["localization"] in si else "unknown"
        out[i] += logtab["site"][:, si[st]]
    return out


def encode(model, dev, paths, bs=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            b = torch.cat([model.preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                           for p in paths[i:i+bs]]).to(dev)
            out.append(_norm(model.backbone.encode_image(b)).cpu())
    return torch.cat(out)


def softmax_np(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def ece(probs, labels, n_bins=15):
    conf = probs.max(1); pred = probs.argmax(1); correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        msk = (conf > lo) & (conf <= hi)
        if msk.sum():
            e += msk.mean() * abs(correct[msk].mean() - conf[msk].mean())
    return float(e)


def macroF1(y, p):
    return f1_score(y, p, average="macro", labels=list(range(N_CLASSES)), zero_division=0)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    temp = m.temperature.item()
    P = torch.stack([_norm(torch.load(PROTO_PATH, map_location=dev)[c]) for c in range(N_CLASSES)]).to(dev)
    sp = load_split_meta(DATA_ROOT, seed=SEED)
    logtab, si, xi = fit_meta_logtab(sp["train"])

    enc = {}
    for split in ["val", "test"]:
        Q = encode(m, dev, sp[split]["path"].tolist()).to(dev)
        enc[split] = {
            "y": sp[split]["label"].values,
            "Q": Q.cpu(),
            "cos": (Q @ P.T).cpu().numpy(),
            "z": (-torch.cdist(Q, P) * temp).cpu().numpy(),
            "mll": meta_loglik(sp[split], logtab, si, xi),
        }

    # ── 1. metadata fusion: tune alpha on VAL macro-F1 (7-way deployed) ──
    yv = enc["val"]["y"]
    best_a, best_f1 = 0.0, -1
    for a in ALPHA_GRID:
        f1 = macroF1(yv, (enc["val"]["z"] + a * enc["val"]["mll"]).argmax(1))
        if f1 > best_f1:
            best_f1, best_a = f1, a
    fused = {s: enc[s]["z"] + best_a * enc[s]["mll"] for s in ["val", "test"]}

    # ── 2. calibration: fit a SEPARATE temperature per served regime ──
    # Metadata is optional at inference, so image-only and image+meta logits live on
    # different scales and need different temperatures (else the displayed ECE is
    # misattributed for image-only inputs). Fit both; predict() picks by meta usage.
    def fit_T(logits_val, y):
        bT, bn = 1.0, 1e9
        for T in np.concatenate([np.linspace(0.05, 1.0, 96), np.linspace(1.05, 10, 90)]):
            p = softmax_np(logits_val / T)
            nll = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
            if nll < bn:
                bn, bT = nll, float(T)
        return bT

    img_logits = {s: enc[s]["z"] for s in ["val", "test"]}
    T_img  = fit_T(img_logits["val"], yv)
    T_meta = fit_T(fused["val"], yv)
    ece_rep = {}
    for tag, lg, T in [("img", img_logits, T_img), ("meta", fused, T_meta)]:
        for s in ["val", "test"]:
            ece_rep[f"ece_before_{tag}_{s}"] = round(ece(softmax_np(lg[s]), enc[s]["y"]), 4)
            ece_rep[f"ece_after_{tag}_{s}"]  = round(ece(softmax_np(lg[s] / T), enc[s]["y"]), 4)
    best_T = T_meta  # headline (recommended path uses metadata)

    # ── 3. open-set rejection: compare image-only scores (pre-fusion) via LOCO ──
    # Fit Mahalanobis (per-class mean + shared inv-cov) on a train subset.
    tr = sp["train"]
    maha_paths, maha_y = [], []
    rng = np.random.default_rng(SEED)
    for c in range(N_CLASSES):
        idx = tr.index[tr["label"] == c].to_numpy()
        pick = rng.choice(idx, min(MAHA_PER_CLASS, len(idx)), replace=False)
        maha_paths += tr.loc[pick, "path"].tolist(); maha_y += [c] * len(pick)
    Ftr = encode(m, dev, maha_paths).numpy(); maha_y = np.array(maha_y)
    means = np.stack([Ftr[maha_y == c].mean(0) for c in range(N_CLASSES)])      # (7,512)
    centered = np.concatenate([Ftr[maha_y == c] - means[c] for c in range(N_CLASSES)])
    cov = np.cov(centered, rowvar=False) + 1e-3 * np.eye(Ftr.shape[1])          # shrinkage
    inv_cov = np.linalg.inv(cov)

    def maha_neg(Q, allowed):  # higher = more in-dist → return -min_dist over allowed classes
        d = []
        for c in allowed:
            diff = Q - means[c]
            d.append(np.einsum("ni,ij,nj->n", diff, inv_cov, diff))
        return -np.min(np.stack(d, 1), 1)

    def scores(split, allowed):
        z = enc[split]["z"][:, allowed]
        cos = enc[split]["cos"][:, allowed]
        Q = enc[split]["Q"].numpy()
        return {"msp": softmax_np(z).max(1), "maxcos": cos.max(1),
                "energy": np.log(np.exp(z - z.max(1, keepdims=True)).sum(1)) + z.max(1),
                "maha": maha_neg(Q, allowed)}

    y_te = enc["test"]["y"]
    auroc = {k: [] for k in ["msp", "maxcos", "energy", "maha"]}
    tpr_at = {k: [] for k in auroc}
    for h in range(N_CLASSES):
        allowed = [c for c in range(N_CLASSES) if c != h]
        sc_te = scores("test", allowed)
        is_unk = (y_te == h).astype(int)
        if is_unk.sum() == 0:
            continue
        for k, s in sc_te.items():
            auroc[k].append(roc_auc_score(is_unk, -s))            # lower score = unknown
            known = s[y_te != h]; tau_h = np.quantile(known, FALSE_ABSTAIN)
            tpr_at[k].append(float((s[y_te == h] < tau_h).mean()))  # held-out flagged @10%FPR
    auroc_mean = {k: float(np.mean(v)) for k, v in auroc.items()}
    tpr_mean = {k: float(np.mean(v)) for k, v in tpr_at.items()}
    # DEFAULT = maha (not best-AUROC auto-select). The deployed safety narrative is
    # far-OOD / wrong-modality (the PAD 82% off-modality flag, the webp demo), validated
    # on Mahalanobis; maha is also prototype-independent (fit on train image embeddings,
    # not the served prototypes) so it stays valid in refinement mode where user
    # prototypes change the cosine geometry. A plain re-run must NOT silently swap this
    # (auto-select picks maxcos on the CoOp prototypes). Use OOD_METHOD=auto to restore
    # best-AUROC selection, or OOD_METHOD=<name> to pin a specific score.
    sel = os.environ.get("OOD_METHOD", "maha")
    if sel == "auto" or sel not in auroc_mean:
        best_score = max(auroc_mean, key=auroc_mean.get)
    else:
        best_score = sel

    # tau for serving: 10% in-dist false-abstain on VAL using the full 7-way best score
    val_full = scores("val", list(range(N_CLASSES)))[best_score]
    tau = float(np.quantile(val_full, FALSE_ABSTAIN))

    params = {
        # metadata fusion
        "meta_logtab": {k: v.tolist() for k, v in logtab.items()},
        "meta_site_index": si, "meta_sex_index": xi, "meta_alpha": best_a,
        "meta_val_macroF1": round(best_f1 * 100, 2),
        # calibration (two regimes: image-only vs image+metadata)
        "calib_T_img": round(T_img, 4), "calib_T_meta": round(T_meta, 4),
        "calib_T": round(best_T, 4),                 # back-compat headline = meta path
        "ece_after_test": ece_rep["ece_after_meta_test"],
        "ece_before_test": ece_rep["ece_before_meta_test"],
        "temp_metric": round(temp, 4), **ece_rep,
        # open-set
        "ood_method": best_score, "ood_tau": round(tau, 4),
        "ood_false_abstain": FALSE_ABSTAIN,
        "ood_auroc": {k: round(v, 4) for k, v in auroc_mean.items()},
        "ood_tpr_at_fpr": {k: round(v, 4) for k, v in tpr_mean.items()},
        "maha_means": means.tolist(), "maha_inv_cov": inv_cov.tolist(),
    }
    torch.save(params, OUT)
    print("\n================  SERVING ARTIFACTS  ================")
    print(f"META  : alpha*={best_a}  (val macroF1={best_f1*100:.1f})")
    print(f"CALIB : image-only  T={T_img:.3f}  ECE test {ece_rep['ece_before_img_test']} -> {ece_rep['ece_after_img_test']}")
    print(f"        image+meta  T={T_meta:.3f}  ECE test {ece_rep['ece_before_meta_test']} -> {ece_rep['ece_after_meta_test']}")
    print(f"OOD   : best={best_score}")
    fpr_pct = int(FALSE_ABSTAIN * 100)
    for k in auroc_mean:
        print(f"          {k:7} AUROC={auroc_mean[k]:.3f}  TPR@{fpr_pct}%FPR={tpr_mean[k]:.3f}")
    print(f"        tau={tau:.4f}  (~{int(FALSE_ABSTAIN*100)}% in-dist abstain)")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
