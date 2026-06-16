"""
GATE EXPERIMENT: do structured patient inputs (age, sex, anatomical site) improve
CLASSIFICATION — measured on MACRO-F1 (balanced), not just overall accuracy?

Rationale (advisor): HAM10000 is ~67% nevi. Any class prior regresses predictions
toward nevi and inflates *overall* accuracy without improving diagnosis. The honest
test is whether the gain survives on macro-F1 AND in a balanced 5-way episodic
protocol where the marginal prior cancels and only conditional age/site structure
can help.

Metadata is fused as a class-CONDITIONAL likelihood (Naive-Bayes late fusion,
NO marginal P(c) prior):
    log P(c | img, meta) = log P(c | img) + alpha * log P(meta | c)
    log P(meta | c) = log P(age_bin|c) + log P(sex|c) + log P(site|c)   [missing terms dropped]
alpha is tuned on VAL by macro-F1. Reported on TEST (overall + macro-F1 + per-class).

Two protocols:
  A) deployed 7-way  : query vs 20-shot M3 prototypes (matches the app; imbalanced)
  B) balanced 5-way episodic : per-episode classes balanced (marginal prior cancels)

Run: conda run -n raresight python src/training/_diag_meta_fusion.py
"""
import sys, os, json, glob, numpy as np, torch, pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, balanced_accuracy_score
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.data.preprocessing import HAM10000_LABEL_MAP

N_CLASSES = 7
SEED = 42
CKPT = "checkpoints/raresight_nblk4mix.pth"
DATA_ROOT = "data/ham10000"
CLASS_NAMES = {0:"Actinic ker",1:"BCC",2:"Benign ker",3:"Dermatofibroma",
               4:"Melanoma",5:"Nevi",6:"Vascular"}
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def load_split_meta(data_root, val_size=0.1, test_size=0.1, seed=42):
    """Reproduce load_ham10000's exact stratified split, returning aligned metadata."""
    meta = pd.read_csv(os.path.join(data_root, "HAM10000_metadata.csv"))
    id_to_path = {}
    for jpg in glob.glob(os.path.join(data_root, "**", "*.jpg"), recursive=True):
        id_to_path[os.path.splitext(os.path.basename(jpg))[0]] = jpg
    meta = meta[meta["image_id"].isin(id_to_path)].copy()
    meta["label"] = meta["dx"].map(HAM10000_LABEL_MAP).astype(int)
    meta["path"] = meta["image_id"].map(id_to_path)
    labels = meta["label"].values
    idx_all = np.arange(len(meta))
    sss_t = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trv, te = next(sss_t.split(idx_all, labels))
    val_frac = val_size / (1.0 - test_size)
    sss_v = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_rel, va_rel = next(sss_v.split(trv, labels[trv]))
    splits = {"train": meta.iloc[trv[tr_rel]], "val": meta.iloc[trv[va_rel]], "test": meta.iloc[te]}
    return {k: v.reset_index(drop=True) for k, v in splits.items()}


def age_bin(a):
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return None
    return int(min(max(a, 0), 89) // 10)   # decade 0..8


def fit_meta_likelihood(train_df, smooth=1.0):
    """Class-conditional log-likelihood tables P(feature|c) with Laplace smoothing."""
    sites = sorted(train_df["localization"].fillna("unknown").unique())
    sexes = ["male", "female", "unknown"]
    bins = list(range(9))
    tab = {"age": np.full((N_CLASSES, len(bins)), smooth),
           "sex": np.full((N_CLASSES, len(sexes)), smooth),
           "site": np.full((N_CLASSES, len(sites)), smooth)}
    si = {s: i for i, s in enumerate(sites)}
    xi = {s: i for i, s in enumerate(sexes)}
    for _, r in train_df.iterrows():
        c = int(r["label"])
        ab = age_bin(r["age"])
        if ab is not None:
            tab["age"][c, ab] += 1
        sx = r["sex"] if r["sex"] in xi else "unknown"
        tab["sex"][c, xi[sx]] += 1
        st = r["localization"] if r["localization"] in si else "unknown"
        if st not in si:
            st = "unknown"
        tab["site"][c, si.get(st, si.get("unknown", 0))] += 1
    logtab = {k: np.log(v / v.sum(axis=1, keepdims=True)) for k, v in tab.items()}
    return logtab, si, xi


def meta_loglik(df, logtab, si, xi):
    """(N, 7) matrix of log P(meta|c), dropping missing terms per-sample."""
    N = len(df)
    out = np.zeros((N, N_CLASSES))
    for i, (_, r) in enumerate(df.iterrows()):
        ab = age_bin(r["age"])
        if ab is not None:
            out[i] += logtab["age"][:, ab]
        sx = r["sex"] if r["sex"] in xi else "unknown"
        out[i] += logtab["sex"][:, xi[sx]]
        st = r["localization"]
        if st in si:
            out[i] += logtab["site"][:, si[st]]
        else:
            out[i] += logtab["site"][:, si.get("unknown", 0)]
    return out


def encode_images(model, dev, paths, bs=16):
    embs = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            batch = torch.cat([model.preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                               for p in paths[i:i+bs]]).to(dev)
            embs.append(_norm(model.backbone.encode_image(batch)).cpu())
    return torch.cat(embs)


def metrics(y_true, y_pred):
    return (100*np.mean(y_true == y_pred),
            100*f1_score(y_true, y_pred, average="macro", labels=list(range(N_CLASSES)), zero_division=0),
            100*balanced_accuracy_score(y_true, y_pred))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    temp = m.temperature.item()
    sp = load_split_meta(DATA_ROOT, seed=SEED)
    print({k: len(v) for k, v in sp.items()})

    logtab, si, xi = fit_meta_likelihood(sp["train"])
    P = torch.stack([_norm(torch.load("src/app/assets/disease_prototypes.pt",
                          map_location=dev)[c].to(dev)) for c in range(N_CLASSES)])

    # ── Protocol A: deployed 7-way ──
    print("\n=== PROTOCOL A: deployed 7-way (imbalanced) ===")
    res = {}
    for split in ["val", "test"]:
        df = sp[split]
        Q = encode_images(m, dev, df["path"].tolist()).to(dev)
        img_logp = torch.log_softmax(-torch.cdist(Q, P) * temp, dim=1).cpu().numpy()  # log P(c|img)
        mll = meta_loglik(df, logtab, si, xi)                                          # log P(meta|c)
        res[split] = {"y": df["label"].values, "img_logp": img_logp, "mll": mll}

    # tune alpha on VAL by macro-F1
    yv = res["val"]["y"]
    best_a, best_f1 = 0.0, -1
    for a in ALPHA_GRID:
        pred = (res["val"]["img_logp"] + a * res["val"]["mll"]).argmax(1)
        _, f1, _ = metrics(yv, pred)
        if f1 > best_f1:
            best_f1, best_a = f1, a
    yt = res["test"]["y"]
    base_pred = res["test"]["img_logp"].argmax(1)
    meta_pred = (res["test"]["img_logp"] + best_a * res["test"]["mll"]).argmax(1)
    b = metrics(yt, base_pred); mt = metrics(yt, meta_pred)
    print(f"  alpha*={best_a} (val macroF1={best_f1:.1f})")
    print(f"  {'':16}{'overall':>9}{'macroF1':>9}{'balAcc':>9}")
    print(f"  {'image-only':16}{b[0]:>8.1f}%{b[1]:>8.1f}%{b[2]:>8.1f}%")
    print(f"  {'image+meta':16}{mt[0]:>8.1f}%{mt[1]:>8.1f}%{mt[2]:>8.1f}%")
    print(f"  Δ overall={mt[0]-b[0]:+.1f}  Δ macroF1={mt[1]-b[1]:+.1f}  Δ balAcc={mt[2]-b[2]:+.1f}")
    protoA = {"alpha": best_a, "image_only": b, "image_meta": mt}

    # ── Protocol B: balanced 5-way episodic ──
    print("\n=== PROTOCOL B: balanced 5-way 5-shot episodic ===")
    def episodic(df_sup, df_qry, alpha, n_ep, seed):
        rng = np.random.default_rng(seed)
        by_cls = {c: df_qry.index[df_qry["label"] == c].to_numpy() for c in range(N_CLASSES)}
        sup_by = {c: df_sup.index[df_sup["label"] == c].to_numpy() for c in range(N_CLASSES)}
        # precompute embeddings lazily via cache
        yt_all, base_all, meta_all = [], [], []
        for _ in range(n_ep):
            cls = rng.choice(N_CLASSES, 5, replace=False)
            # support protos (5-shot image) from df_sup
            protos = []
            for c in cls:
                idx = rng.choice(sup_by[c], 5, replace=len(sup_by[c]) < 5)
                e = encode_cache(df_sup, idx)
                protos.append(_norm(e.mean(0)))
            protos = torch.stack(protos).to(dev)
            for c in cls:
                idx = rng.choice(by_cls[c], 15, replace=len(by_cls[c]) < 15)
                qe = encode_cache(df_qry, idx).to(dev)
                ilp = torch.log_softmax(-torch.cdist(qe, protos) * temp, 1).cpu().numpy()
                # meta loglik restricted to the 5 episode classes
                mll_full = meta_loglik(df_qry.loc[idx], logtab, si, xi)[:, cls]
                base_all.extend(cls[ilp.argmax(1)])
                meta_all.extend(cls[(ilp + alpha * mll_full).argmax(1)])
                yt_all.extend([c] * len(idx))
        return np.array(yt_all), np.array(base_all), np.array(meta_all)

    # embedding cache keyed by (split_id, idx)
    _cache = {}
    def encode_cache(df, idxs):
        out = []
        todo = [i for i in idxs if (id(df), i) not in _cache]
        if todo:
            embs = encode_images(m, dev, df.loc[todo, "path"].tolist())
            for j, i in enumerate(todo):
                _cache[(id(df), i)] = embs[j]
        for i in idxs:
            out.append(_cache[(id(df), i)])
        return torch.stack(out)

    # tune alpha on val episodes, eval on test episodes
    bestB_a, bestB_f1 = 0.0, -1
    for a in [0.0, 0.5, 1.0, 2.0]:
        y, _, mp = episodic(sp["train"], sp["val"], a, 80, seed=1)
        _, f1, _ = metrics(y, mp)
        if f1 > bestB_f1:
            bestB_f1, bestB_a = f1, a
    y, bp, mp = episodic(sp["train"], sp["test"], bestB_a, 200, seed=7)
    bB = metrics(y, bp); mB = metrics(y, mp)
    print(f"  alpha*={bestB_a} (val macroF1={bestB_f1:.1f})")
    print(f"  {'image-only':16}{bB[0]:>8.1f}%{bB[1]:>8.1f}%{bB[2]:>8.1f}%")
    print(f"  {'image+meta':16}{mB[0]:>8.1f}%{mB[1]:>8.1f}%{mB[2]:>8.1f}%")
    print(f"  Δ overall={mB[0]-bB[0]:+.1f}  Δ macroF1={mB[1]-bB[1]:+.1f}  Δ balAcc={mB[2]-bB[2]:+.1f}")

    verdict = ("METADATA HELPS (macroF1 up in both)" if (mt[1] > b[1] and mB[1] > bB[1])
               else "MIXED — gain not robust on macro-F1; frame as prior-calibration only")
    print(f"\n  VERDICT: {verdict}")
    json.dump({"protoA": {"alpha": protoA["alpha"], "image_only": list(b), "image_meta": list(mt)},
               "protoB": {"alpha": bestB_a, "image_only": list(bB), "image_meta": list(mB)},
               "verdict": verdict},
              open("checkpoints/diag_meta_fusion.json", "w"), indent=2)
    print("Saved → checkpoints/diag_meta_fusion.json")


if __name__ == "__main__":
    main()
