"""
Reliability diagram + Brier score for the DEPLOYED 7-way path (RO8).

The proposal (RO8) asked for calibration diagrams (ECE, Brier). The deployed serving
artifacts already report ECE after temperature scaling (~0.048); this script reproduces
that exact 7-way image+metadata protocol, adds the multiclass **Brier score** and a
**reliability diagram** (before vs after temperature scaling), and cross-checks the saved
ECE so the figure is provably the deployed number — NOT the higher episodic ECE (~0.316).

Reuses the canonical helpers from build_serving_artifacts.py (same encode / split / ECE /
metadata fusion) so nothing diverges from serving. Read-only: no artifact is rewritten.

Run:  conda run -n raresight python src/training/eval_calibration_curve.py
Out:  figures/reliability_diagram.png , checkpoints/calibration_curve.json
"""

import sys, os, json, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.app.tools.build_serving_artifacts import (
    load_split_meta, meta_loglik, encode, softmax_np, ece, N_CLASSES, DATA_ROOT, SEED,
)

CKPT = "checkpoints/raresight_nblk4mix.pth"
PROTO_PATH = "src/app/assets/disease_prototypes.pt"
PARAMS_PATH = "src/app/assets/serving_params.pt"
FIG = "figures/reliability_diagram.png"
OUT = "checkpoints/calibration_curve.json"
N_BINS = 15


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def brier_multiclass(probs, labels):
    onehot = np.zeros_like(probs); onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean())


def reliability(probs, labels, n_bins=N_BINS):
    conf = probs.max(1); pred = probs.argmax(1); correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    mids, accs, confs, counts = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        msk = (conf > lo) & (conf <= hi)
        mids.append((lo + hi) / 2)
        if msk.sum():
            accs.append(correct[msk].mean()); confs.append(conf[msk].mean()); counts.append(int(msk.sum()))
        else:
            accs.append(np.nan); confs.append(np.nan); counts.append(0)
    return np.array(mids), np.array(accs), np.array(confs), np.array(counts)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    temp = m.temperature.item()
    params = torch.load(PARAMS_PATH, map_location="cpu", weights_only=False)  # trusted local artifact (has numpy)
    T_meta = float(params["calib_T_meta"]); T_img = float(params["calib_T_img"])
    alpha = float(params["meta_alpha"])
    logtab = {k: np.array(v) for k, v in params["meta_logtab"].items()}
    si, xi = params["meta_site_index"], params["meta_sex_index"]

    P = torch.stack([_norm(torch.load(PROTO_PATH, map_location=dev)[c]) for c in range(N_CLASSES)]).to(dev)
    sp = load_split_meta(DATA_ROOT, seed=SEED)

    Q = encode(m, dev, sp["test"]["path"].tolist()).to(dev)
    y = sp["test"]["label"].values
    z = (-torch.cdist(Q, P) * temp).cpu().numpy()                 # image-only logits
    mll = meta_loglik(sp["test"], logtab, si, xi)
    fused = z + alpha * mll                                       # deployed image+meta logits

    regimes = {
        "image_meta": {"logits": fused, "T": T_meta, "label": "7-way deployed (image+meta)"},
        "image_only": {"logits": z,     "T": T_img,  "label": "7-way (image-only)"},
    }
    out = {"_config": {"protocol": "7-way deployed, HAM10000 test", "n_test": int(len(y)),
                       "temp_metric": round(temp, 3), "alpha_meta": alpha,
                       "T_meta": round(T_meta, 4), "T_img": round(T_img, 4), "n_bins": N_BINS}}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (key, r) in zip(axes, regimes.items()):
        p_before = softmax_np(r["logits"]); p_after = softmax_np(r["logits"] / r["T"])
        ece_b, ece_a = ece(p_before, y), ece(p_after, y)
        brier_b, brier_a = brier_multiclass(p_before, y), brier_multiclass(p_after, y)
        out[key] = {"ece_before": round(ece_b, 4), "ece_after": round(ece_a, 4),
                    "brier_before": round(brier_b, 4), "brier_after": round(brier_a, 4)}
        print(f"{r['label']:<32} ECE {ece_b:.4f}→{ece_a:.4f}   Brier {brier_b:.4f}→{brier_a:.4f}")

        mids, accs, _, counts = reliability(p_after, y)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect")
        valid = counts > 0
        ax.bar(mids[valid], accs[valid], width=1.0/N_BINS*0.9, alpha=0.75,
               edgecolor="black", label="accuracy", color="#4C72B0")
        ax.bar(mids[valid], mids[valid] - accs[valid], width=1.0/N_BINS*0.9,
               bottom=accs[valid], alpha=0.35, color="#C44E52", label="gap")
        ax.set_title(f"{r['label']}\nECE {ece_a:.3f} · Brier {brier_a:.3f} (after T={r['T']:.2f})")
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    plt.savefig(FIG, dpi=150)
    print(f"Saved figure → {FIG}")

    # cross-check against the saved serving numbers
    saved = params.get("ece_after_meta_test")
    out["_crosscheck"] = {"serving_ece_after_meta_test": saved,
                          "recomputed_ece_after_image_meta": out["image_meta"]["ece_after"],
                          "match": abs((saved or 0) - out["image_meta"]["ece_after"]) < 0.01}
    print(f"Cross-check: serving ece_after_meta_test={saved}  recomputed={out['image_meta']['ece_after']}")
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
