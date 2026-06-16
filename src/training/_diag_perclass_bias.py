"""Lever-A DIAGNOSTIC (no rebuild, no GPU training): can a per-class logit bias on the
CURRENT deployed prototypes lift every class toward >=60% recall, or is the patient-upload
path representation-limited?

We classify the full 7-way HAM10000 val/test splits against the deployed M3 prototypes
(disease_prototypes.pt), then fit per-class offsets b_c so prediction = argmax_c (z_c + b_c):

  baseline        : b = 0 (current deployed behaviour)
  logit-adjust    : b_c = -tau * log(prior_c), tau swept on VAL for macro-recall (principled)
  free / macroF1  : coordinate-ascent b_c maximising VAL macro-recall (upper bound of bias)
  free / maximin  : coordinate-ascent b_c maximising VAL min-class recall (the >=60% question)

All offsets are fit on VAL and reported on TEST. The confusion matrix (under the macro-recall
bias) shows whether an "all classes >=60%" solution even exists, or whether the weak classes
mutually confuse (representation limit -> needs Lever D, the linear probe).

Run: conda run -n raresight python src/training/_diag_perclass_bias.py
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.data.preprocessing import load_ham10000
from PIL import Image

N = 7
NAMES = {0:"Actinic ker",1:"BasalCellCa",2:"BenignKerat",3:"Dermatofib",
         4:"Melanoma",5:"Melanoc.nevi",6:"Vascular"}
CKPT = "checkpoints/raresight_nblk4mix.pth"


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def encode_split(m, dev, split):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    paths, labels = load_ham10000(data_root=os.path.join(root, "data", "ham10000"),
                                  split=split, val_size=0.1, test_size=0.1, seed=42)
    labels = np.array([int(l) for l in labels])
    embs = []
    with torch.no_grad():
        for i in range(0, len(paths), 16):
            batch = torch.cat([m.preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                               for p in paths[i:i+16]]).to(dev)
            embs.append(_norm(m.backbone.encode_image(batch)).cpu())
    return torch.cat(embs).numpy(), labels


def recalls(z, y, b):
    pred = (z + b).argmax(1)
    return np.array([(pred[y == c] == c).mean() if (y == c).any() else np.nan
                     for c in range(N)])


def macro(z, y, b):
    return np.nanmean(recalls(z, y, b))


def minrec(z, y, b):
    return np.nanmin(recalls(z, y, b))


def overall(z, y, b):
    return ((z + b).argmax(1) == y).mean()


def coord_ascent(z, y, objective, passes=4, grid=np.arange(-8, 8.01, 0.25)):
    """Maximise objective(z,y,b) over per-class offsets via coordinate ascent."""
    b = np.zeros(N)
    best = objective(z, y, b)
    for _ in range(passes):
        for c in range(N):
            trials = b.copy()
            scores = []
            for g in grid:
                trials[c] = g
                scores.append(objective(z, y, trials))
            jbest = int(np.argmax(scores))
            b[c] = grid[jbest]
            best = scores[jbest]
    return b, best


def report(tag, z, y, b):
    r = recalls(z, y, b) * 100
    below = [NAMES[c] for c in range(N) if r[c] < 60]
    print(f"\n[{tag}]  overall={overall(z,y,b)*100:5.2f}%  macro_recall={np.nanmean(r):5.2f}%  "
          f"min_class={np.nanmin(r):5.2f}%")
    print("   " + "  ".join(f"{NAMES[c]}={r[c]:.0f}" for c in range(N)))
    print(f"   classes <60%: {below if below else 'NONE — all >=60%'}")
    return {"overall": round(overall(z,y,b)*100,2), "macro_recall": round(float(np.nanmean(r)),2),
            "min_class": round(float(np.nanmin(r)),2),
            "per_class": {NAMES[c]: round(float(r[c]),1) for c in range(N)},
            "below60": below}


def confusion(z, y, b):
    pred = (z + b).argmax(1)
    M = np.zeros((N, N), int)
    for t, p in zip(y, pred):
        M[t, p] += 1
    print("\nConfusion matrix under macro-recall bias (rows=true, cols=pred, row-normalised %):")
    print("            " + " ".join(f"{NAMES[c][:6]:>6}" for c in range(N)))
    for t in range(N):
        row = M[t] / max(M[t].sum(), 1) * 100
        print(f"  {NAMES[t]:<11} " + " ".join(f"{row[p]:6.0f}" for p in range(N)))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    temp = m.temperature.item()
    protos = torch.load("src/app/assets/disease_prototypes.pt", map_location=dev)
    P = torch.stack([_norm(protos[c].to(dev)) for c in range(N)]).cpu().numpy()  # (7,512)
    print(f"Deployed prototypes loaded. temp={temp:.3f}")

    Qv, yv = encode_split(m, dev, "val");  print(f"val:  {len(Qv)} imgs")
    Qt, yt = encode_split(m, dev, "test"); print(f"test: {len(Qt)} imgs")
    zv = -np.linalg.norm(Qv[:, None, :] - P[None], axis=2) * temp   # (Nv,7) logits
    zt = -np.linalg.norm(Qt[:, None, :] - P[None], axis=2) * temp

    results = {}
    results["baseline"] = report("baseline b=0 (TEST)", zt, yt, np.zeros(N))

    # principled logit adjustment: b_c = -tau*log(prior_c), tau swept on VAL macro-recall
    _, counts = np.unique(yv, return_counts=True)
    prior = counts / counts.sum()
    logp = np.log(prior + 1e-9)
    best_tau, best_v = 0.0, -1
    for tau in np.arange(0, 6.01, 0.1):
        v = macro(zv, yv, -tau * logp)
        if v > best_v: best_v, best_tau = v, tau
    b_la = -best_tau * logp
    print(f"\nlogit-adjust: best tau={best_tau:.1f} (val macro={best_v*100:.2f}%)")
    results["logit_adjust"] = report(f"logit-adjust tau={best_tau:.1f} (TEST)", zt, yt, b_la)

    # free per-class bias, two objectives, fit on VAL
    b_macro, _ = coord_ascent(zv, yv, macro)
    results["free_macro"] = report("free bias / val-macro (TEST)", zt, yt, b_macro)
    print(f"   [val under this bias: macro={macro(zv,yv,b_macro)*100:.2f}% min={minrec(zv,yv,b_macro)*100:.2f}%]")

    b_mm, _ = coord_ascent(zv, yv, minrec)
    results["free_maximin"] = report("free bias / val-maximin (TEST)", zt, yt, b_mm)
    print(f"   [val under this bias: macro={macro(zv,yv,b_mm)*100:.2f}% min={minrec(zv,yv,b_mm)*100:.2f}%]")

    confusion(zt, yt, b_macro)

    results["_meta"] = {"temp": round(temp,3), "best_tau": round(best_tau,2),
                        "val_n": len(Qv), "test_n": len(Qt),
                        "val_maximin_floor": round(minrec(zv,yv,b_mm)*100,2)}
    json.dump(results, open("checkpoints/diag_perclass_bias.json","w"), indent=2)
    print("\nSaved → checkpoints/diag_perclass_bias.json")
    print(f"\nVERDICT: best per-class bias floors min-class recall at "
          f"{results['free_maximin']['min_class']:.0f}% (test) / "
          f"{minrec(zv,yv,b_mm)*100:.0f}% (val). "
          f"{'A+B+C plausible' if results['free_maximin']['min_class']>=58 else 'training-free bias INSUFFICIENT → need Lever D'} for the >=60% floor.")


if __name__ == "__main__":
    main()
