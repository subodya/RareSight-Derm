"""Best-effort FROZEN-FEATURE ceiling for the 7-way patient-upload path toward the
every-class>=60% target: class-balanced linear probe + metadata fusion + maximin bias.
If melanoma/benign-keratosis still miss 60% here, the frozen backbone is the bottleneck
and backbone fine-tuning is the only remaining lever.

Everything is built from load_split_meta (image<->metadata row-aligned). Probe trained on
train embeddings (class-balanced CE, selected on val macro-recall); metadata fused as
alpha*logP(meta|c) (alpha tuned on val); then per-class bias fit on val to maximise the
WORST class recall. Reported per-class on TEST (natural 7-way) for three regimes.

NOTE: metadata fusion only applies when the patient supplies age/sex/site; probe-only is
the no-metadata deployed number. Run: conda run -n raresight python src/training/_diag_probe_meta.py
"""
import sys, os, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.training._diag_meta_fusion import (load_split_meta, fit_meta_likelihood,
                                            meta_loglik, encode_images)

N = 7
NAMES = {0:"Actinic ker",1:"BasalCellCa",2:"BenignKerat",3:"Dermatofib",
         4:"Melanoma",5:"Melanoc.nevi",6:"Vascular"}
CKPT = "checkpoints/raresight_nblk4mix.pth"
CACHE = "checkpoints/_emb_meta_cache.npz"


def recalls(pred, y):
    return np.array([(pred[y == c] == c).mean() if (y == c).any() else np.nan for c in range(N)])


def macro(z, y, b): return np.nanmean(recalls((z + b).argmax(1), y))
def minrec(z, y, b): return np.nanmin(recalls((z + b).argmax(1), y))


def coord_ascent(z, y, objective, passes=4, grid=np.arange(-6, 6.01, 0.25)):
    b = np.zeros(N)
    for _ in range(passes):
        for c in range(N):
            t = b.copy(); s = []
            for g in grid:
                t[c] = g; s.append(objective(z, y, t))
            b[c] = grid[int(np.argmax(s))]
    return b


def report(tag, z, y, b=None):
    if b is None: b = np.zeros(N)
    r = recalls((z + b).argmax(1), y) * 100
    below = [NAMES[c] for c in range(N) if r[c] < 60]
    ov = ((z + b).argmax(1) == y).mean() * 100
    print(f"\n[{tag}]  overall={ov:5.2f}%  macro={np.nanmean(r):5.2f}%  min={np.nanmin(r):5.2f}%")
    print("   " + "  ".join(f"{NAMES[c]}={r[c]:.0f}" for c in range(N)))
    print(f"   <60%: {below if below else 'NONE — all >=60%'}")
    return {"overall": round(ov,2), "macro": round(float(np.nanmean(r)),2),
            "min": round(float(np.nanmin(r)),2),
            "per_class": {NAMES[c]: round(float(r[c]),1) for c in range(N)}, "below60": below}


def confusion(z, y, b, tag):
    pred = (z + b).argmax(1); M = np.zeros((N, N), int)
    for t, p in zip(y, pred): M[t, p] += 1
    print(f"\nConfusion ({tag}, row-normalised %):")
    print("            " + " ".join(f"{NAMES[c][:6]:>6}" for c in range(N)))
    for t in range(N):
        row = M[t] / max(M[t].sum(), 1) * 100
        print(f"  {NAMES[t]:<11} " + " ".join(f"{row[p]:6.0f}" for p in range(N)))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    sp = load_split_meta("data/ham10000", seed=42)
    print({k: len(v) for k, v in sp.items()})

    if os.path.exists(CACHE):
        d = np.load(CACHE)
        Xtr,Xv,Xt = d["Xtr"],d["Xv"],d["Xt"]
        print("loaded cached embeddings")
    else:
        print("encoding (train ~10.7k, ~10 min)...")
        Xtr = encode_images(m, dev, sp["train"]["path"].tolist()).numpy().astype(np.float32)
        Xv  = encode_images(m, dev, sp["val"]["path"].tolist()).numpy().astype(np.float32)
        Xt  = encode_images(m, dev, sp["test"]["path"].tolist()).numpy().astype(np.float32)
        np.savez(CACHE, Xtr=Xtr, Xv=Xv, Xt=Xt)
    ytr = sp["train"]["label"].values; yv = sp["val"]["label"].values; yt = sp["test"]["label"].values

    # ── class-balanced linear probe ──
    counts = np.bincount(ytr, minlength=N)
    w = torch.tensor(counts.sum()/(N*np.maximum(counts,1)), dtype=torch.float32)
    head = nn.Linear(512, N); opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xtr_t,ytr_t = torch.tensor(Xtr), torch.tensor(ytr).long()
    Xv_t,Xt_t = torch.tensor(Xv), torch.tensor(Xt)
    best, best_state = -1, None
    for step in range(1, 801):
        head.train(); opt.zero_grad(); lossf(head(Xtr_t), ytr_t).backward(); opt.step()
        if step % 50 == 0:
            head.eval()
            with torch.no_grad(): vmac = np.nanmean(recalls(head(Xv_t).argmax(1).numpy(), yv))
            if vmac > best: best, best_state = vmac, {k:v.clone() for k,v in head.state_dict().items()}
    head.load_state_dict(best_state); head.eval()
    with torch.no_grad():
        zv = torch.log_softmax(head(Xv_t),1).numpy()   # log P(c|img) val
        zt = torch.log_softmax(head(Xt_t),1).numpy()   # test
    print(f"\nprobe best val macro={best*100:.2f}%")

    # ── metadata log-likelihood ──
    logtab, si, xi = fit_meta_likelihood(sp["train"])
    mv = meta_loglik(sp["val"], logtab, si, xi)
    mt = meta_loglik(sp["test"], logtab, si, xi)

    results = {}
    results["probe_only"] = report("probe only (TEST)", zt, yt)

    # fuse metadata: tune alpha on val for maximin (push the floor)
    best_a, best_v = 0.0, -1
    for a in np.arange(0, 3.01, 0.25):
        v = minrec(zv + a*mv, yv, np.zeros(N))
        if v > best_v: best_v, best_a = v, a
    print(f"\nmeta alpha*={best_a:.2f} (val min-recall={best_v*100:.1f}%)")
    results["probe_meta"] = report(f"probe+meta a={best_a:.2f} (TEST)", zt + best_a*mt, yt)

    # add per-class maximin bias on top (fit on val)
    zfv, zft = zv + best_a*mv, zt + best_a*mt
    b_mm = coord_ascent(zfv, yv, minrec)
    print(f"   [val under probe+meta+bias: min={minrec(zfv,yv,b_mm)*100:.1f}% macro={macro(zfv,yv,b_mm)*100:.1f}%]")
    results["probe_meta_bias"] = report("probe+meta+maximin-bias (TEST)", zft, yt, b_mm)

    confusion(zft, yt, b_mm, "probe+meta+bias")
    results["_meta"] = {"meta_alpha": round(float(best_a),2),
                        "val_floor_probe_meta_bias": round(float(minrec(zfv,yv,b_mm)*100),2)}
    json.dump(results, open("checkpoints/diag_probe_meta.json","w"), indent=2)
    print("\nSaved → checkpoints/diag_probe_meta.json")
    r = results["probe_meta_bias"]
    print(f"\nVERDICT (frozen-feature best, probe+meta+bias): min-class={r['min']:.0f}% macro={r['macro']:.0f}% "
          f"— {'ALL >=60%, target MET on frozen features' if not r['below60'] else 'still <60%: '+str(r['below60'])+' → needs backbone fine-tuning'}")


if __name__ == "__main__":
    main()
