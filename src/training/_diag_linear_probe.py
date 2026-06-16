"""Lever-D DIAGNOSTIC: how high can a CLASS-BALANCED LINEAR PROBE on frozen BiomedCLIP
features push per-class recall on the 7-way patient-upload path? The Lever-A diagnostic
proved a per-class bias can't lift the worst class above ~41% (representation limit). D
adds separability by training a linear head (no backbone training) — this quantifies the
ceiling so we can decide whether to wire it into the deployed no-user-support branch.

Train nn.Linear(512->7) on frozen HAM10000 TRAIN embeddings with class-balanced CE
(weight_c = N/(K*count_c)); select on VAL macro-recall; report per-class TEST recall +
confusion vs the deployed-prototype baseline. Train embeddings cached for reuse.

Run: conda run -n raresight python src/training/_diag_linear_probe.py
"""
import sys, os, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.data.preprocessing import load_ham10000
from PIL import Image

N = 7
NAMES = {0:"Actinic ker",1:"BasalCellCa",2:"BenignKerat",3:"Dermatofib",
         4:"Melanoma",5:"Melanoc.nevi",6:"Vascular"}
CKPT = "checkpoints/raresight_nblk4mix.pth"
CACHE = "checkpoints/_emb_cache.npz"


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
    return torch.cat(embs).numpy().astype(np.float32), labels


def recalls(pred, y):
    return np.array([(pred[y == c] == c).mean() if (y == c).any() else np.nan for c in range(N)])


def summary(tag, pred, y):
    r = recalls(pred, y) * 100
    below = [NAMES[c] for c in range(N) if r[c] < 60]
    ov = (pred == y).mean() * 100
    print(f"\n[{tag}]  overall={ov:5.2f}%  macro_recall={np.nanmean(r):5.2f}%  min_class={np.nanmin(r):5.2f}%")
    print("   " + "  ".join(f"{NAMES[c]}={r[c]:.0f}" for c in range(N)))
    print(f"   classes <60%: {below if below else 'NONE — all >=60%'}")
    return {"overall": round(ov,2), "macro_recall": round(float(np.nanmean(r)),2),
            "min_class": round(float(np.nanmin(r)),2),
            "per_class": {NAMES[c]: round(float(r[c]),1) for c in range(N)}, "below60": below}


def confusion(pred, y, tag):
    M = np.zeros((N, N), int)
    for t, p in zip(y, pred): M[t, p] += 1
    print(f"\nConfusion ({tag}, rows=true, row-normalised %):")
    print("            " + " ".join(f"{NAMES[c][:6]:>6}" for c in range(N)))
    for t in range(N):
        row = M[t] / max(M[t].sum(), 1) * 100
        print(f"  {NAMES[t]:<11} " + " ".join(f"{row[p]:6.0f}" for p in range(N)))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    temp = m.temperature.item()

    if os.path.exists(CACHE):
        d = np.load(CACHE)
        Xtr,ytr,Xv,yv,Xt,yt = d["Xtr"],d["ytr"],d["Xv"],d["yv"],d["Xt"],d["yt"]
        print(f"Loaded cached embeddings: train {len(Xtr)}, val {len(Xv)}, test {len(Xt)}")
    else:
        print("Encoding splits (train is ~10.7k imgs, ~10 min)...")
        Xtr,ytr = encode_split(m, dev, "train")
        Xv,yv   = encode_split(m, dev, "val")
        Xt,yt   = encode_split(m, dev, "test")
        np.savez(CACHE, Xtr=Xtr,ytr=ytr,Xv=Xv,yv=yv,Xt=Xt,yt=yt)
        print(f"Encoded + cached: train {len(Xtr)}, val {len(Xv)}, test {len(Xt)}")

    # ── baseline: deployed prototypes (for side-by-side) ──
    protos = torch.load("src/app/assets/disease_prototypes.pt", map_location="cpu")
    P = torch.stack([_norm(protos[c]) for c in range(N)]).numpy()
    base_pred = (-np.linalg.norm(Xt[:,None,:]-P[None],axis=2)).argmax(1)
    results = {"baseline_deployed_protos": summary("baseline deployed protos (TEST)", base_pred, yt)}

    # ── class-balanced linear probe ──
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr).long()
    counts = np.bincount(ytr, minlength=N)
    w = torch.tensor(counts.sum()/(N*np.maximum(counts,1)), dtype=torch.float32)
    head = nn.Linear(512, N)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xv_t = torch.tensor(Xv); Xt_t = torch.tensor(Xt)
    best_macro, best_state = -1, None
    for step in range(1, 801):
        head.train(); opt.zero_grad()
        loss = lossf(head(Xtr_t), ytr_t); loss.backward(); opt.step()
        if step % 50 == 0:
            head.eval()
            with torch.no_grad():
                vpred = head(Xv_t).argmax(1).numpy()
            vmac = np.nanmean(recalls(vpred, yv))
            if vmac > best_macro:
                best_macro = vmac; best_state = {k:v.clone() for k,v in head.state_dict().items()}
    head.load_state_dict(best_state); head.eval()
    print(f"\nlinear probe: best val macro-recall={best_macro*100:.2f}%")
    with torch.no_grad():
        lp_pred = head(Xt_t).argmax(1).numpy()
    results["linear_probe_balanced"] = summary("class-balanced linear probe (TEST)", lp_pred, yt)

    confusion(base_pred, yt, "deployed protos")
    confusion(lp_pred, yt, "linear probe")

    results["_meta"] = {"temp": round(temp,3), "train_n": int(len(Xtr)),
                        "class_counts_train": counts.tolist()}
    json.dump(results, open("checkpoints/diag_linear_probe.json","w"), indent=2)
    print("\nSaved → checkpoints/diag_linear_probe.json")
    lp = results["linear_probe_balanced"]
    print(f"\nVERDICT: linear probe min-class={lp['min_class']:.0f}% macro={lp['macro_recall']:.0f}% "
          f"({'ALL >=60% — meets goal' if not lp['below60'] else 'still <60%: '+str(lp['below60'])})")


if __name__ == "__main__":
    main()
