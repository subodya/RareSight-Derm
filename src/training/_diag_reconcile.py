"""
DIAGNOSTIC: reconcile evaluate.py's 63.56% against the ablation/Tier-1 ~54.66%.

Same checkpoint, same forward() path, same descriptions — the only suspected
difference is episode sampling (evaluate.py is UNSEEDED; ablation/Tier-1 set
np.random.seed(42)). This script runs the IDENTICAL three prototype methods on
multiple seeded episode sets, in one process / one model load, so every number
is directly comparable:

  fusion     : model.forward() path  (encode_multimodal support, image-only query)  [= evaluate.py / ablation 'multimodal']
  image_only : mean raw image support embeddings                                     [= Tier-1 M0]
  m3_blend   : norm(beta*img_proto + (1-beta)*norm(cupl_text + lam*gap))             [= Tier-1 M3, deployed]

It reports, per seed, overall + per-class accuracy for all three, plus mean+/-std
across seeds. M0 image_only @ seed 42 / 300 ep MUST be ~57.37 and fusion ~54.66
(sanity that we reproduce the prior runs).

Run:
  conda run -n raresight python src/training/_diag_reconcile.py
Env:
  DIAG_SEEDS   comma list of seeds   (default "42,0,1,7")
  DIAG_EP      episodes per seed     (default 300)
  DIAG_SPLIT   split                 (default test)
"""

import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.data.dataset import EpisodicDermaMNIST

N_WAY, K_SHOT, N_QUERY, N_CLASSES = 5, 5, 15, 7
SEEDS = [int(s) for s in os.environ.get("DIAG_SEEDS", "42,0,1,7").split(",")]
EP    = int(os.environ.get("DIAG_EP", "300"))
SPLIT = os.environ.get("DIAG_SPLIT", "test")
CKPT  = "checkpoints/raresight_nblk4mix.pth"
OUT   = "checkpoints/diag_reconcile.json"
DESC_ORIG = os.path.join(os.path.dirname(__file__), "../../src/app/class_descriptions.json")
DESC_CUPL = os.path.join(os.path.dirname(__file__), "../../src/app/cupl_descriptions.json")
CLASS_NAMES = {0:"Actinic keratoses",1:"Basal cell carcinoma",2:"Benign keratosis",
               3:"Dermatofibroma",4:"Melanoma",5:"Melanocytic nevi",6:"Vascular lesions"}
RARE = {3, 6}
METHODS = ["fusion", "image_only", "m3_blend"]


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}  | seeds={SEEDS}  ep={EP}  split={SPLIT}\n")
    orig = json.load(open(os.path.abspath(DESC_ORIG)))
    cupl = json.load(open(os.path.abspath(DESC_CUPL)))

    m = RareSight(device=dev)
    m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False)
    m.eval()
    temp = m.temperature.item()
    print(f"Loaded. alpha={m.alpha.item():.4f} temp={temp:.4f}")

    # Load M3 deployed recipe (gap, beta, lam, per-class CuPL text embeddings)
    blend = torch.load("src/app/assets/blend_params.pt", map_location=dev)
    beta, lam = float(blend["beta"]), float(blend["lam"])
    gap = blend["gap"].to(dev)
    # per-class CuPL text embeddings, gap-shifted & normed (the M3 text term)
    txt_cupl = torch.stack([blend["text_embs"][c].to(dev) for c in range(N_CLASSES)])
    txt_m3 = _norm(txt_cupl + lam * gap)                    # (7,512)
    print(f"M3 recipe: beta={beta} lam={lam} |gap|={gap.norm():.3f}\n")

    def enc_text_orig(texts):
        toks = m.tokenizer(texts, padding="max_length", truncation=True,
                           max_length=256, return_tensors="pt")["input_ids"].to(dev)
        with torch.no_grad():
            return _norm(m.backbone.encode_text(toks))

    txt_orig = torch.stack([enc_text_orig([orig[str(c)]])[0] for c in range(N_CLASSES)])  # (7,512)

    ds = EpisodicDermaMNIST(split=SPLIT, augment=False)

    # report test-split class counts (rare-class within-episode replacement check)
    print("Class counts (split):", {c: ds.class_counts.get(c, 0) for c in range(N_CLASSES)})
    print("  (n_needed per class = k+q = 20; classes below 20 sample WITH replacement -> support/query overlap)\n")

    all_results = {meth: {"overall": [], "per_class": {c: [] for c in range(N_CLASSES)}} for meth in METHODS}

    for seed in SEEDS:
        np.random.seed(seed); torch.manual_seed(seed)
        cor = {meth: 0 for meth in METHODS}
        tot = 0
        pc_cor = {meth: {c: 0 for c in range(N_CLASSES)} for meth in METHODS}
        pc_tot = {c: 0 for c in range(N_CLASSES)}
        t0 = time.time()
        with torch.no_grad():
            for ep in range(EP):
                s_img, _, q_img, q_lbl, oc = ds.sample_episode(N_WAY, K_SHOT, N_QUERY, return_class_ids=True)
                s_img, q_img = s_img.to(dev), q_img.to(dev)
                oc = np.array(oc); q_lbl = q_lbl.numpy()
                lg = oc[q_lbl]                                       # global true labels

                # query image-only embedding (shared by all methods)
                q_emb = _norm(m.backbone.encode_image(q_img))

                # support image embeddings
                s_emb = _norm(m.backbone.encode_image(s_img)).view(N_WAY, K_SHOT, -1)
                img_proto = _norm(s_emb.mean(1))                    # (5,512)

                # --- fusion (forward path): encode_multimodal then mean ---
                s_txt = [orig[str(c)] for c in oc for _ in range(K_SHOT)]
                fused = m.encode_multimodal(s_img, s_txt)
                fus_proto = _norm(fused.view(N_WAY, K_SHOT, -1).mean(1))

                # --- m3 blend: beta*img + (1-beta)*gap-shifted-cupl, per episode class ---
                t = txt_m3[oc]                                      # (5,512)
                m3_proto = _norm(beta * img_proto + (1 - beta) * t)

                protos = {"fusion": fus_proto, "image_only": img_proto, "m3_blend": m3_proto}

                for c in lg:
                    pc_tot[int(c)] += 1
                tot += len(lg)
                for meth in METHODS:
                    pl = torch.softmax(-torch.cdist(q_emb, protos[meth]) * temp, 1).argmax(1).cpu().numpy()
                    pg = oc[pl]
                    cor[meth] += int((pg == lg).sum())
                    for p, t_ in zip(pg, lg):
                        if p == t_:
                            pc_cor[meth][int(t_)] += 1

        dt = (time.time() - t0) / 60
        line = f"seed {seed:<4} ({EP}ep, {dt:.1f}min): "
        for meth in METHODS:
            acc = 100.0 * cor[meth] / tot
            all_results[meth]["overall"].append(acc)
            for c in range(N_CLASSES):
                if pc_tot[c]:
                    all_results[meth]["per_class"][c].append(100.0 * pc_cor[meth][c] / pc_tot[c])
            line += f"{meth}={acc:.2f}%  "
        print(line)

    # ── Summary ──
    sep = "─" * 78
    print(f"\n{sep}\n  RECONCILIATION SUMMARY  (mean±std over seeds {SEEDS}, {EP} ep each)\n{sep}")
    print(f"  {'Method':<14}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}")
    summary = {}
    for meth in METHODS:
        a = np.array(all_results[meth]["overall"])
        summary[meth] = {"mean": round(a.mean(),2), "std": round(a.std(),2),
                         "min": round(a.min(),2), "max": round(a.max(),2),
                         "per_seed": [round(x,2) for x in a.tolist()]}
        print(f"  {meth:<14}{a.mean():>7.2f}%{a.std():>7.2f}%{a.min():>7.2f}%{a.max():>7.2f}%")
    print(f"\n  Δ m3_blend − image_only : {summary['m3_blend']['mean']-summary['image_only']['mean']:+.2f} (mean)")
    print(f"  Δ m3_blend − fusion     : {summary['m3_blend']['mean']-summary['fusion']['mean']:+.2f} (mean)")
    print(f"  Δ image_only − fusion   : {summary['image_only']['mean']-summary['fusion']['mean']:+.2f} (mean)")

    print(f"\n{sep}\n  Per-class mean accuracy (rare *)\n{sep}")
    print(f"  {'Class':<24}{'fusion':>9}{'image':>9}{'m3':>9}")
    pc_summary = {}
    for c in range(N_CLASSES):
        mark = "*" if c in RARE else " "
        vals = {meth: np.mean(all_results[meth]["per_class"][c]) if all_results[meth]["per_class"][c] else float('nan')
                for meth in METHODS}
        pc_summary[c] = {meth: round(vals[meth],1) for meth in METHODS}
        print(f"  {mark}{CLASS_NAMES[c]:<23}{vals['fusion']:>8.1f}%{vals['image_only']:>8.1f}%{vals['m3_blend']:>8.1f}%")

    json.dump({"config": {"seeds": SEEDS, "ep": EP, "split": SPLIT, "temp": round(temp,3),
                          "beta": beta, "lam": lam},
               "overall": summary, "per_class": pc_summary},
              open(OUT, "w"), indent=2)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
