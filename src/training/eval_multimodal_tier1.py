"""
Tier-1 multimodal improvements over image-only prototypes (all training-free).

Methods compared on identical episodes (hyperparams swept on VAL, reported on TEST):
  M0 image_only                  : prototype = mean support image embedding (beta=1)
  M1 blend_orig   (beta)         : aligned-space blend with the ORIGINAL 1-line text
  M2 blend_cupl   (beta)         : aligned-space blend with CuPL prompt-ensemble text
  M3 blend_cupl_gap (beta, lam)  : M2 + modality-gap correction (shift text toward images)
  M4 blend_cupl_gap_perclass     : M3 with a PER-CLASS beta_c (exploits class heterogeneity)
  M5 tipx_logit   (alpha)        : Tip-Adapter-style logit fusion (image few-shot + text zero-shot)

Text embedding per class:
  orig : normalize(encode_text(single description))                 [class_descriptions.json]
  cupl : normalize(mean_p normalize(encode_text(prompt_p)))         [cupl_descriptions.json]

Run:
  conda run -n raresight python src/training/eval_multimodal_tier1.py
Env overrides: T1_VAL_EP (default 200), T1_TEST_EP (default 300)
"""

import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.data.dataset import EpisodicDermaMNIST

N_WAY, K_SHOT, N_QUERY = 5, 5, 15
VAL_EP  = int(os.environ.get("T1_VAL_EP", "200"))
TEST_EP = int(os.environ.get("T1_TEST_EP", "300"))
SEED, N_CLASSES = 42, 7
CKPT = "checkpoints/raresight_nblk4mix.pth"
OUT  = "checkpoints/tier1_results.json"
DESC_ORIG = os.path.join(os.path.dirname(__file__), "../../src/app/class_descriptions.json")
DESC_CUPL = os.path.join(os.path.dirname(__file__), "../../src/app/cupl_descriptions.json")

BETA_GRID  = [round(x, 2) for x in np.arange(0.50, 1.001, 0.05)]
LAM_GRID   = [0.0, 0.25, 0.5, 0.75, 1.0]
ALPHA_GRID = [round(x, 2) for x in np.arange(0.0, 2.01, 0.25)]
CLASS_NAMES = {0:"Actinic keratoses",1:"Basal cell carcinoma",2:"Benign keratosis",
               3:"Dermatofibroma",4:"Melanoma",5:"Melanocytic nevi",6:"Vascular lesions"}
RARE = {3, 6}


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    orig = json.load(open(os.path.abspath(DESC_ORIG)))
    cupl = json.load(open(os.path.abspath(DESC_CUPL)))
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    temp = m.temperature.item()

    def enc_text(texts):
        toks = m.tokenizer(texts, padding="max_length", truncation=True,
                           max_length=256, return_tensors="pt")["input_ids"].to(dev)
        with torch.no_grad():
            return _norm(m.backbone.encode_text(toks))

    # ── Precompute per-class text embeddings (global, reused every episode) ──
    print("Encoding class text embeddings (orig + CuPL ensemble)...")
    txt_orig = torch.stack([enc_text([orig[str(c)]])[0] for c in range(N_CLASSES)]).cpu()  # (7,512)
    txt_cupl = torch.stack([_norm(enc_text(cupl[str(c)]).mean(0)) for c in range(N_CLASSES)]).cpu()  # (7,512)

    # ── Cache episode embeddings for a split ──
    def cache_split(split, n_ep):
        ds = EpisodicDermaMNIST(split=split, augment=False)
        eps = []
        np.random.seed(SEED); torch.manual_seed(SEED)
        with torch.no_grad():
            for _ in range(n_ep):
                s_img, _, q_img, q_lbl, oc = ds.sample_episode(N_WAY, K_SHOT, N_QUERY, return_class_ids=True)
                q_emb = _norm(m.backbone.encode_image(q_img.to(dev))).cpu()
                ip = _norm(_norm(m.backbone.encode_image(s_img.to(dev))).view(N_WAY, K_SHOT, -1).mean(1)).cpu()
                eps.append({"q": q_emb, "ip": ip, "oc": np.array(oc), "ql": q_lbl.numpy()})
        return eps

    print(f"Caching VAL ({VAL_EP}) and TEST ({TEST_EP}) episode embeddings...")
    t0 = time.time()
    val_eps  = cache_split("val",  VAL_EP)
    test_eps = cache_split("test", TEST_EP)
    print(f"  cached in {(time.time()-t0)/60:.1f} min")

    # Global modality gap from val: mean image prototype - mean CuPL text
    mean_img = torch.stack([e["ip"] for e in val_eps]).reshape(-1, 512).mean(0)
    gap = mean_img - txt_cupl.mean(0)
    gap = gap / gap.norm()
    print(f"  modality-gap |mean_img-mean_txt| = {(mean_img - txt_cupl.mean(0)).norm():.3f}")

    # ── Scoring helpers (operate on cached CPU tensors) ──
    def episode_logits(e, proto):
        return -torch.cdist(e["q"], proto) * temp     # (Nq, 5)

    def acc_from_logits(eps, logit_fn, per_class=False):
        cor = tot = 0
        pc = {c: [0, 0] for c in range(N_CLASSES)}
        for e in eps:
            logits = logit_fn(e)
            pl = logits.argmax(1).numpy()
            pg = e["oc"][pl]; lg = e["oc"][e["ql"]]
            cor += int((pg == lg).sum()); tot += len(lg)
            if per_class:
                for p, t in zip(pg, lg):
                    pc[int(t)][1] += 1
                    if p == t: pc[int(t)][0] += 1
        acc = 100.0 * cor / tot
        if per_class:
            return acc, {c: (100.0*pc[c][0]/pc[c][1] if pc[c][1] else None) for c in range(N_CLASSES)}
        return acc

    def shifted_text(txt, lam):
        return _norm(txt + lam * gap)

    # method logit-fns (closures over hyperparams)
    def lf_image(e):
        return episode_logits(e, e["ip"])

    def lf_blend(txt, beta):
        def f(e):
            t = txt[e["oc"]]                                   # (5,512)
            proto = _norm(beta * e["ip"] + (1 - beta) * t)
            return episode_logits(e, proto)
        return f

    def lf_blend_perclass(txt, beta_vec):
        def f(e):
            t = txt[e["oc"]]
            b = torch.tensor([float(beta_vec[c]) for c in e["oc"]], dtype=torch.float32).unsqueeze(1)   # (5,1)
            proto = _norm(b * e["ip"] + (1 - b) * t)
            return episode_logits(e, proto)
        return f

    def lf_tipx(txt, alpha):
        def f(e):
            t = txt[e["oc"]]
            img_logits = episode_logits(e, e["ip"])
            txt_logits = (e["q"] @ t.T) * temp                 # cosine * temp
            return img_logits + alpha * txt_logits
        return f

    results = {}

    # M0 image-only
    img_acc = acc_from_logits(test_eps, lf_image)
    print(f"\nM0 image_only           TEST = {img_acc:.2f}%")
    results["M0_image_only"] = {"test": round(img_acc, 2)}

    # helper: sweep 1-D grid on val
    def sweep(make_fn, grid, label):
        best_v, best_h = -1, None
        for h in grid:
            v = acc_from_logits(val_eps, make_fn(h))
            if v > best_v: best_v, best_h = v, h
        test_acc = acc_from_logits(test_eps, make_fn(best_h))
        print(f"{label:<24} val_best={best_v:.2f}% @ {best_h}  ->  TEST = {test_acc:.2f}%  (Δ{test_acc-img_acc:+.2f})")
        return {"best_hparam": best_h, "val": round(best_v, 2), "test": round(test_acc, 2),
                "delta_vs_image": round(test_acc - img_acc, 2)}

    results["M1_blend_orig"] = sweep(lambda b: lf_blend(txt_orig, b), BETA_GRID, "M1 blend_orig (beta)")
    results["M2_blend_cupl"] = sweep(lambda b: lf_blend(txt_cupl, b), BETA_GRID, "M2 blend_cupl (beta)")

    # M3 blend_cupl_gap: 2-D sweep (lam, beta)
    best = {"test": -1}
    for lam in LAM_GRID:
        txt_l = shifted_text(txt_cupl, lam)
        for b in BETA_GRID:
            v = acc_from_logits(val_eps, lf_blend(txt_l, b))
            if v > best.get("val", -1):
                best = {"lam": lam, "beta": b, "val": v}
    txt_best = shifted_text(txt_cupl, best["lam"])
    t3 = acc_from_logits(test_eps, lf_blend(txt_best, best["beta"]))
    print(f"{'M3 blend_cupl_gap':<24} val_best={best['val']:.2f}% @ lam={best['lam']},beta={best['beta']}  ->  TEST = {t3:.2f}%  (Δ{t3-img_acc:+.2f})")
    results["M3_blend_cupl_gap"] = {"best_lam": best["lam"], "best_beta": best["beta"],
                                    "val": round(best["val"], 2), "test": round(t3, 2),
                                    "delta_vs_image": round(t3 - img_acc, 2)}

    # M4 per-class beta (on the gap-shifted CuPL text). Fit beta_c by per-class val acc.
    #   For each beta, get per-class val acc; choose beta_c = argmax per class.
    txt_gap = shifted_text(txt_cupl, best["lam"])
    per_class_val = {c: {"beta": BETA_GRID[-1], "acc": -1} for c in range(N_CLASSES)}
    for b in BETA_GRID:
        _, pc = acc_from_logits(val_eps, lf_blend(txt_gap, b), per_class=True)
        for c in range(N_CLASSES):
            if pc[c] is not None and pc[c] > per_class_val[c]["acc"]:
                per_class_val[c] = {"beta": b, "acc": pc[c]}
    beta_vec = {c: per_class_val[c]["beta"] for c in range(N_CLASSES)}
    t4, t4_pc = acc_from_logits(test_eps, lf_blend_perclass(txt_gap, beta_vec), per_class=True)
    print(f"{'M4 blend_perclass_beta':<24} per-class beta={beta_vec}  ->  TEST = {t4:.2f}%  (Δ{t4-img_acc:+.2f})")
    results["M4_blend_cupl_gap_perclass"] = {"beta_per_class": beta_vec, "test": round(t4, 2),
                                             "delta_vs_image": round(t4 - img_acc, 2)}

    results["M5_tipx_logit"] = sweep(lambda a: lf_tipx(txt_cupl, a), ALPHA_GRID, "M5 tipx_logit (alpha)")

    # Per-class table for image-only vs the best blend method (M3) and M4
    _, pc_img = acc_from_logits(test_eps, lf_image, per_class=True)
    _, pc_m3  = acc_from_logits(test_eps, lf_blend(txt_best, best["beta"]), per_class=True)
    print(f"\n  Per-class TEST (rare *):   {'image':>8}{'M3':>8}{'M4':>8}")
    pc_out = {}
    for c in range(N_CLASSES):
        if pc_img[c] is None: continue
        mark = "*" if c in RARE else " "
        print(f"  {mark}{CLASS_NAMES[c]:<22}{pc_img[c]:>7.1f}%{pc_m3[c]:>7.1f}%{t4_pc[c]:>7.1f}%")
        pc_out[str(c)] = {"image": round(pc_img[c],1), "M3": round(pc_m3[c],1),
                          "M4": round(t4_pc[c],1) if t4_pc[c] is not None else None}
    results["per_class_test"] = pc_out
    results["_config"] = {"val_ep": VAL_EP, "test_ep": TEST_EP, "seed": SEED,
                          "image_only_test": round(img_acc, 2), "temp": round(temp, 3)}

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
