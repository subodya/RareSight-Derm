"""
Weighted prototypes vs the plain mean prototype (training-free).

Proposal RO3 promised a "weighted prototype network" that downweights noisy/outlier
support examples. This evaluates that idea as a drop-in, training-free re-weighting of
the support embeddings — no model change, no deployed-artifact change.

For a class with K normalized support embeddings {x_i} and mean m = norm(mean_i x_i):
    w_i = softmax_i( (x_i . m) / tau )           # closer-to-centroid shots weigh more
    proto = norm( sum_i w_i x_i )
tau -> inf recovers the uniform mean. tau is the ONLY tuned knob (swept on val); the
uniform mean is a fixed, untuned baseline. Reported both image-only and on top of the
deployed M3 (CuPL + modality-gap) text blend so we see if weighting helps the real recipe.

Both arms are computed from the SAME cached episodes (paired), enabling a paired t-test
over per-episode accuracies (fulfils the proposal's RO6 paired-test requirement).

Run:  conda run -n raresight python src/training/eval_weighted_proto.py
Env:  WP_VAL_EP (default 200), WP_TEST_EP (default 300)
Out:  checkpoints/weighted_proto_results.json
"""

import sys, os, json, time, numpy as np, torch
from scipy import stats
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.data.dataset import EpisodicDermaMNIST

N_WAY, K_SHOT, N_QUERY = 5, 5, 15
VAL_EP  = int(os.environ.get("WP_VAL_EP", "200"))
TEST_EP = int(os.environ.get("WP_TEST_EP", "300"))
SEED, N_CLASSES = 42, 7
CKPT = "checkpoints/raresight_nblk4mix.pth"
OUT  = "checkpoints/weighted_proto_results.json"
DESC_CUPL = os.path.join(os.path.dirname(__file__), "../../src/app/cupl_descriptions.json")

TAU_GRID  = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]   # softmax temperature on cosine-to-mean
BETA_GRID = [round(x, 2) for x in np.arange(0.50, 1.001, 0.05)]
LAM_GRID  = [0.0, 0.25, 0.5, 0.75, 1.0]
CLASS_NAMES = {0:"Actinic keratoses",1:"Basal cell carcinoma",2:"Benign keratosis",
               3:"Dermatofibroma",4:"Melanoma",5:"Melanocytic nevi",6:"Vascular lesions"}
RARE = {3, 6}


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def mean_proto(se):                      # se: (N_WAY, K_SHOT, 512) normalized supports
    return _norm(se.mean(1))             # (N_WAY, 512)


def weighted_proto(se, tau):
    m = _norm(se.mean(1))                            # (N,512) class mean
    sim = torch.einsum("nkd,nd->nk", se, m)          # (N,K) cosine-to-mean
    w = torch.softmax(sim / tau, dim=1).unsqueeze(-1)  # (N,K,1)
    return _norm((w * se).sum(1))                     # (N,512)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cupl = json.load(open(os.path.abspath(DESC_CUPL)))
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    temp = m.temperature.item()

    def enc_text(texts):
        toks = m.tokenizer(texts, padding="max_length", truncation=True,
                           max_length=256, return_tensors="pt")["input_ids"].to(dev)
        with torch.no_grad():
            return _norm(m.backbone.encode_text(toks))

    print("Encoding CuPL class text embeddings...")
    txt_cupl = torch.stack([_norm(enc_text(cupl[str(c)]).mean(0)) for c in range(N_CLASSES)]).cpu()

    def cache_split(split, n_ep):
        ds = EpisodicDermaMNIST(split=split, augment=False)
        eps = []
        np.random.seed(SEED); torch.manual_seed(SEED)
        with torch.no_grad():
            for _ in range(n_ep):
                s_img, _, q_img, q_lbl, oc = ds.sample_episode(N_WAY, K_SHOT, N_QUERY, return_class_ids=True)
                q_emb = _norm(m.backbone.encode_image(q_img.to(dev))).cpu()
                se = _norm(m.backbone.encode_image(s_img.to(dev))).view(N_WAY, K_SHOT, -1).cpu()  # per-shot
                eps.append({"q": q_emb, "se": se, "oc": np.array(oc), "ql": q_lbl.numpy()})
        return eps

    print(f"Caching VAL ({VAL_EP}) + TEST ({TEST_EP}) episode embeddings...")
    t0 = time.time()
    val_eps  = cache_split("val",  VAL_EP)
    test_eps = cache_split("test", TEST_EP)
    print(f"  cached in {(time.time()-t0)/60:.1f} min")

    # modality gap from val (mean image prototype - mean CuPL text), for M3
    mean_img = torch.stack([mean_proto(e["se"]) for e in val_eps]).reshape(-1, 512).mean(0)
    gap = _norm(mean_img - txt_cupl.mean(0))

    def shifted_text(lam):
        return _norm(txt_cupl + lam * gap)

    def ep_acc(e, proto):                          # per-episode accuracy (paired t-test unit)
        logits = -torch.cdist(e["q"], proto) * temp
        pg = e["oc"][logits.argmax(1).numpy()]; lg = e["oc"][e["ql"]]
        return (pg == lg)

    def proto_img(e, tau):                          # image prototype: uniform mean or weighted
        return mean_proto(e["se"]) if tau is None else weighted_proto(e["se"], tau)

    def proto_m3(e, tau, beta, txt):                # M3 blend on chosen image prototype
        t = txt[e["oc"]]
        return _norm(beta * proto_img(e, tau) + (1 - beta) * t)

    def overall(eps, proto_fn, per_class=False):
        flags, pc = [], {c: [0, 0] for c in range(N_CLASSES)}
        ep_means = []
        for e in eps:
            f = ep_acc(e, proto_fn(e)); flags.append(f); ep_means.append(f.mean())
            if per_class:
                lg = e["oc"][e["ql"]]
                for ok, t in zip(f, lg):
                    pc[int(t)][1] += 1; pc[int(t)][0] += int(ok)
        acc = 100.0 * np.concatenate(flags).mean()
        ep_means = np.array(ep_means)
        if per_class:
            return acc, ep_means, {c: (100.0*pc[c][0]/pc[c][1] if pc[c][1] else None) for c in range(N_CLASSES)}
        return acc, ep_means

    results = {"_config": {"val_ep": VAL_EP, "test_ep": TEST_EP, "seed": SEED,
                           "n_way": N_WAY, "k_shot": K_SHOT, "temp": round(temp, 3),
                           "protocol": "5-way 5-shot episodic, HAM10000 test, paired"}}

    # ── Tune tau on val for image-only weighting ──
    base_val, _ = overall(val_eps, lambda e: proto_img(e, None))
    best_tau, best_v = None, base_val
    for tau in TAU_GRID:
        v, _ = overall(val_eps, lambda e, tau=tau: proto_img(e, tau))
        if v > best_v: best_v, best_tau = v, tau
    print(f"\nimage-only  val: uniform={base_val:.2f}  best weighted={best_v:.2f} @ tau={best_tau}")

    def report(tag, mean_fn, wtd_fn):
        a_mean, ep_mean = overall(test_eps, mean_fn)
        a_wtd,  ep_wtd, pc_wtd = overall(test_eps, wtd_fn, per_class=True)
        _, _, pc_mean = overall(test_eps, mean_fn, per_class=True)
        tstat, p = stats.ttest_rel(ep_wtd, ep_mean)
        print(f"{tag:<10} mean={a_mean:.2f}  weighted={a_wtd:.2f}  Δ={a_wtd-a_mean:+.2f}  "
              f"paired t p={p:.3f}")
        return {"mean": round(a_mean, 2), "weighted": round(a_wtd, 2),
                "delta": round(a_wtd - a_mean, 2), "paired_t_p": round(float(p), 4),
                "per_class": {str(c): {"mean": None if pc_mean[c] is None else round(pc_mean[c],1),
                                       "weighted": None if pc_wtd[c] is None else round(pc_wtd[c],1)}
                              for c in range(N_CLASSES)}}

    # (A) image-only: uniform mean vs weighted(tau*)
    if best_tau is None:
        print("  (val prefers the uniform mean — weighting did not help; reporting tau=0.2 for the table)")
    tau_use = best_tau if best_tau is not None else 0.2
    results["A_image_only"] = report("image", lambda e: proto_img(e, None),
                                     lambda e: proto_img(e, tau_use))
    results["A_image_only"]["best_tau"] = best_tau
    results["A_image_only"]["val_uniform"] = round(base_val, 2)
    results["A_image_only"]["val_weighted_best"] = round(best_v, 2)

    # (B) M3 recipe: tune (lam,beta) on val with the UNIFORM mean, then swap in weighted proto
    best = {"val": -1}
    for lam in LAM_GRID:
        txt = shifted_text(lam)
        for beta in BETA_GRID:
            v, _ = overall(val_eps, lambda e, lam=lam, beta=beta, txt=txt: proto_m3(e, None, beta, txt))
            if v > best["val"]: best = {"val": v, "lam": lam, "beta": beta}
    txt_best = shifted_text(best["lam"])
    print(f"M3 val-tuned: lam={best['lam']} beta={best['beta']} (val {best['val']:.2f})")
    results["B_m3"] = report("M3",
                             lambda e: proto_m3(e, None,    best["beta"], txt_best),
                             lambda e: proto_m3(e, tau_use, best["beta"], txt_best))
    results["B_m3"]["m3_lam"] = best["lam"]; results["B_m3"]["m3_beta"] = best["beta"]

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nSaved → {OUT}")
    print("Note: same-class embeddings cluster tightly, so a near-zero/negative Δ is the "
          "expected, honest finding — uniform mean is a strong baseline.")


if __name__ == "__main__":
    main()
