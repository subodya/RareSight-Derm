"""
RQ1 shot curve: accuracy vs K (support shots) for the RareSight method family.

For each K in {1, 5, 10}, on IDENTICAL seed-42 episodes (5-way), tune the blend on
the VAL split and report on TEST:
  zero_shot   : cosine(query, CuPL text), NO support  -> K-independent anchor line
  image_only  : mean support image prototype (M0)
  M3          : aligned CuPL+gap blend, (beta, lam) re-tuned on val PER K
  CoOp        : learned-prompt text (checkpoints/coop_indist_ctx.pt) in the SAME
                per-K-tuned blend (only the text source differs from M3)

The modality gap and (beta, lam) are recomputed from each K's own val episodes, so
the curve is honest at every K. We report the chosen (beta, lam) per K: if beta drops
at K=1 (more text weight), that shift is itself the finding — noisy 1-shot image
prototypes lean harder on text.

Run:
  conda run -n raresight python src/training/eval_shot_curve.py
Env overrides: SC_VAL_EP (default 200), SC_TEST_EP (default 300), SC_KS (default "1,5,10")
"""

import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.models.coop_prompt import CoOpPromptLearner
from src.data.dataset import EpisodicDermaMNIST

N_WAY, N_QUERY = 5, 15
VAL_EP  = int(os.environ.get("SC_VAL_EP", "200"))
TEST_EP = int(os.environ.get("SC_TEST_EP", "300"))
KS = [int(k) for k in os.environ.get("SC_KS", "1,5,10").split(",")]
SEED, N_CLASSES = 42, 7
CKPT = "checkpoints/raresight_nblk4mix.pth"
CTX  = "checkpoints/coop_indist_ctx.pt"
OUT  = "checkpoints/shot_curve_results.json"
DESC_CUPL = os.path.join(os.path.dirname(__file__), "../../src/app/cupl_descriptions.json")
# CoOp learner expects the SAME class-name dict train_coop.py trained with.
COOP_NAMES = {0:"actinic keratoses",1:"basal cell carcinoma",2:"benign keratosis",
              3:"dermatofibroma",4:"melanoma",5:"melanocytic nevi",6:"vascular lesions"}

BETA_GRID = [round(x, 2) for x in np.arange(0.50, 1.001, 0.05)]
LAM_GRID  = [0.0, 0.25, 0.5, 0.75, 1.0]


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cupl = json.load(open(os.path.abspath(DESC_CUPL)))
    m = RareSight(device=dev)
    m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    for p in m.parameters(): p.requires_grad = False
    temp = m.temperature.item()

    def enc_text(texts):
        toks = m.tokenizer(texts, padding="max_length", truncation=True,
                           max_length=256, return_tensors="pt")["input_ids"].to(dev)
        with torch.no_grad():
            return _norm(m.backbone.encode_text(toks))

    # ── Per-class text embeddings (K-independent, computed once) ──
    print("Encoding CuPL class text + loading CoOp learned context...")
    txt_cupl = torch.stack([_norm(enc_text(cupl[str(c)]).mean(0)) for c in range(N_CLASSES)]).cpu()  # (7,512)

    ckpt = torch.load(CTX, map_location=dev)
    learner = CoOpPromptLearner(m, COOP_NAMES, n_ctx=int(ckpt["n_ctx"]), device=dev)
    learner.ctx.data.copy_(ckpt["ctx"].to(dev))
    with torch.no_grad():
        txt_coop = learner.text_features(list(range(N_CLASSES))).cpu()         # (7,512)

    # ── Cache episode embeddings for a split at a given K ──
    def cache_split(split, n_ep, k_shot):
        ds = EpisodicDermaMNIST(split=split, augment=False)
        eps = []
        np.random.seed(SEED); torch.manual_seed(SEED)
        with torch.no_grad():
            for _ in range(n_ep):
                s_img, _, q_img, q_lbl, oc = ds.sample_episode(
                    N_WAY, k_shot, N_QUERY, return_class_ids=True)
                q_emb = _norm(m.backbone.encode_image(q_img.to(dev))).cpu()
                ip = _norm(_norm(m.backbone.encode_image(s_img.to(dev)))
                           .view(N_WAY, k_shot, -1).mean(1)).cpu()
                eps.append({"q": q_emb, "ip": ip, "oc": np.array(oc), "ql": q_lbl.numpy()})
        return eps

    # ── Scoring helpers (operate on cached CPU tensors) ──
    def acc(eps, logit_fn):
        cor = tot = 0
        for e in eps:
            pl = logit_fn(e).argmax(1).numpy()
            pg = e["oc"][pl]; lg = e["oc"][e["ql"]]
            cor += int((pg == lg).sum()); tot += len(lg)
        return 100.0 * cor / tot

    def lf_image(e):
        return -torch.cdist(e["q"], e["ip"]) * temp

    def lf_zeroshot(txt):
        def f(e):
            return (e["q"] @ txt[e["oc"]].T) * temp          # cosine * temp, no support
        return f

    def lf_blend(txt, gap, beta, lam):
        def f(e):
            t = _norm(txt[e["oc"]] + lam * gap)
            proto = _norm(beta * e["ip"] + (1 - beta) * t)
            return -torch.cdist(e["q"], proto) * temp
        return f

    results = {"_config": {"val_ep": VAL_EP, "test_ep": TEST_EP, "seed": SEED,
                           "n_way": N_WAY, "n_query": N_QUERY, "temp": round(temp, 3),
                           "ks": KS}, "curve": {}}

    print(f"\nShot curve over K={KS}  (val {VAL_EP} ep, test {TEST_EP} ep, 5-way, seed {SEED})")
    print(f"{'K':>3} {'zero':>7} {'image':>7} {'M3':>7} {'CoOp':>7}   "
          f"{'beta':>5} {'lam':>5}  {'M3-img':>7} {'CoOp-img':>9} {'CoOp-M3':>8}")
    for k in KS:
        t0 = time.time()
        val_eps  = cache_split("val",  VAL_EP, k)
        test_eps = cache_split("test", TEST_EP, k)

        # modality gap from THIS K's val episodes
        mean_img = torch.stack([e["ip"] for e in val_eps]).reshape(-1, 512).mean(0)
        gap = _norm(mean_img - txt_cupl.mean(0))

        # tune (beta, lam) for the blend on val (shared by M3 and CoOp)
        best = {"val": -1, "beta": BETA_GRID[-1], "lam": 0.0}
        for lam in LAM_GRID:
            for beta in BETA_GRID:
                v = acc(val_eps, lf_blend(txt_cupl, gap, beta, lam))
                if v > best["val"]:
                    best = {"val": v, "beta": beta, "lam": lam}
        beta, lam = best["beta"], best["lam"]

        a_zero  = acc(test_eps, lf_zeroshot(txt_cupl))
        a_img   = acc(test_eps, lf_image)
        a_m3    = acc(test_eps, lf_blend(txt_cupl, gap, beta, lam))
        a_coop  = acc(test_eps, lf_blend(txt_coop, gap, beta, lam))

        results["curve"][str(k)] = {
            "zero_shot": round(a_zero, 2), "image_only": round(a_img, 2),
            "M3": round(a_m3, 2), "CoOp": round(a_coop, 2),
            "blend_beta": beta, "blend_lam": lam, "val_best": round(best["val"], 2),
            "M3_minus_image": round(a_m3 - a_img, 2),
            "CoOp_minus_image": round(a_coop - a_img, 2),
            "CoOp_minus_M3": round(a_coop - a_m3, 2),
        }
        print(f"{k:>3} {a_zero:>7.2f} {a_img:>7.2f} {a_m3:>7.2f} {a_coop:>7.2f}   "
              f"{beta:>5.2f} {lam:>5.2f}  {a_m3-a_img:>+7.2f} {a_coop-a_img:>+9.2f} "
              f"{a_coop-a_m3:>+8.2f}   ({(time.time()-t0)/60:.1f} min)")

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
