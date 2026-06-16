"""
Test-time augmentation (TTA) for the query image, training-free.

Proposal RO3/RO4 promised a test-time-augmentation strategy. Dermoscopy lesions have no
canonical orientation, so the dihedral group D4 (identity, 3 rotations, 2 flips, 2
transposes = 8 views) is a label-preserving augmentation. We encode each view of the
QUERY image and average the normalized embeddings (query-only is the canonical meaning of
TTA); the support prototype is unchanged. Reported image-only and on top of the deployed
M3 (CuPL + modality-gap) blend.

Both arms (TTA off / on) use the SAME cached episodes (paired) → paired t-test over
per-episode accuracies (RO6).

Run:  conda run -n raresight python src/training/eval_tta.py
Env:  TTA_VAL_EP (default 200, for M3 tuning), TTA_TEST_EP (default 300)
Out:  checkpoints/tta_results.json
"""

import sys, os, json, time, numpy as np, torch
from scipy import stats
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.data.dataset import EpisodicDermaMNIST

N_WAY, K_SHOT, N_QUERY = 5, 5, 15
VAL_EP  = int(os.environ.get("TTA_VAL_EP", "200"))
TEST_EP = int(os.environ.get("TTA_TEST_EP", "300"))
SEED, N_CLASSES = 42, 7
CKPT = "checkpoints/raresight_nblk4mix.pth"
OUT  = "checkpoints/tta_results.json"
DESC_CUPL = os.path.join(os.path.dirname(__file__), "../../src/app/cupl_descriptions.json")

BETA_GRID = [round(x, 2) for x in np.arange(0.50, 1.001, 0.05)]
LAM_GRID  = [0.0, 0.25, 0.5, 0.75, 1.0]


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def d4_views(img):
    """img: (B,3,H,W) normalized tensor → list of 8 D4 views (geometry only)."""
    views = []
    for k in range(4):                       # rotations 0/90/180/270
        r = torch.rot90(img, k, dims=[-2, -1])
        views.append(r)
        views.append(torch.flip(r, dims=[-1]))   # + horizontal flip = full D4
    return views


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cupl = json.load(open(os.path.abspath(DESC_CUPL)))
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    temp = m.temperature.item()

    def enc_img(x):
        return _norm(m.backbone.encode_image(x.to(dev))).cpu()

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
                ip = _norm(enc_img(s_img).view(N_WAY, K_SHOT, -1).mean(1))            # (5,512)
                q_plain = enc_img(q_img)                                              # (Nq,512)
                # TTA: average normalized embeddings over the 8 D4 views, renormalize
                acc = torch.zeros_like(q_plain)
                for v in d4_views(q_img):
                    acc += enc_img(v)
                q_tta = _norm(acc / 8.0)
                eps.append({"q": q_plain, "q_tta": q_tta, "ip": ip,
                            "oc": np.array(oc), "ql": q_lbl.numpy()})
        return eps

    print(f"Caching VAL ({VAL_EP}) + TEST ({TEST_EP}) episodes (×8 TTA views — slower)...")
    t0 = time.time()
    val_eps  = cache_split("val",  VAL_EP)
    test_eps = cache_split("test", TEST_EP)
    print(f"  cached in {(time.time()-t0)/60:.1f} min")

    mean_img = torch.stack([e["ip"] for e in val_eps]).reshape(-1, 512).mean(0)
    gap = _norm(mean_img - txt_cupl.mean(0))

    def shifted_text(lam):
        return _norm(txt_cupl + lam * gap)

    def proto_img(e):
        return e["ip"]

    def proto_m3(e, beta, txt):
        return _norm(beta * e["ip"] + (1 - beta) * txt[e["oc"]])

    def ep_acc(e, proto, qkey):
        logits = -torch.cdist(e[qkey], proto) * temp
        pg = e["oc"][logits.argmax(1).numpy()]; lg = e["oc"][e["ql"]]
        return (pg == lg)

    def overall(eps, proto_fn, qkey):
        flags, ep_means = [], []
        for e in eps:
            f = ep_acc(e, proto_fn(e), qkey); flags.append(f); ep_means.append(f.mean())
        return 100.0 * np.concatenate(flags).mean(), np.array(ep_means)

    results = {"_config": {"val_ep": VAL_EP, "test_ep": TEST_EP, "seed": SEED,
                           "n_way": N_WAY, "k_shot": K_SHOT, "temp": round(temp, 3),
                           "tta": "query-only, D4 (8 views), embedding average",
                           "protocol": "5-way 5-shot episodic, HAM10000 test, paired"}}

    def report(tag, proto_fn):
        a_off, ep_off = overall(test_eps, proto_fn, "q")
        a_on,  ep_on  = overall(test_eps, proto_fn, "q_tta")
        tstat, p = stats.ttest_rel(ep_on, ep_off)
        print(f"{tag:<10} no-TTA={a_off:.2f}  TTA={a_on:.2f}  Δ={a_on-a_off:+.2f}  paired t p={p:.3f}")
        return {"no_tta": round(a_off, 2), "tta": round(a_on, 2),
                "delta": round(a_on - a_off, 2), "paired_t_p": round(float(p), 4)}

    # (A) image-only
    results["A_image_only"] = report("image", proto_img)

    # (B) M3: tune (lam,beta) on val WITHOUT TTA (plain query), then apply to test both arms
    best = {"val": -1}
    for lam in LAM_GRID:
        txt = shifted_text(lam)
        for beta in BETA_GRID:
            v, _ = overall(val_eps, lambda e, beta=beta, txt=txt: proto_m3(e, beta, txt), "q")
            if v > best["val"]: best = {"val": v, "lam": lam, "beta": beta}
    txt_best = shifted_text(best["lam"])
    print(f"M3 val-tuned: lam={best['lam']} beta={best['beta']} (val {best['val']:.2f})")
    results["B_m3"] = report("M3", lambda e: proto_m3(e, best["beta"], txt_best))
    results["B_m3"]["m3_lam"] = best["lam"]; results["B_m3"]["m3_beta"] = best["beta"]

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
