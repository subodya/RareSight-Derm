"""
Significance check (per advisor): is the CoOp-vs-M3 deficit on NOVEL classes real,
or inside episode noise?

RESULTS.md novel 3-way {1,3,4}: image 68.92 / M3 70.77 / CoOp 70.30  (CoOp -0.47 vs M3).
That -0.47 is a point estimate. Before building CoCoOp to "close" it, test it: collect
PER-EPISODE accuracy for image-only / M3 / CoOp over the SAME 300 episodes (rng reset)
and run paired t-tests. If CoOp-vs-M3 is non-significant, the honest finding is already
"on novel classes CoOp is statistically indistinguishable from M3" and CoCoOp would be
chasing noise.

Reuses the already-trained context  checkpoints/coop_novel_134_ctx.pt  (no retraining).
Read-only: touches no deployed artifact.

Run:  python src/training/_diag_coop_novel_sig.py
"""
import sys, os, json, numpy as np, torch
from scipy import stats
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.models.coop_prompt import CoOpPromptLearner
from src.data.dataset import EpisodicDermaMNIST

NOVEL   = [1, 3, 4]
N_WAY   = len(NOVEL)
K_SHOT, N_QUERY = 5, 15
EV      = 300
SEED    = 42
CKPT    = "checkpoints/raresight_nblk4mix.pth"
BLEND   = "src/app/assets/blend_params.pt"
CTX_IN  = "checkpoints/coop_novel_134_ctx.pt"
OUT     = "checkpoints/coop_novel_sig.json"
CLASS_NAMES = {0:"actinic keratoses",1:"basal cell carcinoma",2:"benign keratosis",
               3:"dermatofibroma",4:"melanoma",5:"melanocytic nevi",6:"vascular lesions"}


def _norm(x): return x / x.norm(dim=-1, keepdim=True)


def sample_from(ds, allowed, n_way, rng):
    cls = rng.choice(allowed, size=n_way, replace=False).tolist()
    s_imgs, q_imgs, q_lbl = [], [], []
    for i, c in enumerate(cls):
        idx = ds.class_to_indices[c]
        need = K_SHOT + N_QUERY
        sel = rng.choice(idx, size=need, replace=len(idx) < need)
        for j in sel[:K_SHOT]: s_imgs.append(ds._load_image(j))
        for j in sel[K_SHOT:]:
            q_imgs.append(ds._load_image(j)); q_lbl.append(i)
    return torch.stack(s_imgs), torch.stack(q_imgs), torch.tensor(q_lbl), cls


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  novel split={NOVEL}  n_way={N_WAY}  episodes={EV}")

    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    for p in m.parameters(): p.requires_grad = False
    temp = m.temperature.item()

    bp = torch.load(BLEND, map_location=dev)
    beta, lam, gap = bp["beta"], bp["lam"], bp["gap"].to(dev)
    cupl_text = {c: bp["text_embs"][c].to(dev) for c in bp["text_embs"]}

    learner = CoOpPromptLearner(m, CLASS_NAMES, n_ctx=4, device=dev)
    saved = torch.load(CTX_IN, map_location=dev)
    learner.ctx.data.copy_(saved["ctx"].to(dev))
    print(f"loaded CoOp ctx from {CTX_IN}  (mode={saved.get('mode')}, n_ctx={saved.get('n_ctx')})")

    test_ds = EpisodicDermaMNIST(split="test", augment=False)

    def blend_proto(img_proto, text_feat):
        txt = _norm(text_feat + lam * gap)
        return _norm(beta * img_proto + (1 - beta) * txt)

    @torch.no_grad()
    def encode_imgs(imgs):
        return _norm(m.backbone.encode_image(imgs.to(dev)))

    cupl_fn    = lambda cls: torch.stack([cupl_text[c] for c in cls])
    learned_fn = lambda cls: learner.text_features(cls)

    @torch.no_grad()
    def per_episode_acc(text_fn):
        """Return np.array of length EV: accuracy per episode (same episode sequence)."""
        rng = np.random.RandomState(SEED)
        accs = np.empty(EV)
        for e in range(EV):
            s, q, ql, cls = sample_from(test_ds, NOVEL, N_WAY, rng)
            ip = _norm(encode_imgs(s).view(N_WAY, K_SHOT, -1).mean(1))
            qe = encode_imgs(q)
            proto = ip if text_fn is None else blend_proto(ip, text_fn(cls))
            pred = torch.softmax(-torch.cdist(qe, proto) * temp, 1).argmax(1).cpu()
            accs[e] = float((pred == ql).float().mean())
        return accs

    print("scoring image-only ...");  img  = per_episode_acc(None)
    print("scoring M3 ...");          m3   = per_episode_acc(cupl_fn)
    print("scoring CoOp ...");        coop = per_episode_acc(learned_fn)

    def summ(name, a):
        print(f"  {name:<12} {100*a.mean():.2f}%  (sd {100*a.std():.2f})")
        return round(100 * float(a.mean()), 2)

    print("\n=== per-episode means (novel 3-way {1,3,4}) ===")
    res = {"split": NOVEL, "n_way": N_WAY, "episodes": EV, "seed": SEED,
           "image_only": summ("image-only", img),
           "M3": summ("M3", m3),
           "CoOp": summ("CoOp", coop)}

    def paired(name, a, b):
        t, p = stats.ttest_rel(a, b)
        d = 100 * float((a - b).mean())
        print(f"  {name:<16} d={d:+.2f}pt  paired t={t:+.3f}  p={p:.4f}  "
              f"{'SIGNIFICANT' if p < 0.05 else 'ns (noise)'}")
        return {"delta_pt": round(d, 2), "t": round(float(t), 3), "p": round(float(p), 4),
                "significant": bool(p < 0.05)}

    print("\n=== paired t-tests (same episodes) ===")
    res["CoOp_vs_M3"]    = paired("CoOp vs M3",    coop, m3)
    res["CoOp_vs_image"] = paired("CoOp vs image", coop, img)
    res["M3_vs_image"]   = paired("M3 vs image",   m3,   img)

    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nSaved -> {OUT}")
    print("\nINTERPRETATION:")
    if not res["CoOp_vs_M3"]["significant"]:
        print("  CoOp-vs-M3 on novel classes is NOT significant -> the -0.47 'gap' is noise.")
        print("  CoCoOp's bar is therefore: do NOT regress in-distribution, and ideally")
        print("  produce a SIGNIFICANT novel gain over BOTH CoOp and M3.")
    else:
        print("  CoOp-vs-M3 novel gap IS significant -> CoCoOp has a real deficit to close.")


if __name__ == "__main__":
    main()
