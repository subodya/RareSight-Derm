"""
BiomedCoOp-style learnable prompt tuning for RareSight (training-based RQ2 capstone).

Trains M context vectors (3072 params) so the learned per-class text feature, placed
in BiomedCLIP's aligned space, improves few-shot prototypes over CuPL (M3).

Design decisions (per advisor):
  * Trainable surface = context vectors ONLY (backbone frozen).
  * blend recipe HELD FIXED at M3 values (beta, lam, gap from assets/blend_params.pt)
    so we isolate "better prompt" from "better blend knobs".
  * BiomedCoOp regularizer: feature distillation pulling learned text toward the CuPL
    ensemble (fights prompt collapse / overfitting on few classes).
  * EVALUATED ON TWO PROTOCOLS:
      - indist : train context on all 7 classes, eval 5-way on all 7
      - novel  : train context on 5 BASE classes, eval 2-way on HELD-OUT {3,6}
    The novel protocol is the honest test of whether learned prompts GENERALIZE,
    not just memorize trained classes (the class-non-disjointness viva risk).

Run:
  COOP_MODE=indist  python src/training/train_coop.py     (use direct env python.exe)
  COOP_MODE=novel   python src/training/train_coop.py
"""
import sys, os, json, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.models.coop_prompt import CoOpPromptLearner
from src.data.dataset import EpisodicDermaMNIST

MODE       = os.environ.get("COOP_MODE", "indist")          # indist | novel
N_CTX      = int(os.environ.get("COOP_NCTX", "4"))
EPISODES   = int(os.environ.get("COOP_EPISODES", "1500"))
LR         = 2e-3
REG_W      = float(os.environ.get("COOP_REG", "1.0"))        # CuPL distillation weight
K_SHOT, N_QUERY = 5, 15
VAL_INT, VAL_EP = 150, 60
SEED = 42
CKPT = "checkpoints/raresight_nblk4mix.pth"
BLEND = "src/app/assets/blend_params.pt"
CLASS_NAMES = {0:"actinic keratoses",1:"basal cell carcinoma",2:"benign keratosis",
               3:"dermatofibroma",4:"melanoma",5:"melanocytic nevi",6:"vascular lesions"}
NOVEL = [int(c) for c in os.environ.get("COOP_NOVEL", "3,6").split(",")]
BASE  = [c for c in range(7) if c not in NOVEL]
# distinct output tag per novel split so runs don't overwrite each other
_TAG = MODE if MODE != "novel" else "novel_" + "".join(str(c) for c in NOVEL)
OUT = f"checkpoints/coop_{_TAG}_results.json"
CTX_OUT = f"checkpoints/coop_{_TAG}_ctx.pt"


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
    return (torch.stack(s_imgs), torch.stack(q_imgs),
            torch.tensor(q_lbl), cls)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_way = len(NOVEL) if MODE == "novel" else 5
    train_classes = BASE if MODE == "novel" else list(range(7))
    eval_classes  = NOVEL if MODE == "novel" else list(range(7))
    print(f"MODE={MODE}  n_way={n_way}  train_classes={train_classes}  eval_classes={eval_classes}")

    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    for p in m.parameters(): p.requires_grad = False
    temp = m.temperature.item()

    bp = torch.load(BLEND, map_location=dev)
    beta, lam, gap = bp["beta"], bp["lam"], bp["gap"].to(dev)
    cupl_text = {c: bp["text_embs"][c].to(dev) for c in bp["text_embs"]}     # CuPL targets

    learner = CoOpPromptLearner(m, CLASS_NAMES, n_ctx=N_CTX, device=dev)
    assert learner.verify_forward() < 1e-3, "CoOp forward path does not match encode_text!"
    opt = torch.optim.AdamW([learner.ctx], lr=LR, weight_decay=1e-4)

    train_ds = EpisodicDermaMNIST(split="train", augment=True)
    val_ds   = EpisodicDermaMNIST(split="val",   augment=False)
    test_ds  = EpisodicDermaMNIST(split="test",  augment=False)

    def blend_proto(img_proto, text_feat):
        txt = _norm(text_feat + lam * gap)
        return _norm(beta * img_proto + (1 - beta) * txt)

    @torch.no_grad()
    def encode_imgs(imgs):
        return _norm(m.backbone.encode_image(imgs.to(dev)))

    @torch.no_grad()
    def eval_method(ds, classes, n_episodes, text_fn):
        """text_fn(cls_list)->(n,512) normalized text features, or None for image-only."""
        rng = np.random.RandomState(SEED)
        cor = tot = 0
        for _ in range(n_episodes):
            s, q, ql, cls = sample_from(ds, classes, n_way, rng)
            ip = _norm(encode_imgs(s).view(n_way, K_SHOT, -1).mean(1))
            qe = encode_imgs(q)
            proto = ip if text_fn is None else blend_proto(ip, text_fn(cls))
            pred = torch.softmax(-torch.cdist(qe, proto) * temp, 1).argmax(1).cpu()
            cor += int((pred == ql).sum()); tot += len(ql)
        return 100.0 * cor / tot

    cupl_fn = lambda cls: torch.stack([cupl_text[c] for c in cls])

    # ── Train context vectors ────────────────────────────────────────────
    rng = np.random.RandomState(SEED)
    best_val, best_ctx = -1, learner.ctx.detach().clone()
    t0 = time.time()
    print(f"\nTraining {N_CTX}-token context for {EPISODES} episodes (only ctx, {learner.ctx.numel()} params)...")
    for ep in range(1, EPISODES + 1):
        s, q, ql, cls = sample_from(train_ds, train_classes, n_way, rng)
        ip = _norm(encode_imgs(s).view(n_way, K_SHOT, -1).mean(1))
        qe = encode_imgs(q)
        text_feat = learner.text_features(cls)                       # grad -> ctx
        proto = blend_proto(ip, text_feat)
        logits = -torch.cdist(qe, proto) * temp
        ce = F.cross_entropy(logits, ql.to(dev), reduction="none")
        focal = ((1 - torch.exp(-ce)) ** 2 * ce).mean()
        # BiomedCoOp distillation: keep learned text near CuPL ensemble
        reg = (1 - F.cosine_similarity(text_feat, cupl_fn(cls), dim=-1)).mean()
        loss = focal + REG_W * reg
        opt.zero_grad(); loss.backward(); opt.step()

        if ep % VAL_INT == 0:
            learned_fn = lambda cls: learner.text_features(cls)
            v = eval_method(val_ds, train_classes, VAL_EP, learned_fn)
            tag = ""
            if v > best_val:
                best_val, best_ctx = v, learner.ctx.detach().clone(); tag = " *best"
            print(f"  ep {ep:>4}  loss={loss.item():.3f} (focal={focal.item():.3f} reg={reg.item():.3f})  val={v:.2f}%{tag}")

    learner.ctx.data.copy_(best_ctx)
    torch.save({"ctx": best_ctx.cpu(), "n_ctx": N_CTX, "mode": MODE}, CTX_OUT)
    print(f"\nBest val={best_val:.2f}%  ({(time.time()-t0)/60:.1f} min)  ctx -> {CTX_OUT}")

    # ── Final comparison on the mode's eval protocol ─────────────────────
    EV = 300
    learned_fn = lambda cls: learner.text_features(cls)
    res = {
        "mode": MODE, "n_way": n_way, "eval_classes": eval_classes, "n_ctx": N_CTX,
        "eval_episodes": EV, "best_val": round(best_val, 2),
        "image_only": round(eval_method(test_ds, eval_classes, EV, None), 2),
        "M3_cupl_gap": round(eval_method(test_ds, eval_classes, EV, cupl_fn), 2),
        "CoOp_learned": round(eval_method(test_ds, eval_classes, EV, learned_fn), 2),
    }
    res["delta_coop_vs_image"] = round(res["CoOp_learned"] - res["image_only"], 2)
    res["delta_coop_vs_M3"]    = round(res["CoOp_learned"] - res["M3_cupl_gap"], 2)
    print("\n=== TEST (%s, %d-way) ===" % (MODE, n_way))
    for k in ["image_only", "M3_cupl_gap", "CoOp_learned"]:
        print(f"  {k:<14} {res[k]:.2f}%")
    print(f"  CoOp vs image: {res['delta_coop_vs_image']:+.2f}   CoOp vs M3: {res['delta_coop_vs_M3']:+.2f}")
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
