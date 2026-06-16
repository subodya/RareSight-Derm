"""
Train CoCoOp-inspired support-conditional prompts (see src/models/cocoop_prompt.py).

Same harness/discipline as train_coop.py:
  * Trainable = ctx (M x 768) + Meta-Net only; BiomedCLIP backbone frozen.
  * blend beta/lam/gap FIXED at M3 values (isolate "better prompt" from "better blend").
  * BiomedCoOp CuPL distillation regulariser (REG_W) keeps learned text near the CuPL
    ensemble. For CoCoOp this is a KNIFE-EDGE: too high suppresses the Meta-Net shift to
    ~0 (= CoOp in disguise); too low lets text drift toward the image prototype. So
    REG_W is swept, model selection is on BASE-class val ONLY, and we LOG the mean
    Meta-Net shift magnitude as the sanity check that conditioning is non-trivial.
  * Writes NEW artifacts only (cocoop_*). Deployed disease_prototypes.pt / blend_params.pt
    are never touched (M3 stays the deployed safety net).

Run (env python):
  COOP_MODE=indist COOP_REG=0.5 python src/training/train_cocoop.py
  COOP_MODE=novel  COOP_REG=0.5 COOP_NOVEL=1,3,4 python src/training/train_cocoop.py
"""
import sys, os, json, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.models.cocoop_prompt import CoCoOpPromptLearner
from src.data.dataset import EpisodicDermaMNIST

MODE     = os.environ.get("COOP_MODE", "novel")
N_CTX    = int(os.environ.get("COOP_NCTX", "4"))
EPISODES = int(os.environ.get("COOP_EPISODES", "1500"))
LR       = 2e-3
REG_W    = float(os.environ.get("COOP_REG", "0.5"))
K_SHOT, N_QUERY = 5, 15
VAL_INT, VAL_EP = 150, 60
SEED = 42
CKPT  = "checkpoints/raresight_nblk4mix.pth"
BLEND = "src/app/assets/blend_params.pt"
CLASS_NAMES = {0:"actinic keratoses",1:"basal cell carcinoma",2:"benign keratosis",
               3:"dermatofibroma",4:"melanoma",5:"melanocytic nevi",6:"vascular lesions"}
NOVEL = [int(c) for c in os.environ.get("COOP_NOVEL", "1,3,4").split(",")]
BASE  = [c for c in range(7) if c not in NOVEL]
_RTAG = f"reg{REG_W}".replace(".", "")
_TAG  = (MODE if MODE != "novel" else "novel_" + "".join(str(c) for c in NOVEL)) + f"_{_RTAG}"
OUT     = f"checkpoints/cocoop_{_TAG}_results.json"
CTX_OUT = f"checkpoints/cocoop_{_TAG}.pt"


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
    n_way = len(NOVEL) if MODE == "novel" else 5
    train_classes = BASE if MODE == "novel" else list(range(7))
    eval_classes  = NOVEL if MODE == "novel" else list(range(7))
    print(f"MODE={MODE} REG_W={REG_W} n_way={n_way} train={train_classes} eval={eval_classes}")

    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    for p in m.parameters(): p.requires_grad = False
    temp = m.temperature.item()

    bp = torch.load(BLEND, map_location=dev)
    beta, lam, gap = bp["beta"], bp["lam"], bp["gap"].to(dev)
    cupl_text = {c: bp["text_embs"][c].to(dev) for c in bp["text_embs"]}

    learner = CoCoOpPromptLearner(m, CLASS_NAMES, n_ctx=N_CTX, device=dev)
    assert learner.verify_forward() < 1e-3, "forward path does not match encode_text!"
    opt = torch.optim.AdamW(learner.trainable_params(), lr=LR, weight_decay=1e-4)

    train_ds = EpisodicDermaMNIST(split="train", augment=True)
    val_ds   = EpisodicDermaMNIST(split="val",   augment=False)
    test_ds  = EpisodicDermaMNIST(split="test",  augment=False)

    def blend_proto(img_proto, text_feat):
        txt = _norm(text_feat + lam * gap)
        return _norm(beta * img_proto + (1 - beta) * txt)

    @torch.no_grad()
    def encode_imgs(imgs):
        return _norm(m.backbone.encode_image(imgs.to(dev)))

    cupl_fn = lambda cls, ip: torch.stack([cupl_text[c] for c in cls])

    @torch.no_grad()
    def eval_method(ds, classes, n_episodes, text_fn):
        """text_fn(cls, ip)->(n,512) text feats, or None for image-only."""
        rng = np.random.RandomState(SEED)
        cor = tot = 0
        for _ in range(n_episodes):
            s, q, ql, cls = sample_from(ds, classes, n_way, rng)
            ip = _norm(encode_imgs(s).view(n_way, K_SHOT, -1).mean(1))
            qe = encode_imgs(q)
            proto = ip if text_fn is None else blend_proto(ip, text_fn(cls, ip))
            pred = torch.softmax(-torch.cdist(qe, proto) * temp, 1).argmax(1).cpu()
            cor += int((pred == ql).sum()); tot += len(ql)
        return 100.0 * cor / tot

    # ── Train ctx + Meta-Net ─────────────────────────────────────────────
    rng = np.random.RandomState(SEED)
    best_val = -1
    best_state = {k: v.detach().clone() for k, v in learner.state_dict().items()}
    t0 = time.time()
    n_train = sum(p.numel() for p in learner.trainable_params())
    print(f"\nTraining ctx+MetaNet for {EPISODES} ep ({n_train} trainable params)...")
    shift_running = 0.0
    for ep in range(1, EPISODES + 1):
        s, q, ql, cls = sample_from(train_ds, train_classes, n_way, rng)
        ip = _norm(encode_imgs(s).view(n_way, K_SHOT, -1).mean(1))
        qe = encode_imgs(q)
        text_feat = learner.text_features(cls, img_protos=ip)        # grad -> ctx + meta_net
        proto = blend_proto(ip, text_feat)
        logits = -torch.cdist(qe, proto) * temp
        ce = F.cross_entropy(logits, ql.to(dev), reduction="none")
        focal = ((1 - torch.exp(-ce)) ** 2 * ce).mean()
        reg = (1 - F.cosine_similarity(text_feat, cupl_fn(cls, ip), dim=-1)).mean()
        loss = focal + REG_W * reg
        opt.zero_grad(); loss.backward(); opt.step()
        shift_running += learner._last_shift_mag

        if ep % VAL_INT == 0:
            v = eval_method(val_ds, train_classes, VAL_EP,
                            lambda c, ip: learner.text_features(c, img_protos=ip))
            avg_shift = shift_running / VAL_INT; shift_running = 0.0
            tag = ""
            if v > best_val:
                best_val = v
                best_state = {k: vv.detach().clone() for k, vv in learner.state_dict().items()}
                tag = " *best"
            print(f"  ep {ep:>4} loss={loss.item():.3f} (focal={focal.item():.3f} "
                  f"reg={reg.item():.3f}) |shift|={avg_shift:.4f} val={v:.2f}%{tag}")

    learner.load_state_dict(best_state)
    torch.save({"state_dict": {k: v.cpu() for k, v in best_state.items()},
                "n_ctx": N_CTX, "mode": MODE, "reg_w": REG_W, "novel": NOVEL},
               CTX_OUT)
    print(f"\nBest base-val={best_val:.2f}%  ({(time.time()-t0)/60:.1f} min)  -> {CTX_OUT}")

    # ── Quick self-eval on the mode's protocol (rigorous compare is separate) ──
    EV = int(os.environ.get("COOP_EVAL", "300"))
    cocoop_fn = lambda c, ip: learner.text_features(c, img_protos=ip)
    res = {"mode": MODE, "reg_w": REG_W, "n_way": n_way, "eval_classes": eval_classes,
           "n_ctx": N_CTX, "best_val": round(best_val, 2),
           "image_only": round(eval_method(test_ds, eval_classes, EV, None), 2),
           "M3":         round(eval_method(test_ds, eval_classes, EV, cupl_fn), 2),
           "CoCoOp":     round(eval_method(test_ds, eval_classes, EV, cocoop_fn), 2)}
    res["delta_cocoop_vs_M3"] = round(res["CoCoOp"] - res["M3"], 2)
    print(f"\n=== TEST ({MODE}, {n_way}-way, REG_W={REG_W}) ===")
    for k in ["image_only", "M3", "CoCoOp"]:
        print(f"  {k:<12} {res[k]:.2f}%")
    print(f"  CoCoOp vs M3: {res['delta_cocoop_vs_M3']:+.2f}")
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
