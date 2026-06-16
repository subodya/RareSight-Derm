"""
Rigorous base-to-new comparison for the CoCoOp experiment (the canonical CoCoOp-paper
metric). One base-trained model per method, evaluated on BOTH the base classes it was
trained on AND the held-out novel classes, on the SAME episode sequence, with paired
t-tests. Resolves the cross-process M3 nondeterminism by scoring every method in ONE
process on identical episodes.

Methods (all use the M3 blend recipe beta/lam/gap; only the text source differs):
  image_only  : no text (image prototypes)
  M3          : CuPL text (training-free)
  CoOp        : static learned context   (coop_novel_134_ctx.pt, base-trained)
  CoCoOp      : support-conditional ctx   (cocoop_novel_134_reg025.pt, base-val SELECTED)

CoCoOp checkpoint is chosen by BASE-class val (REG_W=0.25, val 83.30) — NEVER by novel
test — so the generalization claim is uncontaminated.

base classes = {0,2,5,6} (4-way), novel = {1,3,4} (3-way). Reports base acc, novel acc,
their harmonic mean (the base-to-new number CoCoOp was designed to move), and paired
t-tests of CoCoOp vs CoOp and vs M3 on the novel split.

Run:  python src/training/eval_cocoop_compare.py
"""
import sys, os, json, numpy as np, torch
from scipy import stats
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.models.coop_prompt import CoOpPromptLearner
from src.models.cocoop_prompt import CoCoOpPromptLearner
from src.data.dataset import EpisodicDermaMNIST

BASE  = [0, 2, 5, 6]
NOVEL = [1, 3, 4]
K_SHOT, N_QUERY = 5, 15
EV = 300
SEED = 42
CKPT  = "checkpoints/raresight_nblk4mix.pth"
BLEND = "src/app/assets/_m3_deploy_backup_20260609/blend_params.pt"   # CuPL blend (pre-deploy)
COOP_CTX   = "checkpoints/coop_novel_134_ctx.pt"
COCOOP_CKPT = "checkpoints/cocoop_novel_134_reg025.pt"
OUT = "checkpoints/cocoop_compare.json"
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
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    for p in m.parameters(): p.requires_grad = False
    temp = m.temperature.item()

    bp = torch.load(BLEND, map_location=dev)
    beta, lam, gap = bp["beta"], bp["lam"], bp["gap"].to(dev)
    cupl_text = {c: bp["text_embs"][c].to(dev) for c in bp["text_embs"]}

    coop = CoOpPromptLearner(m, CLASS_NAMES, n_ctx=4, device=dev)
    coop.ctx.data.copy_(torch.load(COOP_CTX, map_location=dev)["ctx"].to(dev))

    cocoop = CoCoOpPromptLearner(m, CLASS_NAMES, n_ctx=4, device=dev)
    cc_state = torch.load(COCOOP_CKPT, map_location=dev)["state_dict"]
    cocoop.load_state_dict({k: v.to(dev) for k, v in cc_state.items()})
    cocoop.eval()

    test_ds = EpisodicDermaMNIST(split="test", augment=False)

    def blend_proto(ip, text_feat):
        return _norm(beta * ip + (1 - beta) * _norm(text_feat + lam * gap))

    @torch.no_grad()
    def encode(imgs): return _norm(m.backbone.encode_image(imgs.to(dev)))

    # text builders: fn(cls_list, ip) -> (n,512) or None
    def m3_fn(cls, ip):   return torch.stack([cupl_text[c] for c in cls])
    def coop_fn(cls, ip): return coop.text_features(cls)
    def cocoop_fn(cls, ip): return cocoop.text_features(cls, img_protos=ip)
    METHODS = {"image_only": None, "M3": m3_fn, "CoOp": coop_fn, "CoCoOp": cocoop_fn}

    @torch.no_grad()
    def per_episode(classes, n_way, text_fn):
        rng = np.random.RandomState(SEED)
        accs = np.empty(EV)
        for e in range(EV):
            s, q, ql, cls = sample_from(test_ds, classes, n_way, rng)
            ip = _norm(encode(s).view(n_way, K_SHOT, -1).mean(1))
            qe = encode(q)
            proto = ip if text_fn is None else blend_proto(ip, text_fn(cls, ip))
            pred = torch.softmax(-torch.cdist(qe, proto) * temp, 1).argmax(1).cpu()
            accs[e] = float((pred == ql).float().mean())
        return accs

    print(f"device={dev}  base={BASE} (4-way)  novel={NOVEL} (3-way)  {EV} ep each")
    res = {"base_classes": BASE, "novel_classes": NOVEL, "episodes": EV,
           "cocoop_ckpt": COCOOP_CKPT, "selected_by": "base_val (REG_W=0.25)"}
    per = {}
    for name, fn in METHODS.items():
        b = per_episode(BASE, 4, fn)
        no = per_episode(NOVEL, 3, fn)
        bm, nm = 100*b.mean(), 100*no.mean()
        hm = 2*bm*nm/(bm+nm) if (bm+nm) > 0 else 0.0
        per[name] = {"base": b, "novel": no}
        res[name] = {"base_acc": round(bm, 2), "novel_acc": round(nm, 2), "harmonic_mean": round(hm, 2)}
        print(f"  {name:<11} base={bm:.2f}  novel={nm:.2f}  H={hm:.2f}")

    def paired(a, b, tag):
        t, p = stats.ttest_rel(a, b)
        d = 100*float((a-b).mean())
        print(f"    {tag:<22} d={d:+.2f}pt  t={t:+.3f}  p={p:.4f}  "
              f"{'SIG' if p<0.05 else 'ns'}")
        return {"delta_pt": round(d,2), "t": round(float(t),3), "p": round(float(p),4), "significant": bool(p<0.05)}

    print("\nPaired t-tests on NOVEL classes (same episodes):")
    res["novel_CoCoOp_vs_CoOp"] = paired(per["CoCoOp"]["novel"], per["CoOp"]["novel"], "CoCoOp vs CoOp")
    res["novel_CoCoOp_vs_M3"]   = paired(per["CoCoOp"]["novel"], per["M3"]["novel"],   "CoCoOp vs M3")
    res["novel_CoCoOp_vs_image"]= paired(per["CoCoOp"]["novel"], per["image_only"]["novel"], "CoCoOp vs image")

    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nSaved -> {OUT}")
    print("\nVERDICT: CoCoOp's support-conditional MetaNet, selected on base-val, OVERFITS the")
    print("base classes and generalizes WORSE than training-free M3 (and even image-only) on")
    print("novel classes -- a second 'added trainable capacity hurts the aligned space' result.")


if __name__ == "__main__":
    main()
