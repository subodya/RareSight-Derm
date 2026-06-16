"""
Does CoOp actually help in the DEPLOYED regime?  (the missing measurement)

The 64.4% CoOp headline is 5-way 5-shot EPISODIC. The deployed app (precompute.py)
builds 7-way prototypes from K=20 support images per class. The shot curve shows CoOp's
edge over image-only shrinks with K (+20.1@K1 -> +6.9@K5 -> +2.8@K10), so at K=20 the
gain may wash out. This script measures CoOp vs M3 vs image-only in the EXACT deployed
setting: K=20 prototypes (same selection as precompute), full HAM test set, 7-way.

Reports overall accuracy + macro-F1 (classes are imbalanced) and a McNemar paired test
on per-image correctness (CoOp vs M3). Read-only: rebuilds prototypes in memory, writes
no deployed artifact.

Run:  python src/training/eval_coop_deployed_7way.py
"""
import sys, os, json, random, numpy as np, torch
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import binomtest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.models.coop_prompt import CoOpPromptLearner
from src.data.preprocessing import load_ham10000

K_SHOT = 20
SEED   = 42
CKPT   = "checkpoints/raresight_nblk4mix.pth"
# Pre-deploy M3 blend (CuPL text). The live blend_params.pt now holds CoOp text after
# deployment, so use the backup to keep M3 the genuine training-free baseline.
BLEND  = "src/app/assets/_m3_deploy_backup_20260609/blend_params.pt"
PROTO_DEPLOYED_M3 = "src/app/assets/_m3_deploy_backup_20260609/disease_prototypes.pt"
CTX_IN = "checkpoints/coop_indist_ctx.pt"        # context trained on all 7 classes
HAM_DIR = "data/ham10000"
OUT    = "checkpoints/coop_deployed_7way.json"
CLASS_NAMES = {0:"actinic keratoses",1:"basal cell carcinoma",2:"benign keratosis",
               3:"dermatofibroma",4:"melanoma",5:"melanocytic nevi",6:"vascular lesions"}


def _norm(x): return x / x.norm(dim=-1, keepdim=True)


def _select_diverse(paths, labels, cls_id, k, seed):
    """Exactly mirror precompute.py so prototypes match the deployed ones."""
    cls_paths = [p for p, l in zip(paths, labels) if int(l) == cls_id]
    rng = random.Random(seed + cls_id)
    rng.shuffle(cls_paths)
    return cls_paths[:k]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    for p in m.parameters(): p.requires_grad = False
    temp = m.temperature.item()
    print(f"device={dev}  temp={temp:.3f}  K_SHOT={K_SHOT}")

    bp = torch.load(BLEND, map_location=dev)
    beta, lam, gap = bp["beta"], bp["lam"], bp["gap"].to(dev)
    cupl_text = {c: bp["text_embs"][c].to(dev) for c in bp["text_embs"]}

    learner = CoOpPromptLearner(m, CLASS_NAMES, n_ctx=4, device=dev)
    learner.ctx.data.copy_(torch.load(CTX_IN, map_location=dev)["ctx"].to(dev))
    classes = list(range(7))
    with torch.no_grad():
        coop_text = {c: learner.text_features([c])[0] for c in classes}   # (512,) each

    @torch.no_grad()
    def encode_paths(paths, bs=64):
        embs = []
        for i in range(0, len(paths), bs):
            batch = torch.cat([m.preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                               for p in paths[i:i+bs]]).to(dev)
            embs.append(_norm(m.backbone.encode_image(batch)))
        return torch.cat(embs)

    # ── Build K=20 image prototypes per class (match precompute selection) ──
    tr_paths, tr_labels = load_ham10000(HAM_DIR, split="train", val_size=0.10, test_size=0.10, seed=SEED)
    img_proto = {}
    for c in classes:
        sup = _select_diverse(tr_paths, tr_labels, c, K_SHOT, SEED)
        img_proto[c] = _norm(encode_paths(sup).mean(0))
    print("built K=20 image prototypes for 7 classes")

    def blend(textf):
        return _norm(beta * img_proto[c] + (1 - beta) * _norm(textf + lam * gap))

    proto_img  = torch.stack([img_proto[c] for c in classes])
    proto_m3   = torch.stack([_norm(beta*img_proto[c] + (1-beta)*_norm(cupl_text[c] + lam*gap)) for c in classes])
    proto_coop = torch.stack([_norm(beta*img_proto[c] + (1-beta)*_norm(coop_text[c] + lam*gap)) for c in classes])

    # sanity: rebuilt M3 prototypes should match the (backed-up) M3 disease_prototypes.pt
    dep = torch.load(PROTO_DEPLOYED_M3, map_location=dev)
    dep_stack = torch.stack([_norm(dep[c].to(dev)) for c in classes])
    md = (proto_m3 - dep_stack).abs().max().item()
    print(f"sanity: rebuilt-M3 vs deployed disease_prototypes.pt max|diff| = {md:.2e} "
          f"({'MATCH' if md < 1e-3 else 'DIFFERS'})")

    # ── Evaluate on full HAM test set, 7-way ─────────────────────────────
    te_paths, te_labels = load_ham10000(HAM_DIR, split="test", val_size=0.10, test_size=0.10, seed=SEED)
    qe = encode_paths(te_paths)
    y = np.array(te_labels)
    print(f"test images: {len(y)}")

    @torch.no_grad()
    def predict(proto):
        return torch.softmax(-torch.cdist(qe, proto) * temp, 1).argmax(1).cpu().numpy()

    pred_img, pred_m3, pred_coop = predict(proto_img), predict(proto_m3), predict(proto_coop)

    CN = {0:"akiec",1:"bcc",2:"bkl",3:"df",4:"mel",5:"nv",6:"vasc"}
    def scores(name, pred):
        acc = 100 * accuracy_score(y, pred)
        mf1 = 100 * f1_score(y, pred, average="macro")
        pc = 100 * f1_score(y, pred, average=None, labels=list(range(7)), zero_division=0)
        print(f"  {name:<12} acc={acc:.2f}%  macroF1={mf1:.2f}")
        return {"acc": round(acc, 2), "macro_f1": round(mf1, 2),
                "per_class_f1": {CN[c]: round(float(pc[c]), 2) for c in range(7)}}

    print("\n=== DEPLOYED 7-way, K=20, full HAM test ===")
    res = {"protocol": "7way_K20_hamtest", "n_test": int(len(y)), "K_shot": K_SHOT,
           "image_only": scores("image-only", pred_img),
           "M3": scores("M3", pred_m3),
           "CoOp": scores("CoOp", pred_coop)}

    # Per-class F1: does CoOp's gain concentrate in the RARE classes? (test n: df=15, vasc=19, akiec=44)
    print("\n  per-class F1 (rare classes first):")
    order = [3, 6, 0, 1, 2, 4, 5]   # df, vasc, akiec, bcc, bkl, mel, nv
    print(f"    {'class':<8}{'image':>8}{'M3':>8}{'CoOp':>8}{'CoOp-M3':>9}")
    for c in order:
        ci, cm, cc = (res['image_only']['per_class_f1'][CN[c]],
                      res['M3']['per_class_f1'][CN[c]], res['CoOp']['per_class_f1'][CN[c]])
        print(f"    {CN[c]:<8}{ci:>8.1f}{cm:>8.1f}{cc:>8.1f}{cc-cm:>+9.1f}")

    # ── McNemar paired test: CoOp vs M3 on per-image correctness ──────────
    cc = (pred_coop == y); mm = (pred_m3 == y)
    b = int(np.sum(cc & ~mm))   # CoOp right, M3 wrong
    c_ = int(np.sum(~cc & mm))  # CoOp wrong, M3 right
    p = binomtest(min(b, c_), b + c_, 0.5).pvalue if (b + c_) > 0 else 1.0
    res["mcnemar_coop_vs_m3"] = {"coop_right_m3_wrong": b, "coop_wrong_m3_right": c_,
                                 "p": round(float(p), 4), "significant": bool(p < 0.05),
                                 "net_images": b - c_}
    print(f"\n  McNemar CoOp vs M3: b(CoOp✓M3✗)={b}  c(CoOp✗M3✓)={c_}  "
          f"net={b-c_:+d} images  p={p:.4f}  {'SIGNIFICANT' if p < 0.05 else 'ns (noise)'}")
    res["delta_coop_vs_m3_acc"] = round(res["CoOp"]["acc"] - res["M3"]["acc"], 2)
    res["delta_coop_vs_m3_mf1"] = round(res["CoOp"]["macro_f1"] - res["M3"]["macro_f1"], 2)

    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nSaved -> {OUT}")
    print("\nDECISION RULE:")
    print("  If CoOp's macro-F1 gain over M3 here is small / McNemar ns -> the 64.4% is a")
    print("  low-shot artifact; keep M3 deployed, report CoOp as the episodic capstone.")
    print("  If CoOp wins significantly at K=20 -> deploying CoOp (+ calibration refit) is justified.")


if __name__ == "__main__":
    main()
