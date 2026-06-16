"""
Deploy CoOp: rebuild the served prototypes using the LEARNED CoOp context instead of
hand-built CuPL text, keeping the SAME M3 blend (beta/lam/gap). Justified by
eval_coop_deployed_7way.py: in the deployed 7-way K=20 regime CoOp beats M3
significantly (macro-F1 47.4 vs 44.95, acc 56.96 vs 51.57, McNemar p<1e-4, +72 images).

Exactly mirrors precompute.py's image-prototype construction (K=20, same seeded
selection) and reuses the existing gap/beta/lam from blend_params.pt — so the only
change vs the deployed M3 prototypes is text source (CuPL -> CoOp). Produces CoOp
prototypes identical to the validated proto_coop in eval_coop_deployed_7way.py.

Overwrites src/app/assets/{disease_prototypes.pt, blend_params.pt}. The M3 originals are
backed up in src/app/assets/_m3_deploy_backup_20260609/. disease_metadata.json and the
reference_images/ gallery are left untouched (the support images don't change).

After this, RE-RUN build_serving_artifacts.py to refit calibration/alpha/OOD for the new
prototypes.

Run:  python src/app/tools/precompute_coop.py
"""
import sys, os, random, torch
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.models.raresight_net import RareSight
from src.models.coop_prompt import CoOpPromptLearner
from src.data.preprocessing import load_ham10000

K_SHOT = 20
SEED   = 42
CKPT   = "checkpoints/raresight_nblk4mix.pth"
HAM_DIR = "data/ham10000"
BLEND_PATH = "src/app/assets/blend_params.pt"
PROTO_PATH = "src/app/assets/disease_prototypes.pt"
CTX_IN = "checkpoints/coop_indist_ctx.pt"
CLASS_NAMES = {0:"actinic keratoses",1:"basal cell carcinoma",2:"benign keratosis",
               3:"dermatofibroma",4:"melanoma",5:"melanocytic nevi",6:"vascular lesions"}


def _norm(x): return x / x.norm(dim=-1, keepdim=True)


def _select_diverse(paths, labels, cls_id, k, seed):
    cls_paths = [p for p, l in zip(paths, labels) if int(l) == cls_id]
    rng = random.Random(seed + cls_id)
    rng.shuffle(cls_paths)
    return cls_paths[:k]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    for p in m.parameters(): p.requires_grad = False
    print(f"device={dev}")

    # Reuse the EXACT deployed blend knobs and modality gap (CoOp was trained/eval'd with these).
    bp = torch.load(BLEND_PATH, map_location=dev)
    beta, lam, gap = bp["beta"], bp["lam"], bp["gap"].to(dev)
    print(f"reusing blend: beta={beta} lam={lam} (gap from deployed blend_params)")

    # Learned CoOp text per class (context trained on all 7 classes).
    learner = CoOpPromptLearner(m, CLASS_NAMES, n_ctx=4, device=dev)
    learner.ctx.data.copy_(torch.load(CTX_IN, map_location=dev)["ctx"].to(dev))
    classes = list(range(7))
    with torch.no_grad():
        coop_text = {c: learner.text_features([c])[0].cpu() for c in classes}
    print(f"built CoOp text embeddings for {len(classes)} classes from {CTX_IN}")

    # K=20 image prototypes (identical selection to precompute.py).
    tr_paths, tr_labels = load_ham10000(HAM_DIR, split="train", val_size=0.10, test_size=0.10, seed=SEED)

    @torch.no_grad()
    def img_prototype(paths, bs=64):
        embs = []
        for i in range(0, len(paths), bs):
            batch = torch.cat([m.preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                               for p in paths[i:i+bs]]).to(dev)
            embs.append(_norm(m.backbone.encode_image(batch)))
        return _norm(torch.cat(embs).mean(0)).cpu()

    img_proto = {}
    for c in classes:
        sup = _select_diverse(tr_paths, tr_labels, c, K_SHOT, SEED)
        img_proto[c] = img_prototype(sup)
        print(f"  class {c}: image prototype from {len(sup)} support images")

    # CoOp prototypes via the M3 blend (text = CoOp instead of CuPL).
    gap_c = gap.cpu()
    prototypes = {}
    for c in classes:
        txt_shift = _norm(coop_text[c] + lam * gap_c)
        prototypes[c] = _norm(beta * img_proto[c] + (1.0 - beta) * txt_shift)

    # Sanity: compare to deployed M3 prototypes (should DIFFER — that's the point).
    old = torch.load(PROTO_PATH, map_location="cpu")
    diff = max((prototypes[c] - _norm(old[c])).abs().max().item() for c in classes)
    print(f"\nmax|CoOp - deployed M3| prototype diff = {diff:.4f}  (expected > 0)")

    torch.save(prototypes, PROTO_PATH)
    torch.save({"gap": gap.cpu(), "beta": beta, "lam": lam, "text_embs": coop_text}, BLEND_PATH)
    print(f"WROTE CoOp prototypes -> {PROTO_PATH}")
    print(f"WROTE CoOp blend (text_embs=CoOp) -> {BLEND_PATH}")
    print("NEXT: re-run src/app/tools/build_serving_artifacts.py to refit calibration/alpha/OOD.")


if __name__ == "__main__":
    main()
