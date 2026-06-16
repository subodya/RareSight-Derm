"""
RQ2 — Multi-modal ablation: do image+text prototypes beat image-only prototypes?

For each fixed episode we build the class prototypes three ways and classify the
SAME query images against each, so any accuracy difference is attributable only
to how the prototype was formed:

  (A) multimodal   : fused image+text prototype  (model.encode_multimodal)  ← current
  (B) image_only   : mean of raw image embeddings (no text, no fusion)       ← RQ2 baseline
  (C) fused_neutral: fusion applied but with a neutral/empty text             ← isolates
                     (disentangles "text content" from "the fusion MLP")        text content

Queries are always encoded image-only (matches forward() and the deployed app).

Reports overall accuracy + per-class accuracy (with special attention to the
rare classes 3=Dermatofibroma and 6=Vascular lesions) and saves JSON.

Usage:
    conda run -n raresight python src/training/eval_ablation_multimodal.py
"""

import sys
import os
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.raresight_net import RareSight
from src.data.dataset import EpisodicDermaMNIST

# ── Config ────────────────────────────────────────────────────────────────────
N_WAY      = 5
K_SHOT     = 5
N_QUERY    = 15
N_EPISODES = int(os.environ.get("ABLATION_EPISODES", "300"))   # fixed-seed; raise for tighter CIs
SEED       = 42
N_CLASSES  = 7
CKPT_PATH  = "checkpoints/raresight_nblk4mix.pth"
OUT_PATH   = "checkpoints/ablation_multimodal.json"
DESC_PATH  = os.path.join(os.path.dirname(__file__), "../../src/app/class_descriptions.json")
NEUTRAL_TEXT = "dermoscopy image of a skin lesion"

CLASS_NAMES = {
    0: "Actinic keratoses", 1: "Basal cell carcinoma", 2: "Benign keratosis",
    3: "Dermatofibroma", 4: "Melanoma", 5: "Melanocytic nevi", 6: "Vascular lesions",
}
RARE = {3, 6}

VARIANTS = ["multimodal", "image_only", "fused_neutral"]


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    print(f"Checkpoint : {CKPT_PATH}")
    print(f"Episodes   : {N_EPISODES} x {N_WAY}-way {K_SHOT}-shot (seed={SEED})\n")

    with open(os.path.abspath(DESC_PATH)) as f:
        descriptions = json.load(f)

    test_dataset = EpisodicDermaMNIST(split="test", augment=False)

    model = RareSight(device=device)
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(CKPT_PATH)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device), strict=False)
    model.eval()
    temp = model.temperature.item()
    print(f"Loaded. alpha={model.alpha.item():.4f}  temperature={temp:.4f}\n")

    # Per-variant accumulators
    correct = {v: {c: 0 for c in range(N_CLASSES)} for v in VARIANTS}
    total   = {c: 0 for c in range(N_CLASSES)}
    overall_correct = {v: 0 for v in VARIANTS}
    overall_total   = 0

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    start = time.time()
    with torch.no_grad():
        for ep in range(N_EPISODES):
            s_img, _, q_img, q_lbl_local, orig_cls = test_dataset.sample_episode(
                N_WAY, K_SHOT, N_QUERY, return_class_ids=True
            )
            s_img = s_img.to(device)
            q_img = q_img.to(device)
            q_lbl_local = q_lbl_local.cpu().numpy()
            orig_cls_arr = np.array(orig_cls)

            # Texts (real descriptions, K per class) and neutral texts
            s_txt = [descriptions.get(str(c), NEUTRAL_TEXT)
                     for c in orig_cls for _ in range(K_SHOT)]
            s_txt_neutral = [NEUTRAL_TEXT] * (N_WAY * K_SHOT)

            # --- Query embeddings (shared across all variants) ---
            q_emb = _norm(model.backbone.encode_image(q_img))

            # --- (A) multimodal prototypes ---
            fused = model.encode_multimodal(s_img, s_txt)
            proto_mm = _norm(fused.view(N_WAY, K_SHOT, -1).mean(dim=1))

            # --- (B) image-only prototypes ---
            s_img_emb = _norm(model.backbone.encode_image(s_img))
            proto_img = _norm(s_img_emb.view(N_WAY, K_SHOT, -1).mean(dim=1))

            # --- (C) fused with neutral text ---
            fused_n = model.encode_multimodal(s_img, s_txt_neutral)
            proto_neutral = _norm(fused_n.view(N_WAY, K_SHOT, -1).mean(dim=1))

            protos = {
                "multimodal":    proto_mm,
                "image_only":    proto_img,
                "fused_neutral": proto_neutral,
            }

            labels_global = orig_cls_arr[q_lbl_local]
            for c in labels_global:
                total[int(c)] += 1
            overall_total += len(labels_global)

            for v in VARIANTS:
                dists = torch.cdist(q_emb, protos[v])
                preds_local = torch.softmax(-dists * temp, dim=1).argmax(dim=1).cpu().numpy()
                preds_global = orig_cls_arr[preds_local]
                for pred, true in zip(preds_global, labels_global):
                    if pred == true:
                        correct[v][int(true)] += 1
                        overall_correct[v] += 1

            if (ep + 1) % 50 == 0:
                print(f"  ...{ep+1}/{N_EPISODES} episodes")

    elapsed = (time.time() - start) / 60

    # ── Report ────────────────────────────────────────────────────────────────
    sep = "─" * 78
    print(f"\n{sep}")
    print("  RQ2 — Multi-modal Ablation")
    print(f"  {N_EPISODES} episodes x {N_WAY}-way {K_SHOT}-shot   ({elapsed:.1f} min)")
    print(sep)
    print(f"  {'Variant':<16}{'Overall':>10}")
    overall = {}
    for v in VARIANTS:
        acc = 100.0 * overall_correct[v] / overall_total
        overall[v] = round(acc, 2)
        print(f"  {v:<16}{acc:>9.2f}%")
    print(f"\n  Δ multimodal − image_only : {overall['multimodal']-overall['image_only']:+.2f} pts")
    print(f"  Δ multimodal − fused_neutral (text content): "
          f"{overall['multimodal']-overall['fused_neutral']:+.2f} pts")

    print(f"\n{sep}")
    print(f"  Per-class accuracy (rare classes marked *)")
    print(f"  {'Class':<26}{'img_only':>10}{'fused_neu':>11}{'multimodal':>12}")
    per_class = {}
    for c in range(N_CLASSES):
        if total[c] == 0:
            continue
        row = {v: round(100.0 * correct[v][c] / total[c], 2) for v in VARIANTS}
        per_class[c] = row
        mark = "*" if c in RARE else " "
        print(f"  {mark}{CLASS_NAMES[c]:<25}"
              f"{row['image_only']:>9.1f}%{row['fused_neutral']:>10.1f}%{row['multimodal']:>11.1f}%")
    print(sep)

    results = {
        "checkpoint": CKPT_PATH,
        "n_episodes": N_EPISODES, "n_way": N_WAY, "k_shot": K_SHOT, "seed": SEED,
        "overall": overall,
        "delta_mm_minus_imageonly": round(overall["multimodal"] - overall["image_only"], 2),
        "delta_mm_minus_neutral":   round(overall["multimodal"] - overall["fused_neutral"], 2),
        "per_class": {str(c): per_class[c] for c in per_class},
        "elapsed_min": round(elapsed, 2),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
