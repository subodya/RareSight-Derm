"""
Fusion-only fine-tuning from best checkpoint.

Loads raresight_best.pth (ep-1500, 61.21%), freezes the entire backbone,
and trains only the fusion_net + alpha + temperature for 2000 episodes.
With the backbone frozen there is no catastrophic forgetting — accuracy
should climb steadily from ~61% toward 65-70%.

Usage:
    python src/training/finetune.py

Output:
    checkpoints/raresight_finetuned.pth  — best val-accuracy checkpoint
    checkpoints/finetune_log.json        — per-checkpoint metrics
"""

import sys
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.raresight_net import RareSight
from src.data.dataset import EpisodicDermaMNIST

# ── Config ────────────────────────────────────────────────────────────────────
N_WAY        = 5
K_SHOT       = 5
N_QUERY      = 15
EPISODES     = 2000
LR_FUSION    = 5e-4    # higher — backbone is frozen so we can be aggressive
WEIGHT_DECAY = 1e-4
VAL_INTERVAL = 200
VAL_EPISODES = 100
CKPT_DIR     = "checkpoints"
CKPT_SRC     = os.path.join(CKPT_DIR, "raresight_best.pth")
CKPT_OUT     = os.path.join(CKPT_DIR, "raresight_finetuned.pth")
LOG_PATH     = os.path.join(CKPT_DIR, "finetune_log.json")
DESC_PATH    = os.path.join(os.path.dirname(__file__), "../../src/app/class_descriptions.json")


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ── Validation ────────────────────────────────────────────────────────────────
def evaluate(model, val_dataset, descriptions, device, n_episodes=VAL_EPISODES):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            s_img, _, q_img, q_lbl, orig_cls = val_dataset.sample_episode(
                N_WAY, K_SHOT, N_QUERY, return_class_ids=True
            )
            s_img  = s_img.to(device)
            q_img  = q_img.to(device)
            q_lbl  = q_lbl.to(device)
            s_txt  = [
                descriptions.get(str(c), "dermoscopy image of a skin lesion")
                for c in orig_cls for _ in range(K_SHOT)
            ]
            logits = model(s_img, s_txt, q_img, N_WAY, K_SHOT)
            correct += (logits.argmax(1) == q_lbl).sum().item()
            total   += q_lbl.size(0)
    model.train()
    return 100.0 * correct / total if total > 0 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load descriptions
    with open(os.path.abspath(DESC_PATH)) as f:
        descriptions = json.load(f)

    # Datasets
    print("\nLoading datasets...")
    train_dataset = EpisodicDermaMNIST(split="train", augment=True)
    val_dataset   = EpisodicDermaMNIST(split="val",   augment=False)

    # Model — load best checkpoint with strict=False so the new
    # `temperature` parameter (not in the old checkpoint) is kept at
    # its fresh initialisation value of 10.0
    model = RareSight(device=device)
    if not os.path.exists(CKPT_SRC):
        raise FileNotFoundError(
            f"Source checkpoint not found: {CKPT_SRC}\n"
            "Run train_advanced.py first."
        )
    state = torch.load(CKPT_SRC, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"\nLoaded checkpoint: {CKPT_SRC}")
    if missing:
        print(f"  New params (will be trained from init): {missing}")
    if unexpected:
        print(f"  Ignored keys: {unexpected}")

    # Freeze backbone completely — only fusion_net + alpha + temperature train
    for param in model.backbone.parameters():
        param.requires_grad = False

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable modules: {trainable}")
    print(f"Trainable parameters: {n_trainable:,}")

    model.train()

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR_FUSION,
        weight_decay=WEIGHT_DECAY,
    )
    # Warm restarts: each restart gets a chance to escape local minima
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=500, T_mult=1, eta_min=1e-6
    )
    criterion = FocalLoss(gamma=2.0)

    os.makedirs(CKPT_DIR, exist_ok=True)

    # Run initial validation to confirm loaded checkpoint accuracy
    print("\nBaseline validation (loaded checkpoint)...")
    baseline = evaluate(model, val_dataset, descriptions, device)
    print(f"  Baseline val accuracy: {baseline:.2f}%")

    best_val_acc = baseline
    training_log = [{"episode": 0, "val_acc": round(baseline, 4), "note": "baseline"}]
    start_time = time.time()

    print(f"\nFine-tuning: {EPISODES} episodes (backbone frozen)\n")
    pbar = tqdm(range(1, EPISODES + 1), dynamic_ncols=True)

    for ep in pbar:
        optimizer.zero_grad()

        s_img, _, q_img, q_lbl, orig_cls = train_dataset.sample_episode(
            N_WAY, K_SHOT, N_QUERY, return_class_ids=True
        )
        s_img = s_img.to(device)
        q_img = q_img.to(device)
        q_lbl = q_lbl.to(device)
        s_txt = [
            descriptions.get(str(c), "dermoscopy image of a skin lesion")
            for c in orig_cls for _ in range(K_SHOT)
        ]

        logits = model(s_img, s_txt, q_img, N_WAY, K_SHOT)
        loss   = criterion(logits, q_lbl)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        pbar.set_description(
            f"ep={ep} loss={loss.item():.4f} "
            f"T={model.temperature.item():.2f} "
            f"best={best_val_acc:.1f}%"
        )

        if ep % VAL_INTERVAL == 0:
            elapsed = (time.time() - start_time) / 60
            val_acc = evaluate(model, val_dataset, descriptions, device)
            print(
                f"\n[ep {ep:>4}] val_acc={val_acc:.2f}%  "
                f"loss={loss.item():.4f}  "
                f"T={model.temperature.item():.2f}  "
                f"elapsed={elapsed:.1f}m"
            )
            training_log.append({
                "episode":    ep,
                "val_acc":    round(val_acc, 4),
                "train_loss": round(loss.item(), 6),
                "temperature": round(model.temperature.item(), 4),
                "elapsed_min": round(elapsed, 2),
            })
            with open(LOG_PATH, "w") as f:
                json.dump(training_log, f, indent=2)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), CKPT_OUT)
                print(f"  ✓ New best saved → {CKPT_OUT}  ({best_val_acc:.2f}%)")

    total_time = (time.time() - start_time) / 60
    print(f"\nFine-tuning complete in {total_time:.1f} minutes")
    print(f"Best val accuracy : {best_val_acc:.2f}%")
    print(f"Best checkpoint   : {CKPT_OUT}")
    print(f"Fine-tune log     : {LOG_PATH}")

    # If fine-tuning never beat the baseline, keep the original best
    if not os.path.exists(CKPT_OUT):
        import shutil
        shutil.copy(CKPT_SRC, CKPT_OUT)
        print(f"\nFine-tune did not improve on baseline. Copied {CKPT_SRC} → {CKPT_OUT}")


if __name__ == "__main__":
    main()
