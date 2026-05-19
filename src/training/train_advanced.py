"""
Advanced RareSight training script.
- EpisodicDermaMNIST (HAM10000) dataset
- FocalLoss (gamma=2) for hard-example focus
- 3 ViT blocks unfrozen with differential learning rates
- Val accuracy checkpointing every 500 episodes
- Training log saved to checkpoints/training_log.json
"""

import sys
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.raresight_net import RareSight
from src.data.dataset import EpisodicDermaMNIST

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_WAY = 5
K_SHOT = 5
N_QUERY = 15
EPISODES = 8000
LR_BACKBONE = 5e-6   # unfrozen ViT blocks — stay close to pretrained init
LR_FUSION = 1e-4     # fusion net + alpha
WEIGHT_DECAY = 1e-4
N_UNFREEZE = 3       # how many final ViT blocks to unfreeze
VAL_INTERVAL = 500
VAL_EPISODES = 100
CKPT_DIR = "checkpoints"
CKPT_BEST = os.path.join(CKPT_DIR, "raresight_best.pth")
CKPT_LAST = os.path.join(CKPT_DIR, "raresight_last.pth")
LOG_PATH = os.path.join(CKPT_DIR, "training_log.json")
DESC_PATH = os.path.join(os.path.dirname(__file__), "../../src/app/class_descriptions.json")


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ── Validation ────────────────────────────────────────────────────────────────
def evaluate(
    model: nn.Module,
    val_dataset: EpisodicDermaMNIST,
    descriptions: dict,
    device: str,
    n_episodes: int = VAL_EPISODES,
) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            s_img, _, q_img, q_lbl, orig_cls = val_dataset.sample_episode(
                N_WAY, K_SHOT, N_QUERY, return_class_ids=True
            )
            s_img = s_img.to(device)
            q_img = q_img.to(device)
            q_lbl = q_lbl.to(device)
            s_txt = [
                descriptions.get(str(c), "dermoscopy image of a skin lesion")
                for c in orig_cls
                for _ in range(K_SHOT)
            ]
            logits = model(s_img, s_txt, q_img, N_WAY, K_SHOT)
            preds = logits.argmax(dim=1)
            correct += (preds == q_lbl).sum().item()
            total += q_lbl.size(0)
    model.train()
    return 100.0 * correct / total if total > 0 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load text descriptions
    desc_path = os.path.abspath(DESC_PATH)
    with open(desc_path, "r") as f:
        descriptions = json.load(f)

    # Datasets
    print("\nLoading datasets...")
    train_dataset = EpisodicDermaMNIST(split="train", augment=True)
    val_dataset = EpisodicDermaMNIST(split="val", augment=False)

    # Model
    model = RareSight(device=device)

    # Unfreeze last N_UNFREEZE ViT blocks
    print(f"\nUnfreezing last {N_UNFREEZE} ViT blocks for domain adaptation...")
    try:
        blocks = model.backbone.visual.trunk.blocks
        for block in blocks[-N_UNFREEZE:]:
            for param in block.parameters():
                param.requires_grad = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_trainable:,}")
    except AttributeError as e:
        print(f"  Warning: could not access ViT blocks ({e}). Backbone stays frozen.")

    model.train()

    # Separate parameter groups for differential LR
    backbone_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and "backbone" in name
    ]
    fusion_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and "backbone" not in name
    ]

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": fusion_params, "lr": LR_FUSION},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPISODES, eta_min=1e-7
    )
    criterion = FocalLoss(gamma=2.0)

    os.makedirs(CKPT_DIR, exist_ok=True)

    best_val_acc = 0.0
    training_log = []
    start_time = time.time()

    print(f"\nStarting training: {EPISODES} episodes, {N_WAY}-way {K_SHOT}-shot\n")
    pbar = tqdm(range(1, EPISODES + 1), dynamic_ncols=True)

    for ep in pbar:
        optimizer.zero_grad()

        s_img, _, q_img, q_lbl, orig_cls = train_dataset.sample_episode(
            N_WAY, K_SHOT, N_QUERY, return_class_ids=True
        )
        s_img = s_img.to(device)
        q_img = q_img.to(device)
        q_lbl = q_lbl.to(device)

        # Build support text descriptions (K_SHOT texts per class)
        s_txt = [
            descriptions.get(str(c), "dermoscopy image of a skin lesion")
            for c in orig_cls
            for _ in range(K_SHOT)
        ]

        logits = model(s_img, s_txt, q_img, N_WAY, K_SHOT)
        loss = criterion(logits, q_lbl)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        pbar.set_description(
            f"ep={ep} loss={loss.item():.4f} "
            f"lr_bb={scheduler.get_last_lr()[0]:.1e} "
            f"best_val={best_val_acc:.1f}%"
        )

        # Validation checkpoint
        if ep % VAL_INTERVAL == 0:
            elapsed = (time.time() - start_time) / 60
            val_acc = evaluate(model, val_dataset, descriptions, device)
            print(
                f"\n[ep {ep:>5}] val_acc={val_acc:.2f}%  "
                f"loss={loss.item():.4f}  elapsed={elapsed:.1f}m"
            )

            entry = {
                "episode": ep,
                "val_acc": round(val_acc, 4),
                "train_loss": round(loss.item(), 6),
                "elapsed_min": round(elapsed, 2),
            }
            training_log.append(entry)

            with open(LOG_PATH, "w") as f:
                json.dump(training_log, f, indent=2)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), CKPT_BEST)
                print(f"  ✓ New best model saved ({best_val_acc:.2f}%)")

    # Save final checkpoint
    torch.save(model.state_dict(), CKPT_LAST)
    total_time = (time.time() - start_time) / 60
    print(f"\nTraining complete in {total_time:.1f} minutes")
    print(f"Best val accuracy : {best_val_acc:.2f}%")
    print(f"Best checkpoint   : {CKPT_BEST}")
    print(f"Training log      : {LOG_PATH}")


if __name__ == "__main__":
    main()
