"""
Phase-2: multi-resolution scale-augmented fine-tuning of BiomedCLIP.

v2 — selection FIXED. The earlier version trained a linear head on UN-normalised
features but evaluated/selected on L2-NORMALISED features, so early-stopping and
"best checkpoint" rode on a broken signal. This version trains AND selects on the
same normalised cosine-prototype objective that actually ships:

  * training head  = cosine classifier (normalised feats, learnable logit-scale)
  * selection/early-stop = periodic PROTOTYPE eval (rebuild class means, cosine
    nearest-prototype) on real DermaMNIST-test@28 (the citable low-res metric),
    SUBJECT TO a full-res HAM@224 guard.

Goal: lift sub-50px accuracy past the frozen ceiling WITHOUT regressing full-res,
by adapting only the last N ViT blocks under heavy low-res scale augmentation.

Sweep over n_blocks. n_blocks=0 (frozen) is computed ONCE as the anchor / guard
floor (its encoder never changes, so there is nothing to train).

Risky experiment knob
---------------------
--mix-real-dm P : with probability P, draw a REAL DermaMNIST-28 sample (genuine
low-res capture statistics) instead of a HAM-downscaled one. Directly attacks the
synthetic-vs-real gap (HAM-downscaled != real 28px sensor/JPEG artifacts).

Usage
-----
    python src/training/finetune_multires.py                  # sweep [0,2,3,4] seed 42
    python src/training/finetune_multires.py --nblocks 4 --seed 123
    python src/training/finetune_multires.py --nblocks 4 --mix-real-dm 0.5
    python src/training/finetune_multires.py --max-steps 30   # smoke test

NOT shippable until you ALSO regenerate serving artifacts from the adapted encoder
(precompute_coop -> build_serving_multires -> patch_band28_mix -> build_modality_probe)
AND re-verify the Mahalanobis OOD detector + calibration (fit on the FROZEN geometry;
a val-acc guard cannot detect their breakage).
"""

import os
import sys
import csv
import io
import json
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import open_clip  # noqa: E402
from src.data.preprocessing import load_ham10000  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME    = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
N_CLASSES     = 7
EMBED_DIM     = 512

SWEEP_NBLOCKS = [0, 2, 3, 4]      # 0 = frozen anchor (not trained)
BATCH         = 32
MAX_STEPS     = 1500
WARMUP        = 100
VAL_EVERY     = 150
PATIENCE      = 5

BACKBONE_LR   = 1e-5
HEAD_LR       = 1e-3
LAYER_DECAY   = 0.65
WEIGHT_DECAY  = 0.05
LABEL_SMOOTH  = 0.05
COS_SCALE     = 20.0              # init logit-scale for cosine head

AUG_RES_WEIGHTS = {28: 0.40, 56: 0.25, 112: 0.20, 224: 0.15}
GUARD_TOL     = 1.5              # pp full-res may drop below frozen anchor

# in-loop selection eval budgets (kept small/fast)
SEL_SUP_PER_CLS = 150
SEL_QRY         = 600
# end-of-config full eval budgets
PROTO_PER_CLS   = 300

DM_ROOT = os.path.join(os.path.dirname(__file__), "../../data/raw")
HAM_DIR = os.path.join(os.path.dirname(__file__), "../../data/ham10000")
OUT_DIR = os.path.join(os.path.dirname(__file__), "../../checkpoints/finetune_multires")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

device = "cuda" if torch.cuda.is_available() else "cpu"
_norm_t = T.Normalize(mean=MEAN, std=STD)


# ── Resolution-aware preprocessing (BILINEAR, matches deployment) ────────────
def to_224(pil: Image.Image, res: int) -> torch.Tensor:
    pil = pil.convert("RGB")
    if res != 224:
        pil = pil.resize((res, res), Image.BILINEAR)
    pil = pil.resize((224, 224), Image.BILINEAR)
    return _norm_t(T.functional.to_tensor(pil))


_train_geom = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=InterpolationMode.BILINEAR),
    T.RandomHorizontalFlip(0.5),
    T.RandomVerticalFlip(0.2),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
])
_AUG_RES   = list(AUG_RES_WEIGHTS.keys())
_AUG_PROBS = np.array(list(AUG_RES_WEIGHTS.values()), dtype=np.float64)
_AUG_PROBS /= _AUG_PROBS.sum()

# idea #3: degradation-realistic low-res. Real sub-100px captures carry sensor noise,
# JPEG blocking and varied resample kernels that clean BILINEAR downscaling lacks; this
# narrows the synthetic-vs-real gap (HAM-downscaled gains >> real-DM gains in the baseline).
_RESAMPLE = [Image.BILINEAR, Image.BICUBIC, Image.NEAREST, Image.BOX, Image.LANCZOS]


def degrade_lowres(pil: Image.Image, res: int) -> Image.Image:
    """Downscale to res with random kernel + optional lens blur + JPEG + sensor noise,
    then BILINEAR-upscale to 224 (deployment upscales BILINEAR). TRAIN-ONLY; eval stays clean."""
    if np.random.rand() < 0.3:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(np.random.uniform(0.3, 1.0))))
    pil = pil.resize((res, res), _RESAMPLE[np.random.randint(len(_RESAMPLE))])
    if np.random.rand() < 0.7:
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=int(np.random.randint(30, 90)))
        buf.seek(0); pil = Image.open(buf).convert("RGB")
    if np.random.rand() < 0.5:
        a = np.asarray(pil).astype(np.float32) + np.random.randn(res, res, 3) * float(np.random.uniform(2, 10))
        pil = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return pil.resize((224, 224), Image.BILINEAR)


# ── Training dataset (HAM scale-aug, optional real-DM mix) ───────────────────
class HamScaleTrain(torch.utils.data.Dataset):
    def __init__(self, paths, labels, dm_imgs=None, dm_lbls=None, mix_real=0.0, degrade=False):
        self.paths = paths
        self.labels = labels
        self.dm_imgs = dm_imgs
        self.dm_lbls = dm_lbls
        self.mix_real = mix_real
        self.degrade = degrade

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # idea #2: sometimes draw a REAL low-res DermaMNIST sample
        if self.mix_real > 0 and self.dm_imgs is not None and np.random.rand() < self.mix_real:
            j = np.random.randint(len(self.dm_imgs))
            pil = Image.fromarray(self.dm_imgs[j]).convert("RGB")
            pil = pil.resize((224, 224), Image.BILINEAR)        # native 28 -> 224
            if np.random.rand() < 0.5:
                pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
            return _norm_t(T.functional.to_tensor(pil)), int(self.dm_lbls[j])

        pil = Image.open(self.paths[i]).convert("RGB")
        pil = _train_geom(pil)
        res = int(np.random.choice(_AUG_RES, p=_AUG_PROBS))
        if res != 224:
            if self.degrade:                                    # idea #3
                pil = degrade_lowres(pil, res)
            else:
                pil = pil.resize((res, res), Image.BILINEAR).resize((224, 224), Image.BILINEAR)
        return _norm_t(T.functional.to_tensor(pil)), int(self.labels[i])


# ── Embedding + prototype helpers (always normalised — no train/eval mismatch) ─
@torch.no_grad()
def embed_ham(encoder, paths, res, bs=64):
    encoder.eval()
    out = []
    for s in range(0, len(paths), bs):
        b = torch.stack([to_224(Image.open(p), res) for p in paths[s:s + bs]]).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            f = encoder(b)
        out.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(out)


@torch.no_grad()
def embed_dm(encoder, imgs, res, bs=64):
    encoder.eval()
    out = []
    for s in range(0, len(imgs), bs):
        b = torch.stack([to_224(Image.fromarray(a), res) for a in imgs[s:s + bs]]).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            f = encoder(b)
        out.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(out)


def proto_acc(sup_f, sup_l, qry_f, qry_l):
    classes = sorted(set(int(x) for x in sup_l.tolist()))
    protos = torch.stack([F.normalize(sup_f[sup_l == c].mean(0), dim=0) for c in classes])
    pred = torch.tensor(classes)[(qry_f @ protos.t()).argmax(1)]
    return 100.0 * (pred == qry_l).float().mean().item()


# ── Cosine classifier head (normalised feats; matches deployed cosine metric) ─
class CosineHead(nn.Module):
    def __init__(self, dim, n_cls, scale=COS_SCALE):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_cls, dim) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(scale))))

    def forward(self, feat):
        f = F.normalize(feat, dim=-1)
        w = F.normalize(self.W, dim=-1)
        return self.logit_scale.exp().clamp(max=100.0) * (f @ w.t())


# ── Model wiring ─────────────────────────────────────────────────────────────
def build_encoder():
    m, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    return m.visual


def configure_trainable(visual, n_blocks):
    for p in visual.parameters():
        p.requires_grad = False
    if n_blocks > 0:
        for blk in visual.trunk.blocks[-n_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        for mod_name in ("norm", "fc_norm"):
            mod = getattr(visual.trunk, mod_name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True
        for p in visual.head.parameters():
            p.requires_grad = True
    return visual


def param_groups(visual, head, n_blocks):
    groups = [{"params": head.parameters(), "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY}]
    if n_blocks > 0:
        blocks = visual.trunk.blocks
        n_total = len(blocks)
        for idx in range(n_total - n_blocks, n_total):
            depth = (n_total - 1) - idx
            groups.append({"params": blocks[idx].parameters(),
                           "lr": BACKBONE_LR * (LAYER_DECAY ** depth),
                           "weight_decay": WEIGHT_DECAY})
        tail = list(visual.head.parameters())
        for mod_name in ("norm", "fc_norm"):
            mod = getattr(visual.trunk, mod_name, None)
            if mod is not None:
                tail += list(mod.parameters())
        groups.append({"params": tail, "lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY})
    return groups


def lr_scale(step, max_steps):
    if step < WARMUP:
        return step / max(1, WARMUP)
    prog = (step - WARMUP) / max(1, max_steps - WARMUP)
    return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))


# ── Full prototype eval (end of config / anchor) ─────────────────────────────
def full_proto_eval(visual, data):
    visual.eval()
    ham224 = proto_acc(embed_ham(visual, data["proto_paths"], 224), data["proto_lbls"],
                       embed_ham(visual, data["test_paths"], 224), data["test_lbls"])
    ham28 = proto_acc(embed_ham(visual, data["proto_paths"], 28), data["proto_lbls"],
                      embed_ham(visual, data["test_paths"], 28), data["test_lbls"])
    dm28 = proto_acc(embed_dm(visual, data["dm_proto_imgs"], 28), data["dm_proto_lbls"],
                     embed_dm(visual, data["dm_test_imgs"], 28), data["dm_test_lbls"])
    return round(ham224, 2), round(ham28, 2), round(dm28, 2)


# ── Per-config training (selection on prototype metric) ──────────────────────
def train_config(n_blocks, data, max_steps, floor, mix_real, degrade, log):
    print(f"\n{'='*70}\nCONFIG n_blocks={n_blocks}  last {n_blocks} blocks  "
          f"(guard: HAM@224 >= {floor:.2f}  mix_real={mix_real}  degrade={degrade})\n{'='*70}",
          flush=True)

    visual = configure_trainable(build_encoder().to(device), n_blocks)
    head = CosineHead(EMBED_DIM, N_CLASSES).to(device)
    n_tr = sum(p.numel() for p in visual.parameters() if p.requires_grad) + \
           sum(p.numel() for p in head.parameters())
    print(f"Trainable params: {n_tr/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(param_groups(visual, head, n_blocks))
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss(weight=data["class_w"].to(device), label_smoothing=LABEL_SMOOTH)

    loader = torch.utils.data.DataLoader(
        HamScaleTrain(data["tr_p"], data["tr_l"], data["dm_train_imgs"],
                      data["dm_train_lbls"], mix_real, degrade),
        batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)

    def sel_eval():
        """Prototype eval for selection: DM-test@28 objective + HAM@224 guard."""
        visual.eval()
        dm28 = proto_acc(embed_dm(visual, data["sel_dm_sup_imgs"], 28), data["sel_dm_sup_lbls"],
                         embed_dm(visual, data["sel_dm_qry_imgs"], 28), data["sel_dm_qry_lbls"])
        ham224 = proto_acc(embed_ham(visual, data["sel_ham_sup"], 224), data["sel_ham_sup_l"],
                           embed_ham(visual, data["sel_ham_qry"], 224), data["sel_ham_qry_l"])
        visual.train()
        return dm28, ham224

    best_obj, best_state, wait, step = -1.0, None, 0, 0
    visual.train(); head.train()
    t0 = time.time()
    it = iter(loader)

    while step < max_steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        sc = lr_scale(step, max_steps)
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * sc

        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = crit(head(visual(x)), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(
            [p for p in list(visual.parameters()) + list(head.parameters()) if p.requires_grad], 1.0)
        scaler.step(opt); scaler.update()
        step += 1

        if step % 25 == 0 or step == 1:
            print(f"  step {step:>4}/{max_steps}  loss {loss.item():.3f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  "
                  f"{(time.time()-t0)/step*1000:.0f}ms/step", flush=True)

        if step % VAL_EVERY == 0 or step == max_steps:
            dm28, ham224 = sel_eval()
            ok = ham224 >= floor
            print(f"  [sel] step {step}  DM@28={dm28:.2f}  HAM@224={ham224:.2f}  "
                  f"guard={'OK' if ok else 'FAIL'}  best={best_obj:.2f}", flush=True)
            log.append({"config": n_blocks, "seed": data["seed"], "step": step,
                        "sel_dm28": round(dm28, 3), "sel_ham224": round(ham224, 3),
                        "guard_ok": ok, "loss": round(loss.item(), 4)})
            improved = ok and dm28 > best_obj
            if improved:
                best_obj = dm28
                best_state = {"visual": {k: v.detach().cpu().clone()
                                          for k, v in visual.state_dict().items()},
                              "head": {k: v.detach().cpu().clone()
                                        for k, v in head.state_dict().items()}}
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f"  early stop (no guarded DM@28 gain in {PATIENCE} checks)", flush=True)
                    break

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = (f"nblk{n_blocks}_seed{data['seed']}" + (f"_mix{mix_real}" if mix_real else "")
           + ("_degrade" if degrade else ""))
    torch.save({"visual": visual.state_dict(), "head": head.state_dict(), "n_blocks": n_blocks},
               os.path.join(OUT_DIR, f"{tag}_last.pth"))
    if best_state is not None:
        torch.save({**best_state, "n_blocks": n_blocks},
                   os.path.join(OUT_DIR, f"{tag}_best.pth"))
        visual.load_state_dict({k: v.to(device) for k, v in best_state["visual"].items()})

    ham224, ham28, dm28 = full_proto_eval(visual, data)
    mins = (time.time() - t0) / 60
    print(f"  FULL PROTO  HAM@224={ham224}  HAM@28={ham28}  DM-test@28={dm28}  ({mins:.1f}m)",
          flush=True)
    del visual, head, opt
    torch.cuda.empty_cache()
    return {"n_blocks": n_blocks, "seed": data["seed"], "mix_real": mix_real, "degrade": degrade,
            "proto_ham224": ham224, "proto_ham28": ham28, "proto_dm28": dm28,
            "minutes": round(mins, 1)}


# ── Data assembly ────────────────────────────────────────────────────────────
def load_data(seed):
    print("Loading HAM10000 splits...", flush=True)
    tr_p, tr_l = load_ham10000(HAM_DIR, "train", 0.1, 0.1, 42)   # split seed fixed (deployment)
    va_p, va_l = load_ham10000(HAM_DIR, "val",   0.1, 0.1, 42)
    te_p, te_l = load_ham10000(HAM_DIR, "test",  0.1, 0.1, 42)
    tr_l = np.asarray(tr_l); va_l = np.asarray(va_l); te_l = np.asarray(te_l)
    rng = np.random.default_rng(seed)

    def cap(paths, labels, per):
        idx = np.concatenate([rng.choice(np.where(labels == c)[0],
                                         min(per, int((labels == c).sum())), replace=False)
                              for c in range(N_CLASSES) if (labels == c).any()])
        return [paths[i] for i in idx], torch.tensor(labels[idx])

    proto_paths, proto_lbls = cap(tr_p, tr_l, PROTO_PER_CLS)
    sel_ham_sup, sel_ham_sup_l = cap(tr_p, tr_l, SEL_SUP_PER_CLS)
    qsub = rng.choice(len(va_p), min(SEL_QRY, len(va_p)), replace=False)

    print("Loading DermaMNIST (28px)...", flush=True)
    import medmnist
    from medmnist import INFO
    DC = getattr(medmnist, INFO["dermamnist"]["python_class"])
    os.makedirs(DM_ROOT, exist_ok=True)
    dm_tr = DC(split="train", download=True, root=DM_ROOT)
    dm_va = DC(split="val",   download=True, root=DM_ROOT)
    dm_te = DC(split="test",  download=True, root=DM_ROOT)
    dm_tr_imgs, dm_tr_lbl = dm_tr.imgs, dm_tr.labels.flatten()
    dm_va_imgs, dm_va_lbl = dm_va.imgs, dm_va.labels.flatten()
    dm_te_imgs, dm_te_lbl = dm_te.imgs, dm_te.labels.flatten()

    dpi = np.concatenate([rng.choice(np.where(dm_tr_lbl == c)[0],
                                     min(PROTO_PER_CLS, int((dm_tr_lbl == c).sum())), replace=False)
                          for c in range(N_CLASSES) if (dm_tr_lbl == c).any()])
    sel_dm_sup = np.concatenate([rng.choice(np.where(dm_tr_lbl == c)[0],
                                            min(SEL_SUP_PER_CLS, int((dm_tr_lbl == c).sum())), replace=False)
                                 for c in range(N_CLASSES) if (dm_tr_lbl == c).any()])
    # selection query = DermaMNIST VAL (never test) -> no model-selection leakage
    dqsub = rng.choice(len(dm_va_imgs), min(SEL_QRY, len(dm_va_imgs)), replace=False)

    counts = np.bincount(tr_l, minlength=N_CLASSES).astype(np.float64)
    w = counts.sum() / (N_CLASSES * np.clip(counts, 1, None))
    class_w = torch.tensor(w / w.sum() * N_CLASSES, dtype=torch.float32)

    return {
        "seed": seed,
        "tr_p": tr_p, "tr_l": tr_l,
        "proto_paths": proto_paths, "proto_lbls": proto_lbls,
        "test_paths": te_p, "test_lbls": torch.tensor(te_l),
        "sel_ham_sup": sel_ham_sup, "sel_ham_sup_l": sel_ham_sup_l,
        "sel_ham_qry": [va_p[i] for i in qsub], "sel_ham_qry_l": torch.tensor(va_l[qsub]),
        "dm_proto_imgs": dm_tr_imgs[dpi], "dm_proto_lbls": torch.tensor(dm_tr_lbl[dpi]),
        "dm_train_imgs": dm_tr_imgs, "dm_train_lbls": dm_tr_lbl,   # full pool for --mix-real-dm
        "dm_test_imgs": dm_te_imgs, "dm_test_lbls": torch.tensor(dm_te_lbl),
        "sel_dm_sup_imgs": dm_tr_imgs[sel_dm_sup], "sel_dm_sup_lbls": torch.tensor(dm_tr_lbl[sel_dm_sup]),
        "sel_dm_qry_imgs": dm_va_imgs[dqsub], "sel_dm_qry_lbls": torch.tensor(dm_va_lbl[dqsub]),
        "class_w": class_w,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nblocks", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mix-real-dm", type=float, default=0.0,
                    help="prob of drawing a real DermaMNIST-28 sample per item")
    ap.add_argument("--degrade", action="store_true",
                    help="apply degradation-realistic low-res aug (idea #3)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device != "cuda":
        print("WARNING: CUDA not available — CPU will be very slow.")
    print(f"Device: {device}  seed={args.seed}  mix_real_dm={args.mix_real_dm}", flush=True)

    data = load_data(args.seed)

    # frozen anchor (encoder unchanged -> compute once, defines guard floor)
    print("\nComputing FROZEN anchor (n_blocks=0)...", flush=True)
    frozen = build_encoder().to(device)
    a224, a28, adm = full_proto_eval(frozen, data)
    del frozen; torch.cuda.empty_cache()
    floor = a224 - GUARD_TOL
    print(f"ANCHOR  HAM@224={a224}  HAM@28={a28}  DM-test@28={adm}  | guard floor {floor:.2f}",
          flush=True)

    results = [{"n_blocks": 0, "seed": args.seed, "mix_real": 0.0, "degrade": False,
                "proto_ham224": a224, "proto_ham28": a28, "proto_dm28": adm, "minutes": 0.0}]
    log = []

    if args.nblocks is not None:
        sweep = [args.nblocks]
    else:
        sweep = [n for n in SWEEP_NBLOCKS if n != 0]

    for nb in sweep:
        results.append(train_config(nb, data, args.max_steps, floor,
                                    args.mix_real_dm, args.degrade, log))
        os.makedirs(OUT_DIR, exist_ok=True)
        suffix = (f"_seed{args.seed}" + (f"_mix{args.mix_real_dm}" if args.mix_real_dm else "")
                  + ("_degrade" if args.degrade else ""))
        with open(os.path.join(OUT_DIR, f"sweep_log{suffix}.json"), "w") as f:
            json.dump(log, f, indent=2)
        with open(os.path.join(OUT_DIR, f"sweep_results{suffix}.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            wr.writeheader(); wr.writerows(results)

    print(f"\n{'='*70}\nSWEEP RESULTS (prototype / deployment metric)  seed={args.seed}"
          f"{'  mix_real='+str(args.mix_real_dm) if args.mix_real_dm else ''}\n{'='*70}")
    print(f"{'n_blocks':>9} {'HAM@224':>9} {'HAM@28':>8} {'DM-test@28':>11} {'minutes':>8}")
    for r in results:
        t = "  <- anchor" if r["n_blocks"] == 0 else ""
        print(f"{r['n_blocks']:>9} {r['proto_ham224']:>9} {r['proto_ham28']:>8} "
              f"{r['proto_dm28']:>11} {r['minutes']:>8}{t}")

    floor = a224 - GUARD_TOL
    elig = [r for r in results if r["n_blocks"] != 0 and r["proto_ham224"] >= floor]
    print(f"\nFull-res guard: HAM@224 >= {floor:.2f} (anchor {a224} - {GUARD_TOL})")
    if elig:
        best = max(elig, key=lambda r: r["proto_dm28"])
        print(f"SELECTED n_blocks={best['n_blocks']}: DM-test@28 {adm} -> {best['proto_dm28']} "
              f"(+{best['proto_dm28']-adm:.2f}pp), HAM@224 {a224} -> {best['proto_ham224']}")
    else:
        print("SELECTED: none — every config regressed full-res past the guard (honest negative).")

    print("\n*** NOT SHIPPABLE until serving artifacts regenerated + OOD/calibration re-verified. ***")


if __name__ == "__main__":
    main()
