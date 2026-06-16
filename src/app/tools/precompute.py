"""
RareSight prototype pre-computation — HAM10000 edition.

Builds per-class multi-modal prototypes from HAM10000 (600×450 high-res
dermoscopy images), which is the same dataset the model was trained on.
Using more support images (K_SHOT=20) and proper source images gives
richer, more discriminative class centroids than the old DermaMNIST path.

Usage:
    python src/app/tools/precompute.py

Output:
    src/app/assets/disease_prototypes.pt  — {class_id: tensor(512)}
    src/app/assets/disease_metadata.json  — class names + ref image paths
    src/app/assets/reference_images/      — cls_{id}_ref_{i}.jpg  (5 per class)
"""

import sys
import os
import json
import random
import shutil
import torch
import numpy as np
from PIL import Image

# ── Path setup ───────────────────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)

from src.models.raresight_net import RareSight
from src.data.preprocessing import load_ham10000, HAM10000_LABEL_MAP

# ── Configuration ─────────────────────────────────────────────────────────────
K_SHOT        = 20          # support images per class for prototype averaging
K_REF         = 5           # reference images saved to disk for the UI gallery
SEED          = 42          # reproducible selection
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
HAM_DIR       = os.path.join(project_root, "data", "ham10000")
CHECKPOINT    = os.path.join(project_root, "checkpoints", "raresight_nblk4mix.pth")
DESC_PATH     = os.path.join(project_root, "src", "app", "class_descriptions.json")
CUPL_PATH     = os.path.join(project_root, "src", "app", "cupl_descriptions.json")
ASSETS_DIR    = os.path.join(project_root, "src", "app", "assets")
REF_IMG_DIR   = os.path.join(ASSETS_DIR, "reference_images")

# ── M3 prototype recipe (see eval_multimodal_tier1.py) ────────────────────────
# Prototypes are built in BiomedCLIP's ALIGNED space, NOT via the MLP fusion
# (which an ablation showed HURTS: image-only 57.4% > MLP-multimodal 54.7%).
#   proto = normalize( BETA * image_proto + (1-BETA) * normalize(text_cupl + LAM*gap) )
# This M3 recipe scored +3.56 over image-only on 5-way 5-shot (60.93% vs 57.37%).
BETA = 0.75    # weight on the image prototype (1.0 = image-only)
LAM  = 1.0     # modality-gap correction strength (shift text toward image manifold)

# Friendly display names (matches class_descriptions.json keys 0-6)
CLASSES = {
    0: "Actinic keratoses",
    1: "Basal cell carcinoma",
    2: "Benign keratosis",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Melanocytic nevi",
    6: "Vascular lesions",
}


def _select_diverse(paths: list, labels: np.ndarray, cls_id: int, k: int, seed: int) -> list:
    """Return up to k paths for cls_id, randomly sampled with a fixed seed."""
    cls_paths = [p for p, l in zip(paths, labels) if int(l) == cls_id]
    rng = random.Random(seed + cls_id)
    rng.shuffle(cls_paths)
    if len(cls_paths) < k:
        print(f"  Warning: class {cls_id} has only {len(cls_paths)} images (wanted {k})")
    return cls_paths[:k]


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def _image_prototype(model, image_paths: list, device: str) -> torch.Tensor:
    """Mean of normalized support image embeddings (image-only prototype, 512-d)."""
    tensors = []
    for path in image_paths:
        pil = Image.open(path).convert("RGB")
        # model.preprocess = BiomedCLIP's own transform (matches inference.py)
        tensors.append(model.preprocess(pil).unsqueeze(0))
    s_images = torch.cat(tensors).to(device)
    with torch.no_grad():
        emb = _norm(model.backbone.encode_image(s_images))   # (K, 512)
    return _norm(emb.mean(dim=0)).cpu()


def _cupl_text_embedding(model, prompts: list, device: str) -> torch.Tensor:
    """CuPL prompt-ensemble text embedding: mean of normalized encode_text over prompts."""
    toks = model.tokenizer(prompts, padding="max_length", truncation=True,
                           max_length=256, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        emb = _norm(model.backbone.encode_text(toks))        # (P, 512)
    return _norm(emb.mean(dim=0)).cpu()


def _save_reference_images(src_paths: list, cls_id: int) -> list[str]:
    """Copy the first K_REF images to assets/reference_images/, return rel paths."""
    saved = []
    for i, src in enumerate(src_paths[:K_REF]):
        dst_name = f"cls_{cls_id}_ref_{i}.jpg"
        dst_path = os.path.join(REF_IMG_DIR, dst_name)
        # Re-save via PIL so we get consistent JPEG quality regardless of source
        pil = Image.open(src).convert("RGB")
        pil.save(dst_path, format="JPEG", quality=90)
        saved.append(f"assets/reference_images/{dst_name}")
    return saved


def main():
    print("=" * 60)
    print("RareSight Prototype Pre-computation  (HAM10000 edition)")
    print(f"  K_SHOT  = {K_SHOT}   (support images per class)")
    print(f"  K_REF   = {K_REF}    (reference images saved for UI)")
    print(f"  Device  = {DEVICE}")
    print(f"  Dataset = {HAM_DIR}")
    print("=" * 60)

    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(REF_IMG_DIR, exist_ok=True)

    # ── Load descriptions (original 1-liner for metadata; CuPL bank for text) ──
    with open(DESC_PATH, "r") as f:
        descriptions = json.load(f)
    with open(CUPL_PATH, "r") as f:
        cupl = json.load(f)

    # ── Load model ──────────────────────────────────────────────────────
    model = RareSight(device=DEVICE)
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}\n"
            "Run training + fine-tuning first."
        )
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"\nLoaded checkpoint: {CHECKPOINT}")
    print(f"  Temperature: {model.temperature.item():.4f}")

    # ── Load HAM10000 train split ────────────────────────────────────────
    print(f"\nLoading HAM10000 train split from: {HAM_DIR}")
    try:
        train_paths, train_labels = load_ham10000(
            data_root=HAM_DIR,
            split="train",
            val_size=0.10,
            test_size=0.10,
            seed=SEED,
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    # Class distribution
    print(f"  Total train images: {len(train_paths)}")
    for cls_id, cls_name in CLASSES.items():
        count = int((train_labels == cls_id).sum())
        print(f"  Class {cls_id} ({cls_name}): {count} images")

    # ── Pass 1: image prototypes + CuPL text embeddings per class ────────
    img_protos   = {}   # cls_id -> (512,) image-only prototype
    text_embs    = {}   # cls_id -> (512,) CuPL text embedding
    metadata_dict = {}

    print("\nPass 1: image prototypes + CuPL text embeddings from HAM10000 images...")
    for cls_id, cls_name in CLASSES.items():
        print(f"\n  [{cls_id}] {cls_name}")
        support_paths = _select_diverse(train_paths, train_labels, cls_id, K_SHOT, SEED)
        if not support_paths:
            print(f"    ERROR: no images found for class {cls_id}. Skipping.")
            continue

        img_protos[cls_id] = _image_prototype(model, support_paths, DEVICE)
        prompts = cupl.get(str(cls_id)) or [descriptions.get(str(cls_id), cls_name)]
        text_embs[cls_id] = _cupl_text_embedding(model, prompts, DEVICE)
        ref_paths = _save_reference_images(support_paths, cls_id)

        metadata_dict[str(cls_id)] = {
            "name":             cls_name,
            "description":      descriptions.get(str(cls_id), f"Dermoscopy image of {cls_name.lower()}"),
            "reference_images": ref_paths,
            "n_support":        len(support_paths),
            "n_cupl_prompts":   len(prompts),
            "source":           "ham10000_train",
        }
        print(f"    image proto + CuPL text ({len(prompts)} prompts) built; {len(ref_paths)} refs saved")

    # ── Modality gap: mean image manifold - mean CuPL text (unit vector) ──
    img_stack = torch.stack([img_protos[c] for c in img_protos])
    txt_stack = torch.stack([text_embs[c] for c in text_embs])
    gap = img_stack.mean(0) - txt_stack.mean(0)
    gap = gap / gap.norm()
    print(f"\nModality gap |mean_img - mean_txt| = {(img_stack.mean(0) - txt_stack.mean(0)).norm():.3f}")

    # ── Pass 2: M3 aligned-space blend ───────────────────────────────────
    print(f"Pass 2: aligned blend  proto = norm({BETA}*img + {1-BETA:.2f}*norm(text + {LAM}*gap))")
    prototypes_dict = {}
    for cls_id in img_protos:
        txt_shift = _norm(text_embs[cls_id] + LAM * gap)
        proto = _norm(BETA * img_protos[cls_id] + (1.0 - BETA) * txt_shift)
        prototypes_dict[cls_id] = proto

    # ── Save to disk ────────────────────────────────────────────────────
    pt_path    = os.path.join(ASSETS_DIR, "disease_prototypes.pt")
    json_path  = os.path.join(ASSETS_DIR, "disease_metadata.json")
    blend_path = os.path.join(ASSETS_DIR, "blend_params.pt")

    torch.save(prototypes_dict, pt_path)
    # Persist blend params so refinement-mode (user uploads) uses the SAME recipe.
    torch.save({"gap": gap, "beta": BETA, "lam": LAM,
                "text_embs": text_embs}, blend_path)
    with open(json_path, "w") as f:
        json.dump(metadata_dict, f, indent=4)
    print(f"  Blend params saved : {blend_path}")

    print("\n" + "=" * 60)
    print("Precomputation complete!")
    print(f"  Prototypes : {pt_path}")
    print(f"  Metadata   : {json_path}")
    print(f"  Ref images : {REF_IMG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
