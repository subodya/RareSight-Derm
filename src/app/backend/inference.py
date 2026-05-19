"""
RareSight inference engine — model loading, prediction, attention rollout.
Ported from app.py; framework-agnostic (no Streamlit imports).
"""

import os
import io
import base64
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

APP_DIR = os.path.dirname(os.path.dirname(__file__))          # src/app/
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, "../../"))  # repo root


# ---------------------------------------------------------------------------
# Attention Rollout
# ---------------------------------------------------------------------------

class AttentionRollout:
    """Attention Rollout explainability for BiomedCLIP ViT backbone."""

    def __init__(self, model, head_fusion="mean", discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.grid_size = 14  # ViT-B/16 @ 224×224 → 14×14 patches

    def get_attention_maps(self, image_tensor):
        attention_maps = []
        with torch.no_grad():
            visual = self.model.backbone.visual
            x = visual.trunk.patch_embed(image_tensor)
            x = visual.trunk._pos_embed(x)
            for block in visual.trunk.blocks:
                B, N, C = x.shape
                qkv = block.attn.qkv(x)
                qkv = qkv.reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, _ = qkv[0], qkv[1], qkv[2]
                attn = (q @ k.transpose(-2, -1)) * block.attn.scale
                attn = attn.softmax(dim=-1)
                attention_maps.append(attn[0].cpu())
                x = block(x)
        return attention_maps

    def rollout(self, attention_maps):
        fused = []
        for attn in attention_maps:
            if self.head_fusion == "mean":
                fused.append(attn.mean(dim=0))
            else:
                fused.append(attn.max(dim=0)[0])

        num_tokens = fused[0].shape[0]
        identity = torch.eye(num_tokens)
        result = identity
        for attn in fused:
            a = attn + identity
            a = a / a.sum(dim=-1, keepdim=True)
            result = torch.matmul(a, result)
        return result[0, 1:]  # CLS → patches

    def generate_heatmap(self, image_tensor):
        maps = self.get_attention_maps(image_tensor)
        cls_attn = self.rollout(maps)
        heatmap = cls_attn.reshape(self.grid_size, self.grid_size).numpy()
        if self.discard_ratio > 0:
            threshold = np.percentile(heatmap.flatten(), self.discard_ratio * 100)
            heatmap[heatmap < threshold] = 0
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap

    def overlay_heatmap(self, original_image: Image.Image, heatmap: np.ndarray, alpha=0.4):
        img_np = np.array(original_image.convert("RGB"))
        h, w = img_np.shape[:2]
        hm_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        hm_colored = cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET)
        hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_np.astype(np.uint8), 1 - alpha, hm_colored, alpha, 0)
        return overlay


# ---------------------------------------------------------------------------
# Resource loader (call once at startup)
# ---------------------------------------------------------------------------

def load_resources():
    import sys
    sys.path.insert(0, PROJECT_ROOT)
    from src.models.raresight_net import RareSight

    device = "cuda" if torch.cuda.is_available() else "cpu"

    weight_path = os.path.join(PROJECT_ROOT, "checkpoints", "raresight_finetuned.pth")
    model = RareSight(device=device)
    if os.path.exists(weight_path):
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state, strict=False)
    else:
        raise FileNotFoundError(f"Model weights not found at {weight_path}")
    model.eval()

    pt_path = os.path.join(APP_DIR, "assets", "disease_prototypes.pt")
    json_path = os.path.join(APP_DIR, "assets", "disease_metadata.json")
    if not (os.path.exists(pt_path) and os.path.exists(json_path)):
        raise FileNotFoundError("Precomputed assets missing. Run precompute.py first.")

    prototypes = torch.load(pt_path, map_location=device)
    with open(json_path) as f:
        metadata = json.load(f)

    classes_json = os.path.join(APP_DIR, "class_descriptions.json")
    with open(classes_json) as f:
        class_descriptions = json.load(f)

    rollout = AttentionRollout(model)

    return {
        "model": model,
        "device": device,
        "prototypes": prototypes,
        "metadata": metadata,
        "class_descriptions": class_descriptions,
        "rollout": rollout,
    }


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

RISK = {0: "medium", 1: "high", 2: "low", 3: "low", 4: "high", 5: "low", 6: "low"}


def predict(image: Image.Image, resources: dict, user_prototypes: dict | None = None):
    model = resources["model"]
    device = resources["device"]
    base_protos = resources["prototypes"]
    metadata = resources["metadata"]
    rollout = resources["rollout"]

    q_tensor = model.preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        q_emb = model.backbone.encode_image(q_tensor)
        q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)

    class_order = sorted(int(k) for k in metadata.keys())
    proto_list = []
    for cls_id in class_order:
        if user_prototypes and cls_id in user_prototypes:
            proto_list.append(user_prototypes[cls_id].to(device))
        else:
            proto_list.append(base_protos[cls_id].to(device))
    proto_tensor = torch.stack(proto_list)  # [N, 512]

    dists = torch.cdist(q_emb, proto_tensor)  # [1, N]
    probs = F.softmax(-dists, dim=1).cpu().numpy()[0]

    top3_idx = np.argsort(probs)[::-1][:3]
    top_cls_id = class_order[top3_idx[0]]

    predictions = []
    for rank, idx in enumerate(top3_idx):
        cls_id = class_order[idx]
        predictions.append({
            "rank": rank + 1,
            "class_id": cls_id,
            "class_name": metadata[str(cls_id)]["name"],
            "probability": float(probs[idx]),
            "risk": RISK.get(cls_id, "low"),
        })

    entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
    max_entropy = float(np.log(len(class_order)))
    refer = entropy > 0.75 * max_entropy

    heatmap_b64 = _generate_heatmap_b64(rollout, q_tensor, image)

    ref_images = _load_reference_images(top_cls_id)

    return {
        "predictions": predictions,
        "entropy": entropy,
        "refer_to_specialist": refer,
        "top_class_id": top_cls_id,
        "top_class_name": metadata[str(top_cls_id)]["name"],
        "heatmap_b64": heatmap_b64,
        "reference_images": ref_images,
    }


def compute_user_prototype(images: list[Image.Image], cls_id: int, resources: dict):
    model = resources["model"]
    device = resources["device"]
    metadata = resources["metadata"]
    description = metadata[str(cls_id)].get("description", "")

    tensors = [model.preprocess(img).unsqueeze(0) for img in images]
    s_tensor = torch.cat(tensors).to(device)
    texts = [description] * len(images)

    with torch.no_grad():
        fused = model.encode_multimodal(s_tensor, texts)
        proto = fused.mean(dim=0)
        proto = proto / proto.norm(dim=-1, keepdim=True)

    return proto


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_heatmap_b64(rollout: AttentionRollout, q_tensor, original_image: Image.Image) -> str:
    try:
        heatmap = rollout.generate_heatmap(q_tensor)
        overlay = rollout.overlay_heatmap(original_image, heatmap)
        pil_overlay = Image.fromarray(overlay)
        buf = io.BytesIO()
        pil_overlay.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def _load_reference_images(cls_id: int) -> list[str]:
    ref_dir = os.path.join(APP_DIR, "assets", "reference_images")
    results = []
    for i in range(5):
        path = os.path.join(ref_dir, f"cls_{cls_id}_ref_{i}.jpg")
        if os.path.exists(path):
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            results.append(b64)
        if len(results) == 3:
            break
    return results
