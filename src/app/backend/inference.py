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

    # Deployed encoder. The serving artifacts (prototypes, blend params, calibration)
    # are rebuilt from the adapted checkpoint raresight_nblk4mix.pth, so the served
    # encoder must match it. Fall back to the legacy base checkpoint if the adapted
    # one is absent; RS_CKPT overrides both.
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    candidates = [
        os.environ.get("RS_CKPT"),
        os.path.join(ckpt_dir, "raresight_nblk4mix.pth"),
        os.path.join(ckpt_dir, "raresight_finetuned.pth"),
    ]
    weight_path = next((p for p in candidates if p and os.path.exists(p)), None)
    model = RareSight(device=device)
    if weight_path is not None:
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state, strict=False)
    else:
        searched = [p for p in candidates if p]
        raise FileNotFoundError(f"Model weights not found. Looked in: {searched}")
    model.eval()

    pt_path = os.path.join(APP_DIR, "assets", "disease_prototypes.pt")
    json_path = os.path.join(APP_DIR, "assets", "disease_metadata.json")
    if not (os.path.exists(pt_path) and os.path.exists(json_path)):
        raise FileNotFoundError("Precomputed assets missing. Run precompute.py first.")

    prototypes = torch.load(pt_path, map_location=device)
    with open(json_path) as f:
        metadata = json.load(f)

    # Resolution-banded artifacts (optional). When present, predict() routes a query to the
    # prototype/OOD/calibration bank matched to its NATIVE resolution — the single full-res
    # bank rejected/garbled low-res uploads (see thesis/SESSION_resolution_diagnosis.md).
    # disease_prototypes_multi.pt -> {band: {cls_id: tensor}};
    # serving_params_multi.pt     -> {bands, shared{...}, per_band{band:{...}}}.
    multi_pt = os.path.join(APP_DIR, "assets", "disease_prototypes_multi.pt")
    multi_sp = os.path.join(APP_DIR, "assets", "serving_params_multi.pt")
    prototypes_multi = (torch.load(multi_pt, map_location=device)
                        if os.path.exists(multi_pt) else None)
    serving_multi = (torch.load(multi_sp, map_location=device, weights_only=False)
                     if os.path.exists(multi_sp) else None)

    # Modality router (optional): logistic-regression probe (dermoscopy vs clinical/smartphone)
    # + the parallel CLINICAL path (prototypes/OOD/calibration fit on PAD-UFES-20 smartphone
    # photos). predict() routes smartphone uploads here so they are not garbled/OOD-rejected by
    # the dermoscopy prototypes. df/vasc have no clinical data -> dermoscopy-only (masked here).
    probe_path = os.path.join(APP_DIR, "assets", "modality_probe.pt")
    clin_pt = os.path.join(APP_DIR, "assets", "clinical_prototypes.pt")
    clin_sp = os.path.join(APP_DIR, "assets", "clinical_serving_params.pt")
    modality_probe = (torch.load(probe_path, map_location=device, weights_only=False)
                      if os.path.exists(probe_path) else None)
    clinical_protos = (torch.load(clin_pt, map_location=device, weights_only=False)
                       if os.path.exists(clin_pt) else None)
    clinical_serving = (torch.load(clin_sp, map_location=device, weights_only=False)
                        if os.path.exists(clin_sp) else None)

    # M3 blend params (gap vector, beta, lam, per-class CuPL text embeddings).
    # Used by refinement mode so user-uploaded prototypes use the SAME recipe
    # as the precomputed ones. Optional for backward compatibility.
    blend_path = os.path.join(APP_DIR, "assets", "blend_params.pt")
    blend_params = torch.load(blend_path, map_location=device) if os.path.exists(blend_path) else None

    # Serving artifacts (metadata-fusion tables, calibration temperature, open-set
    # rejection params). Built by tools/build_serving_artifacts.py. Optional.
    serving_path = os.path.join(APP_DIR, "assets", "serving_params.pt")
    # weights_only=False: this is our own artifact and holds plain python/numpy
    # containers (likelihood tables, calibration scalars), not just tensors.
    serving = (torch.load(serving_path, map_location=device, weights_only=False)
               if os.path.exists(serving_path) else None)

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
        "blend_params": blend_params,
        "serving": serving,
        "prototypes_multi": prototypes_multi,
        "serving_multi": serving_multi,
        "modality_probe": modality_probe,
        "clinical_protos": clinical_protos,
        "clinical_serving": clinical_serving,
        "rollout": rollout,
    }


def _modality_is_clinical(q_np, resources):
    """Route by the modality probe: P(clinical) = sigmoid(coef.q + b). Returns (is_clinical, p).
    Biased to default dermoscopy (the primary/trained domain): only switches to the clinical
    path when P(clinical) > T (T chosen on val to keep dermoscopy false-route ~1%)."""
    probe = resources.get("modality_probe")
    if not probe or resources.get("clinical_protos") is None:
        return False, None
    z = float(np.dot(probe["coef"][0], q_np) + probe["intercept"])
    p = 1.0 / (1.0 + np.exp(-z))
    return bool(p > probe["T"]), p


def _encode_tta(model, image, device):
    """Flip/rotation TTA: average L2-normed embeddings over 6 views, renormalize. Used ONLY
    for clinical-routed queries — the clinical prototypes/OOD are built under the SAME TTA
    encoding (scripts/build_clinical_tta.py), so the query must match. Dermoscopy stays
    single-view. Lesions are rotation/flip-invariant, so these are label-preserving views."""
    views = [image,
             image.transpose(Image.FLIP_LEFT_RIGHT), image.transpose(Image.FLIP_TOP_BOTTOM),
             image.transpose(Image.ROTATE_90), image.transpose(Image.ROTATE_180),
             image.transpose(Image.ROTATE_270)]
    with torch.no_grad():
        b = torch.cat([model.preprocess(v).unsqueeze(0) for v in views]).to(device)
        e = model.backbone.encode_image(b)
        e = e / e.norm(dim=-1, keepdim=True)
        e = e.mean(0, keepdim=True)
        e = e / e.norm(dim=-1, keepdim=True)
    return e  # (1, D)


def _select_band(image, resources):
    """Route a query to the resolution band matched to its NATIVE min-side (known BEFORE the
    preprocess upscales to 224). Returns (base_protos_dict, serving_dict, band) mirroring the
    single-band interface, or (None, None, None) when no multi-band artifact is loaded."""
    pmulti = resources.get("prototypes_multi")
    smulti = resources.get("serving_multi")
    if not pmulti or not smulti:
        return None, None, None
    bands = smulti["bands"]
    native = min(image.size)  # PIL size = (W, H)
    # nearest band in log-resolution space (perceptually even spacing)
    band = min(bands, key=lambda b: abs(np.log(b) - np.log(max(native, 1))))
    base_protos = pmulti[band]
    # merge shared (resolution-independent: meta tables, temp) + per-band (protos' OOD/calib/alpha)
    serving = {**smulti["shared"], **smulti["per_band"][band]}
    return base_protos, serving, band


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

RISK = {0: "medium", 1: "high", 2: "low", 3: "low", 4: "high", 5: "low", 6: "low"}

# Demo weight for clinical-note text evidence. NOT data-tuned (HAM10000 has no
# free-text notes), so it is deliberately small and gated — see _note_loglik.
NOTE_BETA = 0.5


def _age_bin(age):
    try:
        a = float(age)
    except (TypeError, ValueError):
        return None
    if a != a:  # NaN
        return None
    return int(min(max(a, 0.0), 89.0) // 10)


def _norm_sex(s):
    if not s:
        return None
    s = str(s).strip().lower()
    if s in ("m", "male"):
        return "male"
    if s in ("f", "female"):
        return "female"
    return "unknown"


def _meta_loglik(serving, metadata, n_classes):
    """log P(meta | class) over classes, summing the available age/sex/site terms.
    Missing fields are dropped. Returns (vector, list-of-fields-used)."""
    lt = serving["meta_logtab"]
    out = np.zeros(n_classes)
    used = []
    ab = _age_bin(metadata.get("age"))
    if ab is not None:
        out = out + np.array(lt["age"])[:, ab]
        used.append("age")
    sx = _norm_sex(metadata.get("sex"))
    if sx is not None:
        xi = serving["meta_sex_index"]
        out = out + np.array(lt["sex"])[:, xi.get(sx, xi["unknown"])]
        used.append("sex")
    site = metadata.get("localization")
    if site:
        si = serving["meta_site_index"]
        idx = si.get(str(site).strip().lower(), si.get("unknown"))
        if idx is not None:
            out = out + np.array(lt["site"])[:, idx]
            used.append("site")
    return out, used


def _ood_score(serving, q_np, z_img, cos):
    """Image-only out-of-distribution score (higher = more in-distribution).
    Computed BEFORE metadata fusion so a confident prior cannot mask an OOD image.
    Must mirror build_serving_artifacts.scores() EXACTLY for the chosen method, since
    ood_tau is a quantile on that method's score scale (e.g. maxcos lives on the
    cosine scale ~[-1,1], NOT the distance-logit scale of z_img)."""
    method = serving.get("ood_method", "maha")
    if method == "maha":
        means = np.array(serving["maha_means"])
        inv = np.array(serving["maha_inv_cov"])
        diff = q_np[None, :] - means                       # (C, D)
        d = np.einsum("ci,ij,cj->c", diff, inv, diff)
        return float(-d.min())
    if method == "energy":
        return float(np.log(np.exp(z_img - z_img.max()).sum()) + z_img.max())
    if method == "msp":
        e = np.exp(z_img - z_img.max())
        return float((e / e.sum()).max())
    return float(cos.max())  # maxcos (cosine to prototypes) / fallback


def _note_loglik(model, device, note, text_embs, class_order, scale=12.0):
    """Match a free-text clinical note to each class's CuPL text embedding in
    BiomedCLIP's text space. Gated: skips short or non-discriminative (near-uniform)
    notes so a generic note cannot move the prediction. Demo capability."""
    toks = model.tokenizer([note], padding="max_length", truncation=True,
                           max_length=256, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        e = model.backbone.encode_text(toks)
        e = (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
    sims = np.array([float(e @ (text_embs[c].cpu().numpy()
                    if hasattr(text_embs[c], "cpu") else np.asarray(text_embs[c])))
                     for c in class_order])
    p = np.exp((sims - sims.max()) * scale)
    p = p / p.sum()
    ent = float(-np.sum(p * np.log(p + 1e-12)))
    if len(note.strip()) < 12 or ent > 0.97 * np.log(len(class_order)):
        return np.zeros(len(class_order)), False, None
    return np.log(p + 1e-12), True, class_order[int(np.argmax(p))]


def predict(image: Image.Image, resources: dict, user_prototypes: dict | None = None,
            metadata: dict | None = None, clinical_note: str | None = None):
    model = resources["model"]
    device = resources["device"]
    base_protos = resources["prototypes"]
    class_meta = resources["metadata"]
    rollout = resources["rollout"]
    serving = resources.get("serving")
    blend = resources.get("blend_params")

    # Resolution routing: if banded artifacts exist and we are NOT in user-refinement mode,
    # swap in the prototype/OOD/calibration bank matched to the query's native resolution.
    band = None
    if user_prototypes is None:
        b_protos, b_serving, band = _select_band(image, resources)
        if b_protos is not None:
            base_protos, serving = b_protos, b_serving

    q_tensor = model.preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        q_emb = model.backbone.encode_image(q_tensor)
        q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)
    q_np = q_emb.cpu().numpy()[0]

    class_order = sorted(int(k) for k in class_meta.keys())
    n = len(class_order)

    # Modality routing: a smartphone/clinical upload goes to the clinical path (its own
    # prototypes + OOD + calibration); dermoscopy stays on the resolution-banded path above.
    # Defaults to dermoscopy (primary domain); switches only when P(clinical) > probe T.
    modality, p_clinical = "dermoscopy", None
    class_mask = np.zeros(n)  # -inf disables classes unavailable in the chosen path (clinical: df/vasc)
    if user_prototypes is None:
        is_clin, p_clinical = _modality_is_clinical(q_np, resources)
        if is_clin:
            modality, band = "clinical", None
            base_protos = resources["clinical_protos"]
            serving = resources.get("clinical_serving")
            avail = {int(c) for c in base_protos.keys()}
            class_mask = np.array([0.0 if c in avail else -np.inf for c in class_order])
            # Match the query encoding to the clinical artifacts: if they were built under TTA
            # (Tier-1 keeper), re-encode the query with TTA so it lands in the same space. The
            # modality decision above still used the base embedding (the probe is base-trained).
            if serving and serving.get("encoding") == "tta_flip_rot6":
                q_emb = _encode_tta(model, image, device)
                q_np = q_emb.cpu().numpy()[0]

    proto_list = []
    for cls_id in class_order:
        if user_prototypes and cls_id in user_prototypes:
            proto_list.append(user_prototypes[cls_id].to(device))
        elif cls_id in base_protos:
            proto_list.append(base_protos[cls_id].to(device))
        else:
            proto_list.append(torch.zeros(q_emb.shape[-1], device=device))  # masked (clinical path)
    proto_tensor = torch.stack(proto_list)  # [N, 512]

    temp = serving["temp_metric"] if serving else model.temperature.item()
    dists = torch.cdist(q_emb, proto_tensor)            # [1, N]
    z_img = (-dists * temp).cpu().numpy()[0] + class_mask   # image-only logits, unavailable classes -> -inf
    cos = (q_emb @ proto_tensor.T).cpu().numpy()[0]     # cosine to prototypes (N,)

    # 1. Open-set rejection — image-only score, BEFORE any fusion.
    is_unknown, ood_score = False, None
    if serving:
        ood_score = _ood_score(serving, q_np, z_img, cos)
        is_unknown = bool(ood_score < serving["ood_tau"])

    img_top = class_order[int(np.argmax(z_img))]

    # 2. Metadata fusion (refines ranking among known classes). The clinical path's serving
    # artifact has no metadata tables (HAM-only) -> guarded; clinical stays image-only for now.
    fused = z_img.copy()
    meta_used = []
    if serving and metadata and "meta_logtab" in serving:
        ll, meta_used = _meta_loglik(serving, metadata, n)
        if meta_used:
            fused = fused + float(serving["meta_alpha"]) * ll

    # 3. Clinical note (gated demo evidence).
    note_used, note_support = False, None
    if serving and blend and clinical_note and clinical_note.strip():
        nll, note_used, note_support_idx = _note_loglik(
            model, device, clinical_note, blend["text_embs"], class_order)
        if note_used:
            fused = fused + NOTE_BETA * nll
            note_support = class_meta[str(note_support_idx)]["name"]

    # 4. Temperature-scaled calibration → displayed confidence.
    # Pick the temperature/ECE fit on the regime actually served: image-only vs
    # image+metadata logits live on different scales (note adds a small extra term,
    # treated under the fused regime). Avoids misattributing the ECE for image-only.
    used_fusion = bool(meta_used) or note_used
    calib_ece = None
    if serving:
        if used_fusion:
            T = float(serving.get("calib_T_meta", serving.get("calib_T", 1.0)))
            calib_ece = serving.get("ece_after_meta_test", serving.get("ece_after_test"))
        else:
            T = float(serving.get("calib_T_img", serving.get("calib_T", 1.0)))
            calib_ece = serving.get("ece_after_img_test", serving.get("ece_after_test"))
    else:
        T = 1.0
    ft = fused / T
    probs = np.exp(ft - ft.max())
    probs = probs / probs.sum()

    top3_idx = np.argsort(probs)[::-1][:3]
    top_cls_id = class_order[int(top3_idx[0])]

    predictions = []
    for rank, idx in enumerate(top3_idx):
        cls_id = class_order[int(idx)]
        predictions.append({
            "rank": rank + 1,
            "class_id": cls_id,
            "class_name": class_meta[str(cls_id)]["name"],
            "probability": float(probs[idx]),
            "risk": RISK.get(cls_id, "low"),
        })

    entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
    # Normalise entropy by the number of AVAILABLE classes (clinical path masks df/vasc -> 5),
    # else the clinical fraction is understated. The clinical path is structurally incomplete
    # (no df/vasc data in PAD), so it refers more readily — a df/vasc clinical photo cannot be
    # classified correctly there and should be flagged, not silently misclassified.
    n_eff = int(np.sum(np.isfinite(class_mask))) if class_mask is not None else n
    max_entropy = float(np.log(max(n_eff, 2)))
    refer_frac = 0.70 if modality == "clinical" else 0.75
    refer = bool(is_unknown or entropy > refer_frac * max_entropy)

    heatmap_b64 = _generate_heatmap_b64(rollout, q_tensor, image)
    ref_images = _load_reference_images(top_cls_id)

    return {
        "predictions": predictions,
        "entropy": entropy,
        "refer_to_specialist": refer,
        "is_unknown": is_unknown,
        "ood_score": ood_score,
        "calibration_ece": calib_ece,
        "native_resolution": int(min(image.size)),
        "resolution_band": band,
        "modality": modality,
        "p_clinical": (round(float(p_clinical), 3) if p_clinical is not None else None),
        "top_class_id": top_cls_id,
        "top_class_name": class_meta[str(top_cls_id)]["name"],
        "metadata_used": meta_used,
        "metadata_changed_ranking": bool(meta_used and img_top != top_cls_id),
        "note_used": note_used,
        "note_supports": note_support,
        "heatmap_b64": heatmap_b64,
        "reference_images": ref_images,
    }


def compute_user_prototype(images: list[Image.Image], cls_id: int, resources: dict):
    """Refinement mode: build a class prototype from user-uploaded images using the
    SAME M3 aligned-space recipe as precompute (image proto blended with the class's
    CuPL text + modality-gap correction). Falls back to image-only if blend params
    are unavailable."""
    model = resources["model"]
    device = resources["device"]
    blend = resources.get("blend_params")

    tensors = [model.preprocess(img).unsqueeze(0) for img in images]
    s_tensor = torch.cat(tensors).to(device)

    with torch.no_grad():
        emb = model.backbone.encode_image(s_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        img_proto = emb.mean(dim=0)
        img_proto = img_proto / img_proto.norm()

    if blend is None or cls_id not in blend.get("text_embs", {}):
        return img_proto  # backward-compatible image-only fallback

    beta, lam = blend["beta"], blend["lam"]
    gap = blend["gap"].to(device)
    text_emb = blend["text_embs"][cls_id].to(device)
    txt_shift = text_emb + lam * gap
    txt_shift = txt_shift / txt_shift.norm()
    proto = beta * img_proto + (1.0 - beta) * txt_shift
    return proto / proto.norm()


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
