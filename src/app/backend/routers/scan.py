"""Scan domain: prediction, few-shot refinement, session management."""

import os
import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from .. import db
from ..inference import APP_DIR, compute_user_prototype, predict
from ..state import resources, session_store

router = APIRouter(prefix="/api", tags=["scan"])

ASSETS_DIR = os.path.join(APP_DIR, "assets")
THUMB_MAX = 480  # px, longest side


def _save_thumbnail(image: Image.Image) -> str:
    thumb = image.copy()
    thumb.thumbnail((THUMB_MAX, THUMB_MAX))
    name = f"case_{uuid.uuid4().hex[:12]}.jpg"
    thumb.convert("RGB").save(os.path.join(db.UPLOADS_DIR, name), "JPEG", quality=88)
    return name


def _reference_image_urls(cls_id: int, limit: int = 3) -> list[str]:
    urls = []
    for i in range(5):
        rel = f"reference_images/cls_{cls_id}_ref_{i}.jpg"
        if os.path.exists(os.path.join(ASSETS_DIR, rel)):
            urls.append(f"/api/images/{rel}")
        if len(urls) == limit:
            break
    return urls


@router.post("/predict")
async def api_predict(
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
    patient_name: str | None = Form(default=None),
    age: str | None = Form(default=None),
    sex: str | None = Form(default=None),
    localization: str | None = Form(default=None),
    scan_type: str | None = Form(default=None),
    clinical_note: str | None = Form(default=None),
):
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    user_protos = session_store.get(session_id) if session_id else None

    # Structured patient inputs fuse into classification (age/sex/site via a
    # class-conditional prior; note via BiomedCLIP text matching). Empty fields
    # are dropped downstream.
    patient_meta = {"age": age, "sex": sex, "localization": localization}

    try:
        result = predict(image, resources, user_protos,
                         metadata=patient_meta, clinical_note=clinical_note)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    # Clinical photos have a dedicated path (the modality router + PAD-fit clinical
    # prototypes), so they are NOT out-of-distribution — no warning. Only genuinely
    # unsupported modalities (e.g. histopathology) get the domain-shift guard.
    SUPPORTED_MODALITIES = ("", "dermoscopy", "clinical photo")
    if scan_type and scan_type.strip().lower() not in SUPPORTED_MODALITIES:
        result["modality_warning"] = (
            f"Model trained on dermoscopy; '{scan_type}' is out-of-distribution — "
            "interpret with caution."
        )

    # Serving-layer triage guardrail: the model's referral flag is uncertainty-
    # based (entropy/OOD), so a *confident* malignancy prediction would read as
    # "routine follow-up". Any suspected malignancy warrants specialist review
    # regardless of confidence — standard triage policy, applied at serving time.
    top = result["predictions"][0]
    if top["risk"] == "high" and not result["refer_to_specialist"]:
        result["refer_to_specialist"] = True
        result["referral_reason"] = "suspected_malignancy"

    # Serve reference images by URL (static mount) rather than inline base64.
    result["reference_images"] = _reference_image_urls(result["top_class_id"])

    thumb = _save_thumbnail(image)
    case_id = db.insert_case(
        patient={
            "name": patient_name,
            "age": age,
            "sex": sex,
            "localization": localization,
            "scan_type": scan_type,
            "clinical_note": clinical_note,
        },
        result=result,
        thumbnail=thumb,
    )
    result["case_id"] = case_id
    result["case_display_id"] = db.display_id(case_id)

    return JSONResponse(result)


@router.post("/refine")
async def api_refine(
    class_id: Annotated[int, Form()],
    files: list[UploadFile] = File(...),
    session_id: str | None = Form(default=None),
):
    if not files:
        raise HTTPException(400, "At least one reference image required")
    if len(files) > 5:
        files = files[:5]

    images = []
    for f in files:
        try:
            img = Image.open(f.file).convert("RGB")
            if img.size[0] < 28 or img.size[1] < 28:
                raise HTTPException(400, f"Image too small: {f.filename}")
            images.append(img)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(400, f"Cannot read image: {f.filename}")

    try:
        proto = compute_user_prototype(images, class_id, resources)
    except Exception as e:
        raise HTTPException(500, f"Prototype computation error: {e}")

    if not session_id:
        session_id = str(uuid.uuid4())

    session_store.setdefault(session_id, {})[class_id] = proto

    return {
        "session_id": session_id,
        "class_id": class_id,
        "images_used": len(images),
        "message": f"Prototype updated for class {class_id}",
    }


@router.delete("/session/{session_id}")
def clear_session(session_id: str):
    session_store.pop(session_id, None)
    return {"cleared": session_id}
