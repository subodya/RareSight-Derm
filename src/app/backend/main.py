"""
RareSight FastAPI backend.

Run:  uvicorn src.app.backend.main:app --reload --port 8000
"""

import uuid
from contextlib import asynccontextmanager
from typing import Annotated

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from .inference import compute_user_prototype, load_resources, predict

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

resources: dict = {}
# Ephemeral in-memory store for user-computed prototypes keyed by session_id
session_store: dict[str, dict[int, torch.Tensor]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    resources.update(load_resources())
    yield
    resources.clear()
    session_store.clear()


app = FastAPI(
    title="RareSight API",
    description="Calibrated multimodal dermatology triage for rural GPs",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": resources.get("device", "unknown"),
        "classes_loaded": len(resources.get("metadata", {})),
        "model": "RareSight (BiomedCLIP + Prototypical Meta-Learning)",
    }


@app.get("/api/classes")
def get_classes():
    metadata = resources["metadata"]
    descriptions = resources["class_descriptions"]
    risk_map = {0: "medium", 1: "high", 2: "low", 3: "low", 4: "high", 5: "low", 6: "low"}
    return {
        str(cls_id): {
            "id": int(cls_id),
            "name": meta["name"],
            "description": descriptions.get(str(cls_id), ""),
            "risk": risk_map.get(int(cls_id), "low"),
        }
        for cls_id, meta in metadata.items()
    }


@app.post("/api/predict")
async def api_predict(
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
):
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    user_protos = session_store.get(session_id) if session_id else None

    try:
        result = predict(image, resources, user_protos)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    return JSONResponse(result)


@app.post("/api/refine")
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


@app.delete("/api/session/{session_id}")
def clear_session(session_id: str):
    session_store.pop(session_id, None)
    return {"cleared": session_id}
