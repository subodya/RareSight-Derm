"""
RareSight FastAPI backend.

Run:  uvicorn src.app.backend.main:app --reload --port 8000

Domain logic lives in routers/ (scan, dashboard, cases, library, profile);
this module only wires app lifecycle, CORS, static assets, and health.
"""

import os

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import db
from .inference import APP_DIR, load_resources
from .routers import cases, dashboard, library, profile, scan
from .state import resources, session_store

ASSETS_DIR = os.path.join(APP_DIR, "assets")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    resources.update(load_resources())
    yield
    resources.clear()
    session_store.clear()


app = FastAPI(
    title="RareSight API",
    description="Calibrated multimodal dermatology triage for rural GPs",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reference images, curated library images, and case thumbnails by URL.
app.mount("/api/images", StaticFiles(directory=ASSETS_DIR), name="images")

app.include_router(scan.router)
app.include_router(dashboard.router)
app.include_router(cases.router)
app.include_router(library.router)
app.include_router(profile.router)


@app.get("/api/health")
def health():
    serving = resources.get("serving") or {}
    ood_auroc = serving.get("ood_auroc", {})
    return {
        "status": "ok",
        "device": resources.get("device", "unknown"),
        "classes_loaded": len(resources.get("metadata", {})),
        "model": "RareSight (BiomedCLIP + Prototypical Meta-Learning)",
        "metrics": {
            # REAL measured numbers from build_serving_artifacts.py (held-out test).
            "calibration_ece": serving.get("ece_after_test"),
            "calibration_ece_uncalibrated": serving.get("ece_before_test"),
            "metadata_fusion_alpha": serving.get("meta_alpha"),
            "openset_auroc": ood_auroc.get(serving.get("ood_method")) if serving else None,
            "openset_method": serving.get("ood_method"),
        },
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
