"""Profile domain: clinician identity + live usage statistics."""

import json
import os

from fastapi import APIRouter

from .. import db
from ..inference import APP_DIR

router = APIRouter(prefix="/api", tags=["profile"])

PROFILE_PATH = os.path.join(APP_DIR, "assets", "profile.json")


@router.get("/profile")
def profile():
    with open(PROFILE_PATH, encoding="utf-8") as f:
        info = json.load(f)
    s = db.stats()
    info["stats"] = {
        "scans": s["total_scans"],
        "confirmed": s["total_scans"] - s["pending_review"],
        "referrals": s["referrals"],
        "avg_confidence": s["avg_confidence"],
    }
    return info
