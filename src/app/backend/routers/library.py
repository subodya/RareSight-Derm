"""Library domain: disease reference encyclopedia backed by model metadata."""

import glob
import json
import os

from fastapi import APIRouter

from ..inference import APP_DIR
from ..state import resources

router = APIRouter(prefix="/api/library", tags=["library"])

ASSETS_DIR = os.path.join(APP_DIR, "assets")
LIBRARY_INFO_PATH = os.path.join(ASSETS_DIR, "library_info.json")

_library_info: dict | None = None


def _load_library_info() -> dict:
    global _library_info
    if _library_info is None:
        with open(LIBRARY_INFO_PATH, encoding="utf-8") as f:
            _library_info = json.load(f)
    return _library_info


def _fix_mojibake(s: str) -> str:
    """disease_metadata.json was written with UTF-8 bytes mis-decoded as cp1252
    (em-dashes render as 'â€"'); round-trip to recover the original text."""
    if "â" not in s:
        return s
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def _image_urls(cls_id: int) -> dict:
    """Gallery from curated library_images (built from HAM10000); falls back to
    the model's reference images so the UI never shows an empty gallery."""
    lib = sorted(glob.glob(os.path.join(ASSETS_DIR, "library_images", f"cls_{cls_id}_*.jpg")))
    if lib:
        urls = [f"/api/images/library_images/{os.path.basename(p)}" for p in lib]
    else:
        refs = sorted(glob.glob(os.path.join(ASSETS_DIR, "reference_images", f"cls_{cls_id}_ref_*.jpg")))
        urls = [f"/api/images/reference_images/{os.path.basename(p)}" for p in refs]
    return {"hero": urls[0] if urls else None, "gallery": urls}


@router.get("/diseases")
def diseases():
    metadata = resources["metadata"]
    info = _load_library_info()
    out = []
    for cls_id_str, meta in sorted(metadata.items(), key=lambda kv: int(kv[0])):
        cls_id = int(cls_id_str)
        extra = info.get(cls_id_str, {})
        images = _image_urls(cls_id)
        out.append({
            "id": cls_id,
            "name": meta["name"],
            "description": _fix_mojibake(meta.get("description", "")),
            "code": extra.get("code", ""),
            "category": extra.get("category", "Benign"),
            "cases": extra.get("cases"),
            "riskFactors": extra.get("riskFactors", ""),
            "ageGroup": extra.get("ageGroup", ""),
            "prevalence": extra.get("prevalence", ""),
            "criteria": extra.get("criteria", []),
            "modality": "Dermoscopy",
            "source": "HAM10000",
            "hero_image": images["hero"],
            "gallery": images["gallery"],
        })
    return out
