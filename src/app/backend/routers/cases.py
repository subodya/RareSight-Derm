"""Cases domain: saved scan history and PDF clinical reports."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from .. import db
from ..report import build_case_report

router = APIRouter(prefix="/api/cases", tags=["cases"])


class CaseUpdate(BaseModel):
    """Editable case metadata. Prediction fields are intentionally absent."""
    patient_name: str | None = None
    age: str | None = None
    sex: str | None = None
    localization: str | None = None
    scan_type: str | None = None
    clinical_note: str | None = None
    status: str | None = None


@router.get("")
def list_cases(limit: int = 50):
    return db.list_cases(limit=limit)


@router.get("/{case_id}")
def get_case(case_id: int):
    case = db.get_case(case_id)
    if case is None:
        raise HTTPException(404, "Case not found")
    return case


@router.patch("/{case_id}")
def update_case(case_id: int, body: CaseUpdate):
    fields = body.model_dump(exclude_unset=True)
    if not db.update_case(case_id, fields):
        raise HTTPException(404, "Case not found")
    return db.get_case(case_id)


@router.delete("/{case_id}")
def delete_case(case_id: int):
    if not db.delete_case(case_id):
        raise HTTPException(404, "Case not found")
    return {"deleted": True, "id": case_id}


@router.get("/{case_id}/report")
def case_report(case_id: int):
    case = db.get_case(case_id)
    if case is None:
        raise HTTPException(404, "Case not found")
    pdf_bytes = build_case_report(case)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
                f'attachment; filename="RareSight_Report_{case["case_id"]}.pdf"'
        },
    )
