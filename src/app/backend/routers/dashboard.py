"""Dashboard domain: KPIs, recent cases, activity feed, scan-volume chart."""

from fastapi import APIRouter

from .. import db

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


def _activity_from_cases(cases: list[dict], limit: int = 8) -> list[dict]:
    events = []
    for c in cases:
        who = c["patient_name"] or c["case_id"]
        events.append({
            "time": c["created_at"],
            "action": "New scan analyzed",
            "meta": f"{c['case_id']} · {who} · {c['top_class_name']}",
            "kind": "upload",
        })
        if c["refer_to_specialist"]:
            events.append({
                "time": c["created_at"],
                "action": "Specialist referral suggested",
                "meta": f"{c['case_id']} → Dermatology",
                "kind": "refer",
            })
        if c["is_unknown"]:
            events.append({
                "time": c["created_at"],
                "action": "Out-of-distribution case flagged",
                "meta": f"{c['case_id']} · manual review required",
                "kind": "system",
            })
    events.sort(key=lambda e: e["time"], reverse=True)
    return events[:limit]


@router.get("/summary")
def summary():
    recent = db.list_cases(limit=6)
    return {
        "stats": db.stats(),
        "recent_cases": recent,
        "activity": _activity_from_cases(db.list_cases(limit=12)),
        "day_counts": db.day_counts(14),
    }
