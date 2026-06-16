"""SQLite persistence for scan cases.

Single-table store; stdlib sqlite3 keeps the deployment dependency-free.
DB lives next to the model assets so a fresh clone starts empty and the
seed script can populate it through the normal /api/predict path.
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

from .inference import APP_DIR

DB_PATH = os.path.join(APP_DIR, "assets", "raresight.db")
UPLOADS_DIR = os.path.join(APP_DIR, "assets", "case_uploads")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    patient_name        TEXT,
    age                 TEXT,
    sex                 TEXT,
    localization        TEXT,
    scan_type           TEXT,
    clinical_note       TEXT,
    top_class_id        INTEGER,
    top_class_name      TEXT,
    confidence          REAL,
    risk                TEXT,
    refer_to_specialist INTEGER,
    is_unknown          INTEGER,
    status              TEXT DEFAULT 'pending',
    thumbnail           TEXT,
    result_json         TEXT
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    with _connect() as conn:
        conn.execute(_SCHEMA)


def display_id(row_id: int) -> str:
    return f"DX-{10000 + row_id}"


def insert_case(*, patient: dict, result: dict, thumbnail: str,
                created_at: str | None = None) -> int:
    top = (result.get("predictions") or [{}])[0]
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO cases
               (created_at, patient_name, age, sex, localization, scan_type,
                clinical_note, top_class_id, top_class_name, confidence, risk,
                refer_to_specialist, is_unknown, status, thumbnail, result_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                created_at or datetime.now(timezone.utc).isoformat(),
                patient.get("name"),
                patient.get("age"),
                patient.get("sex"),
                patient.get("localization"),
                patient.get("scan_type"),
                patient.get("clinical_note"),
                result.get("top_class_id"),
                result.get("top_class_name"),
                float(top.get("probability") or 0.0),
                top.get("risk", "low"),
                int(bool(result.get("refer_to_specialist"))),
                int(bool(result.get("is_unknown"))),
                "referred" if result.get("refer_to_specialist") else "pending",
                thumbnail,
                json.dumps(result),
            ),
        )
        return cur.lastrowid


# Columns a clinician may edit from the UI. Deliberately excludes the
# prediction fields (top_class_name, confidence, risk, …) and result_json so
# the saved AI evidence and the PDF report can never drift from the model output.
EDITABLE_FIELDS = (
    "patient_name", "age", "sex", "localization", "scan_type",
    "clinical_note", "status",
)


def update_case(case_id: int, fields: dict) -> bool:
    cols = {k: v for k, v in fields.items() if k in EDITABLE_FIELDS}
    with _connect() as conn:
        if not cols:  # nothing editable supplied — succeed iff the case exists
            return conn.execute(
                "SELECT 1 FROM cases WHERE id = ?", (case_id,)
            ).fetchone() is not None
        assignments = ", ".join(f"{k} = ?" for k in cols)
        cur = conn.execute(
            f"UPDATE cases SET {assignments} WHERE id = ?",
            (*cols.values(), case_id),
        )
        return cur.rowcount > 0


def delete_case(case_id: int) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT thumbnail FROM cases WHERE id = ?", (case_id,)
        ).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM cases WHERE id = ?", (case_id,))
    if row["thumbnail"]:
        try:
            os.remove(os.path.join(UPLOADS_DIR, row["thumbnail"]))
        except OSError:
            pass
    return True


def _row_summary(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "case_id": display_id(r["id"]),
        "created_at": r["created_at"],
        "patient_name": r["patient_name"],
        "age": r["age"],
        "sex": r["sex"],
        "localization": r["localization"],
        "scan_type": r["scan_type"],
        "top_class_name": r["top_class_name"],
        "confidence": r["confidence"],
        "risk": r["risk"],
        "refer_to_specialist": bool(r["refer_to_specialist"]),
        "is_unknown": bool(r["is_unknown"]),
        "status": r["status"],
        "thumbnail_url": f"/api/images/case_uploads/{r['thumbnail']}" if r["thumbnail"] else None,
    }


def list_cases(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM cases ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_summary(r) for r in rows]


def get_case(case_id: int) -> dict | None:
    with _connect() as conn:
        r = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    if r is None:
        return None
    full = _row_summary(r)
    full["clinical_note"] = r["clinical_note"]
    full["result"] = json.loads(r["result_json"]) if r["result_json"] else None
    return full


def stats() -> dict:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        today = conn.execute(
            "SELECT COUNT(*) FROM cases WHERE date(created_at) = date('now')"
        ).fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM cases WHERE status = 'pending'"
        ).fetchone()[0]
        urgent = conn.execute(
            "SELECT COUNT(*) FROM cases WHERE status = 'pending' AND risk = 'high'"
        ).fetchone()[0]
        referred = conn.execute(
            "SELECT COUNT(*) FROM cases WHERE refer_to_specialist = 1"
        ).fetchone()[0]
        avg_conf = conn.execute("SELECT AVG(confidence) FROM cases").fetchone()[0]
    return {
        "total_scans": total,
        "today_scans": today,
        "pending_review": pending,
        "urgent_pending": urgent,
        "referrals": referred,
        "avg_confidence": float(avg_conf) if avg_conf is not None else None,
    }


def day_counts(days: int = 14) -> list[dict]:
    """Per-day scan / flagged counts for the dashboard chart, oldest first."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT date(created_at) AS d,
                      COUNT(*) AS scans,
                      SUM(refer_to_specialist) AS flags
               FROM cases
               WHERE date(created_at) >= date('now', ?)
               GROUP BY d""",
            (f"-{days - 1} days",),
        ).fetchall()
    by_day = {r["d"]: (r["scans"], r["flags"] or 0) for r in rows}
    out = []
    today = datetime.now(timezone.utc).date()
    for i in range(days - 1, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        scans, flags = by_day.get(d, (0, 0))
        out.append({"date": d, "scans": scans, "flags": flags})
    return out
