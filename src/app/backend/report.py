"""Clinical PDF report generation for saved cases (reportlab / platypus)."""

import io
import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (HRFlowable, Image as RLImage, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from . import db
from .inference import APP_DIR

NAVY = colors.HexColor("#1E3A8A")
INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#6B7280")
RISK_RED = colors.HexColor("#B42318")
OK_GREEN = colors.HexColor("#15803D")
BG_GREY = colors.HexColor("#F3F4F6")

LOGO_PATH = os.path.join(APP_DIR, "frontend", "public", "raresight-logo.png")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=18,
                                textColor=NAVY, spaceAfter=2, alignment=TA_CENTER),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=9,
                                   textColor=MUTED, alignment=TA_CENTER),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=12,
                             textColor=NAVY, spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=9.5,
                               textColor=INK, leading=14),
        "small": ParagraphStyle("small", parent=base["Normal"], fontSize=8,
                                textColor=MUTED, leading=11),
        "verdict": ParagraphStyle("verdict", parent=base["Normal"], fontSize=13,
                                  textColor=INK, leading=18),
    }


def _kv_table(rows, col_widths=(45 * mm, 115 * mm)):
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("TEXTCOLOR", (1, 0), (1, -1), INK),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, BG_GREY),
    ]))
    return t


def build_case_report(case: dict) -> bytes:
    """Render a one-to-two page clinical decision-support report for a case."""
    s = _styles()
    result = case.get("result") or {}
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=f"RareSight Clinical Report {case['case_id']}",
    )
    story = []

    # ── Header ──
    if os.path.exists(LOGO_PATH):
        story.append(RLImage(LOGO_PATH, width=52 * mm, height=19 * mm, kind="proportional"))
        story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("Clinical Decision-Support Report", s["title"]))
    story.append(Paragraph(
        "RareSight · Calibrated multimodal dermatology triage (BiomedCLIP + prototypical meta-learning)",
        s["subtitle"]))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY))
    story.append(Spacer(1, 4 * mm))

    # ── Case + patient details ──
    created = case.get("created_at", "")
    try:
        created = datetime.fromisoformat(created).strftime("%d %B %Y, %H:%M UTC")
    except ValueError:
        pass
    story.append(Paragraph("Case Details", s["h2"]))
    story.append(_kv_table([
        ["Case ID", case["case_id"]],
        ["Date analyzed", created],
        ["Patient", case.get("patient_name") or "—"],
        ["Age / Sex", f"{case.get('age') or '—'} / {case.get('sex') or '—'}"],
        ["Anatomical site", (case.get("localization") or "—").title()],
        ["Modality", case.get("scan_type") or "Dermoscopy"],
        ["Clinical note", case.get("clinical_note") or "—"],
    ]))

    # ── Lesion image + top differential side by side ──
    story.append(Paragraph("AI Assessment", s["h2"]))

    refer = bool(case.get("refer_to_specialist"))
    verdict_color = "#B42318" if refer else "#15803D"
    recommendation = ("Refer to specialist (Dermatology)" if refer
                      else "Routine follow-up (6-month review)")
    conf = case.get("confidence") or 0.0
    assessment = Paragraph(
        f'<b>Top differential:</b> <font color="{NAVY}"><b>{case.get("top_class_name") or "—"}</b></font><br/>'
        f'<b>Confidence (calibrated):</b> {conf * 100:.1f}%<br/>'
        f'<b>Recommendation:</b> <font color="{verdict_color}"><b>{recommendation}</b></font>',
        s["verdict"])

    img_path = os.path.join(db.UPLOADS_DIR, case["thumbnail_url"].split("/")[-1]) \
        if case.get("thumbnail_url") else None
    if img_path and os.path.exists(img_path):
        lesion = RLImage(img_path, width=55 * mm, height=42 * mm, kind="proportional")
        row = Table([[lesion, assessment]], colWidths=[60 * mm, 100 * mm])
        row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(row)
    else:
        story.append(assessment)

    # ── Probability table ──
    preds = result.get("predictions") or []
    if preds:
        story.append(Paragraph("Differential Probabilities", s["h2"]))
        rows = [["Rank", "Condition", "Probability", "Risk level"]]
        for p in preds:
            rows.append([
                str(p.get("rank", "")),
                p.get("class_name", ""),
                f"{(p.get('probability') or 0) * 100:.1f}%",
                (p.get("risk") or "").capitalize(),
            ])
        t = Table(rows, colWidths=[15 * mm, 75 * mm, 30 * mm, 40 * mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_GREY]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E5E7EB")),
        ]))
        story.append(t)

    # ── Model evidence / reliability ──
    story.append(Paragraph("Model Evidence & Reliability", s["h2"]))
    ece = result.get("calibration_ece")
    meta_used = result.get("metadata_used") or []
    evidence_rows = [
        ["Predictive entropy", f"{result.get('entropy', 0):.3f} (lower = more certain)"],
        ["Calibration ECE", f"{ece:.3f} (held-out test)" if ece is not None else "—"],
        ["Patient metadata fused", ", ".join(meta_used) if meta_used else "None provided"],
        ["Clinical note evidence",
         f"Supports {result['note_supports']} (BiomedCLIP text match)"
         if result.get("note_used") else "Not used"],
        ["Out-of-distribution check",
         "FLAGGED — image may show an unknown condition" if result.get("is_unknown")
         else "Passed — image consistent with known classes"],
    ]
    if result.get("modality_warning"):
        evidence_rows.append(["Modality warning", result["modality_warning"]])
    story.append(_kv_table(evidence_rows))

    # ── Disclaimer ──
    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(width="100%", thickness=0.6, color=MUTED))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "This report was generated by RareSight, an AI clinical decision-support system, "
        "and is intended to assist — not replace — clinical judgement. The output is a "
        "calibrated statistical estimate, not a diagnosis. Final diagnosis and management "
        "decisions must be made by a qualified clinician, with histopathological "
        "confirmation where indicated.", s["small"]))

    doc.build(story)
    return buf.getvalue()
