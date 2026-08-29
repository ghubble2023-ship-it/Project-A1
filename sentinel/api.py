"""HTTP surface.

Run with::

    uvicorn sentinel.api:app --host 0.0.0.0 --port 8000

Two primary endpoints (``/analyze/image``, ``/analyze/chat``) plus one
compatibility endpoint. The compatibility route exists because a Next.js
console and an Android client were already written against the older
``/forensics/ocular-comparator`` contract; it maps the new calibrated output
back onto that response shape so the existing front-ends keep working while
they migrate.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import __version__
from .detect import shared_detector
from .fusion import Calibration
from .image.pipeline import analyse_image
from .text.pipeline import analyse_conversation

app = FastAPI(
    title="Sentinel -- Project A1 forensic core",
    version=__version__,
    description=(
        "Explainable image and conversation forensics. Every image score is "
        "produced by a calibration fitted on labelled data and reports the "
        "evidence behind it; every conversation score exposes its rubric."
    ),
)

_CALIBRATION = Calibration.load()

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class Message(BaseModel):
    sender: str | None = Field(default="suspect")
    text: str


class ChatPayload(BaseModel):
    messages: list[Message]
    suspect: str | None = Field(
        default="suspect",
        description="Sender id to analyse. Null analyses every message.",
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "face_detector": shared_detector().backend,
        "calibrated_signals": sorted(_CALIBRATION.entries),
        "calibration_meta": _CALIBRATION.meta.get("view"),
    }


@app.get("/calibration")
def calibration() -> dict[str, Any]:
    """Full calibration, including each signal's measured AUC and weight.

    Exposed deliberately: a forensic score whose thresholds are secret is not
    reviewable, and this system's central claim is that its thresholds were
    measured rather than chosen.
    """
    return {
        "meta": _CALIBRATION.meta,
        "prior_log_odds": _CALIBRATION.prior_log_odds,
        "signals": {
            k: {
                "auc": c.auc,
                "weight": round(c.weight, 4),
                "mu_authentic": c.mu_authentic,
                "sd_authentic": c.sd_authentic,
                "mu_synthetic": c.mu_synthetic,
                "sd_synthetic": c.sd_synthetic,
                "n_authentic": c.n_authentic,
                "n_synthetic": c.n_synthetic,
            }
            for k, c in _CALIBRATION.entries.items()
        },
    }


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
        )
    return data


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await _read_upload(file)
    try:
        return analyse_image(data, _CALIBRATION)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/analyze/chat")
def analyze_chat(payload: ChatPayload) -> dict[str, Any]:
    try:
        return analyse_conversation(
            [m.model_dump() for m in payload.messages], payload.suspect
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/forensics/ocular-comparator")
async def ocular_comparator_compat(file: UploadFile = File(...)) -> dict[str, Any]:
    """Compatibility shim for the pre-existing web and Android clients."""
    data = await _read_upload(file)
    try:
        result = analyse_image(data, _CALIBRATION)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ocular = next(
        (m for m in result["modules"] if m["module"] == "ocular"), {"signals": []}
    )
    by_key = {s["key"]: s for s in ocular["signals"]}

    def val(key: str) -> float:
        s = by_key.get(key) or {}
        return s.get("value") if s.get("available") else None

    verdict = result["verdict"]
    exp = result["explanation"]
    return {
        "ocular_consistency_forensics": {
            "light_vector_angular_divergence_deg": val("ocular_angle_divergence"),
            "catch_light_offset_disparity": val("ocular_offset_disparity"),
            "circularity_difference": val("ocular_circularity_delta"),
            "glint_area_ratio": val("ocular_area_ratio"),
            "environment_dissimilarity": val("corneal_env_dissimilarity"),
            "anomaly_score": verdict["p_synthetic"],
            "verdict": verdict["label"],
            # None, not a default, when the screen-capture gate refused to score.
            # Older clients treat this field as a tri-state; inventing True or
            # False here would put a fabricated judgement back into the response
            # the gate exists to prevent.
            "is_authentic_geometry": (
                None
                if verdict["p_synthetic"] is None
                else verdict["p_synthetic"] < 0.5
            ),
        },
        "dakota_report": {
            "analyst_persona": "Dakota",
            "verdict": verdict["label"],
            "discrepancy_count": len(exp["evidence_for_synthetic"]),
            "flagged_discrepancies": [
                e["headline"] for e in exp["evidence_for_synthetic"]
            ],
            "detailed_physical_breakdown": [
                e["finding"] for e in exp["evidence_for_synthetic"]
            ],
            "supporting_authentic_findings": [
                e["finding"] for e in exp["evidence_for_authentic"]
            ],
            "executive_summary": exp["summary"],
            "caveats": exp["caveats"],
        },
        "full_report": result,
    }
