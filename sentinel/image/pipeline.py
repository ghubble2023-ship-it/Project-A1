"""Runs every image module and fuses their signals into one verdict."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..detect import FaceEyeDetector, shared_detector
from ..explain import explain
from ..fusion import Calibration
from ..types import ModuleReport
from . import ocular, optics, provenance, screen, spectral


def decode(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image; unsupported or corrupt file.")
    return img


def run_modules(
    data: bytes,
    bgr: np.ndarray | None = None,
    detector: FaceEyeDetector | None = None,
) -> list[ModuleReport]:
    """Run every module, isolating failures so one crash cannot lose the rest."""
    bgr = decode(data) if bgr is None else bgr
    det = detector or shared_detector()

    reports: list[ModuleReport] = []
    for name, fn in (
        ("ocular", lambda: ocular.analyse(bgr, det)),
        ("optics", lambda: optics.analyse(bgr)),
        ("spectral", lambda: spectral.analyse(bgr)),
        ("provenance", lambda: provenance.analyse(data)),
    ):
        try:
            reports.append(fn())
        except Exception as exc:  # pragma: no cover - defensive
            reports.append(ModuleReport(module=name, error=f"{type(exc).__name__}: {exc}"))
    return reports


def analyse_image(
    data: bytes,
    calibration: Calibration | None = None,
    detector: FaceEyeDetector | None = None,
) -> dict[str, Any]:
    cal = calibration or Calibration.load()
    bgr = decode(data)
    media = {
        "width": int(bgr.shape[1]),
        "height": int(bgr.shape[0]),
        "bytes": len(data),
    }

    # Gate before evidence. Every signal below is a statement about a capture
    # pipeline; a screenshot does not have one, so there is nothing to weigh and
    # the honest output is a refusal rather than a number. See image/screen.py.
    screen_report = screen.detect(bgr)
    media["screen_capture"] = screen_report.as_dict()
    if screen_report.is_screen_capture:
        return {
            "media": media,
            "verdict": {
                "label": "INCONCLUSIVE_NOT_A_PHOTOGRAPH",
                "p_synthetic": None,
                "total_llr": None,
                "coverage": 0.0,
                "evidence": [],
            },
            "explanation": {
                "verdict": "INCONCLUSIVE_NOT_A_PHOTOGRAPH",
                "probability_synthetic": None,
                "coverage": 0.0,
                "summary": (
                    "This looks like a screen capture, not a photograph, so no "
                    "authenticity judgement is possible. "
                    + screen_report.reason
                    + "."
                ),
                "evidence_for_synthetic": [],
                "evidence_for_authentic": [],
                "not_measured": [],
                "caveats": [
                    "Every measurement this system makes describes how a camera "
                    "recorded a scene: sensor noise, lens dispersion, compression "
                    "history. A screenshot was drawn by the phone, so those traces "
                    "belong to the display, not to the original image.",
                    "This is not a statement that the underlying image is fake. It "
                    "is a statement that a screenshot cannot answer the question. "
                    "To get a verdict, supply the original image file rather than a "
                    "picture of it on screen.",
                    "The detector is tuned to avoid ever refusing a genuine "
                    "photograph: on the evaluation set it flagged 16 of 17 "
                    "screenshots and 0 of 80 photographs.",
                ],
            },
            "modules": [],
        }

    reports = run_modules(data, bgr, detector)
    signals = [s for r in reports for s in r.signals]
    verdict = cal.score(signals)

    return {
        "media": media,
        "verdict": verdict.to_dict(),
        "explanation": explain(verdict, reports),
        "modules": [r.to_dict() for r in reports],
    }
