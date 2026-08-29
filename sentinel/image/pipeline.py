"""Runs every image module and fuses their signals into one verdict."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..detect import FaceEyeDetector, shared_detector
from ..explain import explain
from ..fusion import Calibration
from ..types import ModuleReport
from . import ocular, optics, provenance, spectral


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
    reports = run_modules(data, bgr, detector)

    signals = [s for r in reports for s in r.signals]
    verdict = cal.score(signals)

    return {
        "media": {
            "width": int(bgr.shape[1]),
            "height": int(bgr.shape[0]),
            "bytes": len(data),
        },
        "verdict": verdict.to_dict(),
        "explanation": explain(verdict, reports),
        "modules": [r.to_dict() for r in reports],
    }
