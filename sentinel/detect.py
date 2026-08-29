"""Face and eye localisation, with a graceful ladder of backends.

Why this file exists: the prototype modules this project inherited all call
``cv2.data.haarcascades`` directly at import time. OpenCV 5 dropped the bundled
cascade XML files, so on a current install those modules raise before they
analyse anything. Detection is therefore isolated here, behind one interface,
with an explicit fallback ladder:

1. **YuNet** (``cv2.FaceDetectorYN``) if an ONNX model is available. Best
   quality and it returns eye landmarks directly, so no second cascade pass.
2. **Haar cascades** from ``cv2.data`` when OpenCV ships them.
3. **Abstain** -- report that detection is unavailable rather than crashing.

Set ``SENTINEL_YUNET_MODEL`` to a YuNet ONNX path to enable backend 1.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

YUNET_ENV = "SENTINEL_YUNET_MODEL"


@dataclass
class EyeRegion:
    """An axis-aligned eye box in full-image coordinates."""

    x: int
    y: int
    w: int
    h: int
    #: Landmark centre if the backend supplied one, else the box centre.
    cx: float = 0.0
    cy: float = 0.0

    def clamp(self, width: int, height: int) -> "EyeRegion":
        x = max(0, min(width - 1, self.x))
        y = max(0, min(height - 1, self.y))
        w = max(1, min(width - x, self.w))
        h = max(1, min(height - y, self.h))
        return EyeRegion(x, y, w, h, self.cx, self.cy)


@dataclass
class FaceDetection:
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0
    eyes: list[EyeRegion] = field(default_factory=list)
    backend: str = ""

    @property
    def area(self) -> int:
        return self.w * self.h


class DetectorUnavailable(RuntimeError):
    """No detection backend could be initialised on this install."""


class FaceEyeDetector:
    """Locates the most prominent face and its two eyes."""

    def __init__(self, yunet_model: str | None = None) -> None:
        self.backend = "none"
        self._yunet = None
        self._face_cascade = None
        self._eye_cascade = None

        model = yunet_model or os.environ.get(YUNET_ENV)
        if model and os.path.exists(model) and hasattr(cv2, "FaceDetectorYN"):
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    model, "", (320, 320), 0.7, 0.3, 5000
                )
                self.backend = "yunet"
            except cv2.error:
                self._yunet = None

        if self._yunet is None:
            haar = getattr(getattr(cv2, "data", None), "haarcascades", None)
            if haar and os.path.isdir(haar):
                fc = os.path.join(haar, "haarcascade_frontalface_default.xml")
                ec = os.path.join(haar, "haarcascade_eye.xml")
                if os.path.exists(fc) and os.path.exists(ec):
                    face = cv2.CascadeClassifier(fc)
                    eye = cv2.CascadeClassifier(ec)
                    if not face.empty() and not eye.empty():
                        self._face_cascade = face
                        self._eye_cascade = eye
                        self.backend = "haar"

    @property
    def available(self) -> bool:
        return self.backend != "none"

    def detect(self, bgr: np.ndarray) -> FaceDetection | None:
        """Return the largest detected face, or None. Never raises on no-face."""
        if not self.available:
            raise DetectorUnavailable(
                "No face detector available. Install opencv-python-headless<5 "
                f"for bundled Haar cascades, or set {YUNET_ENV} to a YuNet "
                "ONNX model."
            )
        if self.backend == "yunet":
            return self._detect_yunet(bgr)
        return self._detect_haar(bgr)

    # -- backends --------------------------------------------------------
    def _detect_yunet(self, bgr: np.ndarray) -> FaceDetection | None:
        h, w = bgr.shape[:2]
        self._yunet.setInputSize((w, h))
        _, faces = self._yunet.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        # Columns: x, y, w, h, then 5 landmark pairs, then score.
        best = max(faces, key=lambda f: float(f[2]) * float(f[3]))
        fx, fy, fw, fh = (int(round(v)) for v in best[:4])
        eye_span = max(6, int(round(fw * 0.18)))
        eyes: list[EyeRegion] = []
        for i in (0, 1):  # right-eye, left-eye landmarks
            ex, ey = float(best[4 + i * 2]), float(best[5 + i * 2])
            eyes.append(
                EyeRegion(
                    x=int(round(ex - eye_span / 2)),
                    y=int(round(ey - eye_span / 2)),
                    w=eye_span,
                    h=eye_span,
                    cx=ex,
                    cy=ey,
                ).clamp(w, h)
            )
        eyes.sort(key=lambda e: e.x)
        return FaceDetection(fx, fy, fw, fh, float(best[-1]), eyes, "yunet")

    def _detect_haar(self, bgr: np.ndarray) -> FaceDetection | None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        h, w = gray.shape[:2]
        min_face = max(40, int(min(h, w) * 0.18))
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face, min_face)
        )
        if len(faces) == 0:
            return None
        fx, fy, fw, fh = max(faces, key=lambda b: int(b[2]) * int(b[3]))

        # Eyes sit in the upper ~60% of the face box; searching only there
        # removes most nostril/mouth false positives the eye cascade makes.
        roi_h = int(fh * 0.6)
        roi = gray[fy : fy + roi_h, fx : fx + fw]
        raw = self._eye_cascade.detectMultiScale(
            roi,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(max(12, int(fw * 0.10)), max(12, int(fw * 0.10))),
        )
        eyes = [
            EyeRegion(
                x=fx + int(ex),
                y=fy + int(ey),
                w=int(ew),
                h=int(eh),
                cx=fx + ex + ew / 2.0,
                cy=fy + ey + eh / 2.0,
            ).clamp(w, h)
            for (ex, ey, ew, eh) in raw
        ]
        # Keep the two largest, then order left-to-right in viewer space.
        eyes.sort(key=lambda e: e.w * e.h, reverse=True)
        eyes = sorted(eyes[:2], key=lambda e: e.x)
        return FaceDetection(
            int(fx), int(fy), int(fw), int(fh), 1.0, eyes, "haar"
        )


_SHARED: FaceEyeDetector | None = None


def shared_detector() -> FaceEyeDetector:
    """Process-wide detector; cascade loading is slow enough to matter."""
    global _SHARED
    if _SHARED is None:
        _SHARED = FaceEyeDetector()
    return _SHARED
