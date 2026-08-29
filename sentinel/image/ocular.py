"""Ocular physics: catch-light geometry and cross-eye environment parity.

The premise, which is sound: a cornea is a convex mirror. Two corneas on one
face see the same room from almost the same place, so under a single dominant
light the specular highlight ("catch light") must appear at nearly the same
position on each cornea, at nearly the same size and shape. Generative models
paint each eye from local context and routinely disagree.

Two engineering points the prototype got wrong, fixed here:

* **Everything is normalised by eye width.** The prototype measured glint
  displacement in raw pixels, which makes the same face score differently at
  1024px and 256px. Every distance below is a fraction of the eye box.
* **Sub-pixel centroids.** Taking ``argmax`` of a blurred patch quantises the
  angle badly on small eyes; an intensity-weighted centroid over the highlight
  mask is stable to a fraction of a pixel.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..detect import DetectorUnavailable, EyeRegion, FaceEyeDetector, shared_detector
from ..types import SYNTHETIC_HIGH, ModuleReport, Signal

MODULE = "ocular"

#: Eye patches are resampled to this square before measurement, so that a
#: 20px eye and a 200px eye produce comparable shape statistics.
NORM_DIM = 128

#: Highlight blobs smaller than this (in NORM_DIM space) are unmeasurable.
MIN_HIGHLIGHT_AREA = 4.0


def _weighted_centroid(patch: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Intensity-weighted centroid of `patch` within `mask`, in patch coords."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return float(patch.shape[1]) / 2.0, float(patch.shape[0]) / 2.0
    w = patch[ys, xs].astype(np.float64)
    total = float(w.sum())
    if total <= 0:
        return float(xs.mean()), float(ys.mean())
    return float((xs * w).sum() / total), float((ys * w).sum() / total)


def _analyse_eye(bgr: np.ndarray, eye: EyeRegion) -> dict | None:
    """Measure one eye. Returns None if the patch is unusable."""
    patch = bgr[eye.y : eye.y + eye.h, eye.x : eye.x + eye.w]
    if patch.size == 0 or min(patch.shape[:2]) < 8:
        return None

    norm = cv2.resize(patch, (NORM_DIM, NORM_DIM), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)

    peak = float(gray.max())
    floor = float(gray.min())
    if peak - floor < 12.0:
        # A flat patch (closed eye, sunglasses, motion blur) has no geometry
        # to measure. Better to say so than to measure noise.
        return None

    # Highlight = the top slice of the intensity range within this eye. Using a
    # relative threshold rather than a fixed 190 keeps dark and bright
    # exposures comparable.
    thresh = floor + 0.82 * (peak - floor)
    mask = (gray >= thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    blob = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(blob))
    perim = float(cv2.arcLength(blob, True))
    circularity = (
        min(1.0, 4.0 * math.pi * area / (perim * perim)) if perim > 0 else 0.0
    )
    if area < MIN_HIGHLIGHT_AREA:
        # A one- or two-pixel blob has no reliable shape or centroid. Its
        # area ratio against the other eye explodes to meaningless numbers,
        # so abstain rather than emit noise.
        return None

    blob_mask = np.zeros_like(gray)
    cv2.drawContours(blob_mask, [blob], -1, 255, -1)
    gx, gy = _weighted_centroid(gray, blob_mask > 0)

    # Pupil proxy: darkest region of the blurred patch, again as a centroid.
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    dark_thresh = float(blurred.min()) + 0.18 * (float(blurred.max()) - float(blurred.min()))
    dark_mask = blurred <= dark_thresh
    inv = (255 - blurred).astype(np.uint8)
    px, py = _weighted_centroid(inv, dark_mask)

    dx, dy = gx - px, gy - py
    return {
        "glint_xy": (gx, gy),
        "pupil_xy": (px, py),
        # Offset as a fraction of eye width -- scale invariant.
        "offset_frac": math.hypot(dx, dy) / NORM_DIM,
        "angle_deg": math.degrees(math.atan2(dy, dx)),
        # Highlight area as a fraction of the eye patch -- scale invariant.
        "area_frac": area / float(NORM_DIM * NORM_DIM),
        "circularity": circularity,
        "peak_luma": peak,
        "contrast": peak - floor,
        "norm_bgr": norm,
        "mask": mask,
    }


def _panorama(norm_bgr: np.ndarray) -> np.ndarray:
    """Unwrap a corneal disc to a polar strip.

    A true inverse-parabolic reflection map needs the eye's 3D pose, which a
    2D crop does not give us. A log-polar unwrap about the eye centre captures
    the same thing this test actually needs -- the angular arrangement of what
    the cornea reflects -- without pretending to a calibration we do not have.
    """
    c = NORM_DIM / 2.0
    return cv2.warpPolar(
        norm_bgr,
        (128, 64),
        (c, c),
        c * 0.9,
        cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR,
    )


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    va = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    vb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    vab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    smap = ((2 * mu_a * mu_b + c1) * (2 * vab + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (va + vb + c2)
    )
    return float(np.mean(smap))


def _all_missing(reason: str) -> list[Signal]:
    return [
        Signal.missing(
            "ocular_angle_divergence",
            "Catch-light angular divergence between eyes",
            reason,
            unit="deg",
        ),
        Signal.missing(
            "ocular_offset_disparity",
            "Catch-light offset disparity between eyes",
            reason,
            unit="fraction of eye width",
        ),
        Signal.missing(
            "ocular_circularity_delta",
            "Catch-light shape mismatch between eyes",
            reason,
        ),
        Signal.missing(
            "ocular_area_ratio",
            "Catch-light size ratio between eyes",
            reason,
        ),
        Signal.missing(
            "corneal_env_dissimilarity",
            "Reflected-environment mismatch between eyes",
            reason,
        ),
    ]


def analyse(bgr: np.ndarray, detector: FaceEyeDetector | None = None) -> ModuleReport:
    rep = ModuleReport(module=MODULE)
    det = detector or shared_detector()

    try:
        face = det.detect(bgr)
    except DetectorUnavailable as exc:
        rep.error = str(exc)
        rep.signals = _all_missing("no face detector available")
        return rep

    if face is None:
        rep.signals = _all_missing("no face detected")
        rep.notes.append("No face found; ocular physics not applicable.")
        return rep

    if len(face.eyes) < 2:
        rep.signals = _all_missing(
            f"needs two visible eyes, found {len(face.eyes)}"
        )
        rep.notes.append(
            "Fewer than two eyes localised (occlusion, eyewear, or profile pose)."
        )
        return rep

    left = _analyse_eye(bgr, face.eyes[0])
    right = _analyse_eye(bgr, face.eyes[1])
    if left is None or right is None:
        rep.signals = _all_missing("eye patch had no measurable highlight")
        return rep

    angle_delta = abs(left["angle_deg"] - right["angle_deg"])
    if angle_delta > 180.0:
        angle_delta = 360.0 - angle_delta

    offset_delta = abs(left["offset_frac"] - right["offset_frac"])
    circ_delta = abs(left["circularity"] - right["circularity"])

    a1, a2 = left["area_frac"], right["area_frac"]
    lo, hi = min(a1, a2), max(a1, a2)
    # Ratio in [1, inf); 1.0 means the two highlights are the same size.
    area_ratio = (hi / lo) if lo > 1e-9 else 50.0

    pano_l = cv2.cvtColor(_panorama(left["norm_bgr"]), cv2.COLOR_BGR2GRAY)
    pano_r = cv2.cvtColor(_panorama(right["norm_bgr"]), cv2.COLOR_BGR2GRAY)
    env_dissimilarity = 1.0 - _ssim(pano_l, pano_r)

    rep.signals = [
        Signal(
            key="ocular_angle_divergence",
            label="Catch-light angular divergence between eyes",
            value=round(angle_delta, 3),
            unit="deg",
            direction=SYNTHETIC_HIGH,
            context={
                "left_angle_deg": round(left["angle_deg"], 2),
                "right_angle_deg": round(right["angle_deg"], 2),
            },
        ),
        Signal(
            key="ocular_offset_disparity",
            label="Catch-light offset disparity between eyes",
            value=round(offset_delta, 5),
            unit="fraction of eye width",
            direction=SYNTHETIC_HIGH,
            context={
                "left_offset_frac": round(left["offset_frac"], 4),
                "right_offset_frac": round(right["offset_frac"], 4),
            },
        ),
        Signal(
            key="ocular_circularity_delta",
            label="Catch-light shape mismatch between eyes",
            value=round(circ_delta, 5),
            direction=SYNTHETIC_HIGH,
            context={
                "left_circularity": round(left["circularity"], 4),
                "right_circularity": round(right["circularity"], 4),
            },
        ),
        Signal(
            key="ocular_area_ratio",
            label="Catch-light size ratio between eyes",
            value=round(area_ratio, 4),
            unit="x",
            direction=SYNTHETIC_HIGH,
            context={
                "left_area_frac": round(a1, 6),
                "right_area_frac": round(a2, 6),
            },
        ),
        Signal(
            key="corneal_env_dissimilarity",
            label="Reflected-environment mismatch between eyes",
            value=round(env_dissimilarity, 5),
            direction=SYNTHETIC_HIGH,
        ),
    ]
    rep.notes.append(f"Face via {face.backend} backend; two eyes measured.")
    return rep
