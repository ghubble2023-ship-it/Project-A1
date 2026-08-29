"""Lens physics: lateral chromatic aberration must point away from the centre.

Real glass refracts short and long wavelengths by slightly different amounts,
so the red and blue channels of a photograph are very slightly differently
magnified. The visible consequence is colour fringing on high-contrast edges
whose *direction is radial* -- it points along the line from the optical
centre -- and whose magnitude grows with distance from that centre.

A generative model has no lens. It reproduces fringing as a learned local
texture, so its fringes are not radially organised. Measuring the alignment
between the red-vs-blue displacement and the radial direction is therefore a
physics test rather than a texture test.

Fix over the prototype: it matched a 15x15 template inside a 17x17 window,
which can only ever report a displacement of -1, 0 or +1 pixels -- far too
coarse for an effect that is typically well under a pixel. Here the search
window is wider and the correlation peak is refined to sub-pixel accuracy by
fitting a parabola through its neighbours.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..types import SYNTHETIC_HIGH, ModuleReport, Signal

MODULE = "optics"

PATCH = 9  # half-width of the template
SEARCH = 3  # +/- pixels searched around the template position
MIN_SAMPLES = 25  # below this the alignment mean is not trustworthy


def _subpixel_peak(res: np.ndarray) -> tuple[float, float]:
    """Refine an integer correlation argmax to sub-pixel via parabolic fit."""
    _, _, _, max_loc = cv2.minMaxLoc(res)
    mx, my = max_loc
    h, w = res.shape
    dx = dy = 0.0
    if 0 < mx < w - 1:
        a, b, c = float(res[my, mx - 1]), float(res[my, mx]), float(res[my, mx + 1])
        denom = a - 2 * b + c
        if abs(denom) > 1e-12:
            dx = 0.5 * (a - c) / denom
    if 0 < my < h - 1:
        a, b, c = float(res[my - 1, mx]), float(res[my, mx]), float(res[my + 1, mx])
        denom = a - 2 * b + c
        if abs(denom) > 1e-12:
            dy = 0.5 * (a - c) / denom
    return mx + max(-1.0, min(1.0, dx)), my + max(-1.0, min(1.0, dy))


def analyse(bgr: np.ndarray) -> ModuleReport:
    rep = ModuleReport(module=MODULE)
    h, w = bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    diag = math.hypot(cx, cy)

    b_ch, _, r_ch = cv2.split(bgr)
    r_f = r_ch.astype(np.float32)
    b_f = b_ch.astype(np.float32)

    gx = cv2.Sobel(r_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(r_f, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)

    margin = PATCH + SEARCH + 1
    if h <= 2 * margin or w <= 2 * margin:
        rep.signals = [
            Signal.missing(
                "ca_radial_alignment",
                "Chromatic aberration radial alignment",
                "image too small for block matching",
            ),
            Signal.missing(
                "ca_radial_slope",
                "Chromatic aberration growth with radius",
                "image too small for block matching",
            ),
        ]
        return rep

    interior = edge[margin:-margin, margin:-margin]
    thresh = float(np.percentile(interior, 99.0))
    ys, xs = np.nonzero(edge >= thresh)
    keep = (
        (xs >= margin) & (xs < w - margin) & (ys >= margin) & (ys < h - margin)
    )
    xs, ys = xs[keep], ys[keep]

    # Cap the work: 400 well-spread edge points is plenty for a mean.
    if len(xs) > 400:
        idx = np.linspace(0, len(xs) - 1, 400).astype(int)
        xs, ys = xs[idx], ys[idx]

    cosines: list[float] = []
    radii: list[float] = []
    mags: list[float] = []

    for x, y in zip(xs.tolist(), ys.tolist()):
        rx, ry = x - cx, y - cy
        radius = math.hypot(rx, ry)
        if radius < 0.15 * diag:
            # Near the optical axis the radial direction is ill-defined and
            # the true effect is ~zero, so these points only add noise.
            continue

        tmpl = r_f[y - PATCH : y + PATCH + 1, x - PATCH : x + PATCH + 1]
        win = b_f[
            y - PATCH - SEARCH : y + PATCH + SEARCH + 1,
            x - PATCH - SEARCH : x + PATCH + SEARCH + 1,
        ]
        if tmpl.shape[0] < 3 or win.shape[0] < tmpl.shape[0]:
            continue
        if float(tmpl.std()) < 4.0:
            continue

        res = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
        px, py = _subpixel_peak(res)
        dx = px - SEARCH
        dy = py - SEARCH
        mag = math.hypot(dx, dy)
        if mag < 0.02 or mag > SEARCH:
            continue

        cos_sim = ((dx * rx) + (dy * ry)) / (mag * radius)
        cosines.append(abs(cos_sim))
        radii.append(radius / diag)
        mags.append(mag)

    if len(cosines) < MIN_SAMPLES:
        reason = f"only {len(cosines)} usable chromatic edges (need {MIN_SAMPLES})"
        rep.signals = [
            Signal.missing(
                "ca_radial_alignment", "Chromatic aberration radial alignment", reason
            ),
            Signal.missing(
                "ca_radial_slope", "Chromatic aberration growth with radius", reason
            ),
        ]
        return rep

    mean_alignment = float(np.mean(cosines))

    # Real lateral CA grows with radius. Fit magnitude against normalised
    # radius; a real lens gives a positive slope, a texture artefact gives ~0.
    slope = float(np.polyfit(np.array(radii), np.array(mags), 1)[0])

    rep.signals = [
        Signal(
            key="ca_radial_alignment",
            label="Chromatic aberration radial alignment",
            value=round(mean_alignment, 5),
            unit="|cos| 0-1",
            direction=SYNTHETIC_HIGH,
            context={"samples": len(cosines)},
        ),
        Signal(
            key="ca_radial_slope",
            label="Chromatic aberration growth with radius",
            value=round(slope, 6),
            unit="px per normalised radius",
            direction=SYNTHETIC_HIGH,
            context={"samples": len(cosines)},
        ),
    ]
    rep.notes.append(f"{len(cosines)} edge samples used for lens dispersion.")
    return rep
