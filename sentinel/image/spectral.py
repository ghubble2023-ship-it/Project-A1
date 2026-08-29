"""Sensor and codec fingerprints in the frequency and residual domains.

Three independent things live here:

``upsampling_peak_prominence``
    Transposed convolutions and pixel-shuffle upsamplers leave a periodic
    checkerboard, which in the 2D spectrum is a small set of bright spikes at
    fixed frequencies. The prototype counted pixels above ``mean + 3.8 sigma``
    of the whole spectrum, which mostly counts *natural* low-frequency
    structure and scales with image size. Here the spectrum is first flattened
    by dividing out its radial average, so what remains is only what is
    anomalous *for its own frequency band*, and the statistic is a
    resolution-independent prominence ratio.

``jpeg_blockiness``
    Energy at the 8-pixel grid boundaries relative to its neighbours. High
    values mean heavy or repeated JPEG compression -- useful context, and a
    known confounder that must be reported rather than hidden.

``residual_energy`` / ``residual_kurtosis``
    A camera sensor adds broadband photon and read noise. Diffusion output is
    smooth at the pixel level; its high-pass residual is unusually low-energy
    and unusually heavy-tailed, because what remains is edges rather than
    grain.

Every statistic here is computed on a fixed-size centre crop and normalised,
because raw resolution is the single strongest confounder in this problem
space and must not be allowed to leak in through the back door.
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy.fft import fft2, fftshift

from ..types import SYNTHETIC_HIGH, ModuleReport, Signal

MODULE = "spectral"

#: All spectral work happens on this square, so results do not depend on the
#: source resolution.
ANALYSIS_DIM = 256


def _centre_crop(gray: np.ndarray, dim: int) -> np.ndarray | None:
    h, w = gray.shape[:2]
    if h < dim or w < dim:
        # Upscaling to reach the analysis size would fabricate the very
        # high-frequency content we are about to measure.
        return None
    y0, x0 = (h - dim) // 2, (w - dim) // 2
    return gray[y0 : y0 + dim, x0 : x0 + dim]


def _radial_flatten(mag: np.ndarray) -> np.ndarray:
    """Divide the spectrum by its own radial average.

    Natural images have a steep 1/f falloff. Without removing it, any
    "unusually bright" test simply rediscovers the DC corner. After flattening,
    a value of 2.0 means "twice as bright as other frequencies at this radius".
    """
    n = mag.shape[0]
    cy = cx = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    nbins = int(r.max()) + 1
    total = np.bincount(r.ravel(), weights=mag.ravel(), minlength=nbins)
    count = np.bincount(r.ravel(), minlength=nbins)
    mean = total / np.maximum(count, 1)
    mean[mean <= 1e-9] = 1e-9
    return mag / mean[r]


def _jpeg_blockiness(gray: np.ndarray) -> float:
    """Ratio of gradient energy on the 8px grid to gradient energy off it."""
    g = gray.astype(np.float32)
    dh = np.abs(np.diff(g, axis=1))
    dv = np.abs(np.diff(g, axis=0))

    cols = np.arange(dh.shape[1])
    rows = np.arange(dv.shape[0])
    on_h = dh[:, (cols % 8) == 7]
    off_h = dh[:, (cols % 8) != 7]
    on_v = dv[(rows % 8) == 7, :]
    off_v = dv[(rows % 8) != 7, :]

    on = float(on_h.mean() + on_v.mean())
    off = float(off_h.mean() + off_v.mean())
    return on / off if off > 1e-9 else 1.0


def analyse(bgr: np.ndarray) -> ModuleReport:
    rep = ModuleReport(module=MODULE)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    crop = _centre_crop(gray, ANALYSIS_DIM)

    if crop is None:
        reason = (
            f"image smaller than the {ANALYSIS_DIM}px analysis window "
            f"({gray.shape[1]}x{gray.shape[0]})"
        )
        rep.signals = [
            Signal.missing("upsampling_peak_prominence", "Upsampler grid spikes", reason),
            Signal.missing("jpeg_blockiness", "JPEG 8px grid strength", reason),
            Signal.missing("residual_energy", "High-frequency residual energy", reason),
            Signal.missing("residual_kurtosis", "High-frequency residual kurtosis", reason),
        ]
        rep.notes.append(reason)
        return rep

    win = np.outer(np.hanning(ANALYSIS_DIM), np.hanning(ANALYSIS_DIM))
    spec = np.abs(fftshift(fft2(crop.astype(np.float64) * win)))
    flat = _radial_flatten(spec)

    # Ignore the DC neighbourhood; it is structure, not periodicity.
    n = flat.shape[0]
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    rad = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    band = (rad > n * 0.08) & (rad < n * 0.48)
    vals = flat[band]
    # Prominence: how far the brightest few frequencies stand above the
    # typical one. Scale-free by construction.
    prominence = float(np.percentile(vals, 99.9) / max(np.median(vals), 1e-9))

    blockiness = _jpeg_blockiness(crop)

    # High-pass residual: what a sensor's grain lives in.
    blur = cv2.GaussianBlur(crop.astype(np.float32), (0, 0), 1.2)
    resid = crop.astype(np.float32) - blur
    energy = float(np.sqrt(np.mean(resid**2)))
    sd = float(resid.std())
    if sd > 1e-9:
        z = resid / sd
        kurt = float(np.mean(z**4) - 3.0)
    else:
        kurt = 0.0

    rep.signals = [
        Signal(
            key="upsampling_peak_prominence",
            label="Upsampler grid spikes",
            value=round(prominence, 4),
            unit="x median",
            direction=SYNTHETIC_HIGH,
        ),
        Signal(
            key="jpeg_blockiness",
            label="JPEG 8px grid strength",
            value=round(blockiness, 5),
            unit="on/off grid ratio",
            direction=SYNTHETIC_HIGH,
        ),
        Signal(
            key="residual_energy",
            label="High-frequency residual energy",
            value=round(energy, 5),
            unit="grey levels RMS",
            direction=SYNTHETIC_HIGH,
        ),
        Signal(
            key="residual_kurtosis",
            label="High-frequency residual kurtosis",
            value=round(kurt, 5),
            unit="excess kurtosis",
            direction=SYNTHETIC_HIGH,
        ),
    ]
    return rep
