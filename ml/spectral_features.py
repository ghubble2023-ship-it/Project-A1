"""
Frequency-domain forensic features - the signal the original 8 features
were missing.

Why this and not another hand-tuned "tell": GAN/diffusion image
generators build pictures by repeated upsampling (transposed
convolution / interpolation). That process leaves a periodic,
checkerboard-like fingerprint that is essentially invisible to the eye
but shows up clearly in the 2D Fourier power spectrum as excess energy
at high frequencies and at specific radial bands. This is a real,
published forensic technique (spectral artifact analysis), not a
guess - and unlike "eyes look empty", it's fully computable offline.

Everything here is a plain number derived from the pixels. No model, no
network, no download.
"""
import cv2
import numpy as np
from scipy import stats

RADIAL_BINS = 12

FEATURE_NAMES = (
    [f"spec_radial_{i}" for i in range(RADIAL_BINS)]
    + [
        "spec_high_freq_ratio",
        "spec_slope",
        "residual_std",
        "residual_kurtosis",
        "residual_high_freq_ratio",
        "color_sat_mean",
        "color_sat_std",
        "chroma_lap_var",
    ]
)


def _azimuthal_profile(mag, bins=RADIAL_BINS):
    h, w = mag.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.indices((h, w))
    r = np.hypot(y - cy, x - cx)
    r_max = r.max()
    edges = np.linspace(0, r_max, bins + 1)
    out = []
    for i in range(bins):
        m = (r >= edges[i]) & (r < edges[i + 1])
        out.append(float(mag[m].mean()) if m.any() else 0.0)
    return out


def extract(bgr_image):
    """Returns dict of floats. Never returns None - these features are
    always computable, no face detection required."""
    img = cv2.resize(bgr_image, (256, 256), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # --- 2D power spectrum of the whole image (windowed to kill edge ringing) ---
    win = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
    f = np.fft.fftshift(np.fft.fft2(gray * win))
    logmag = np.log1p(np.abs(f))
    profile = _azimuthal_profile(logmag)
    total = sum(profile) or 1.0
    profile_norm = [p / total for p in profile]

    feats = {f"spec_radial_{i}": profile_norm[i] for i in range(RADIAL_BINS)}
    hi = sum(profile[RADIAL_BINS // 2:])
    feats["spec_high_freq_ratio"] = float(hi / total)

    # log-log slope of the radial profile = how fast energy falls off
    r = np.arange(1, RADIAL_BINS + 1, dtype=float)
    p = np.array(profile, dtype=float) + 1e-9
    slope = np.polyfit(np.log(r), np.log(p), 1)[0]
    feats["spec_slope"] = float(slope)

    # --- high-pass residual statistics ---
    blur = cv2.medianBlur((gray * 255).astype(np.uint8), 3).astype(np.float32) / 255.0
    resid = gray - blur
    feats["residual_std"] = float(resid.std())
    feats["residual_kurtosis"] = float(stats.kurtosis(resid.ravel()))
    rf = np.fft.fftshift(np.fft.fft2(resid * win))
    rprof = _azimuthal_profile(np.log1p(np.abs(rf)))
    rtot = sum(rprof) or 1.0
    feats["residual_high_freq_ratio"] = float(sum(rprof[RADIAL_BINS // 2:]) / rtot)

    # --- colour statistics (generators often compress the saturation range) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    feats["color_sat_mean"] = float(sat.mean())
    feats["color_sat_std"] = float(sat.std())
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    feats["chroma_lap_var"] = float(cv2.Laplacian(ycrcb[:, :, 1], cv2.CV_64F).var())

    return feats
