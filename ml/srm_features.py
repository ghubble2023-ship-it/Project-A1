"""
SRM-lite: co-occurrence features of high-pass noise residuals.

This is the technique that actually works on generated faces even after
the image has been resized and re-compressed (which is exactly what
happened to this dataset - both classes were re-saved at the same JPEG
quality, which wipes out the easy spectral fingerprints). It comes from
steganalysis / GAN-detection literature: apply a set of high-pass
filters, quantise + truncate the residual, then histogram how
neighbouring residual values co-occur. Generated images have subtly
different local pixel-correlation structure than camera images, and
these histograms capture it.

Pure numpy/OpenCV. No neural network, no downloads.
"""
import cv2
import numpy as np

T = 2          # truncation: residual clipped to [-T, T] -> 2T+1 levels
ORDER = 3      # co-occurrence order (triples of adjacent residuals)
LEVELS = 2 * T + 1
BINS = LEVELS ** ORDER

# high-pass kernels (first order, second order, 3x3 "square" SRM kernel)
KERNELS = {
    "d1h": np.array([[-1, 1]], dtype=np.float32),
    "d1v": np.array([[-1], [1]], dtype=np.float32),
    "d2h": np.array([[1, -2, 1]], dtype=np.float32),
    "d2v": np.array([[1], [-2], [1]], dtype=np.float32),
    "sq3": np.array([[-1, 2, -1],
                     [2, -4, 2],
                     [-1, 2, -1]], dtype=np.float32) / 4.0,
}


def _cooc(res_q, axis):
    """3rd-order co-occurrence histogram of a quantised residual plane."""
    if axis == 0:      # vertical neighbours
        a, b, c = res_q[:-2, :], res_q[1:-1, :], res_q[2:, :]
    else:              # horizontal neighbours
        a, b, c = res_q[:, :-2], res_q[:, 1:-1], res_q[:, 2:]
    idx = (a * LEVELS * LEVELS + b * LEVELS + c).ravel()
    h = np.bincount(idx, minlength=BINS).astype(np.float32)
    return h / max(h.sum(), 1.0)


def extract(bgr_image, size=256):
    img = cv2.resize(bgr_image, (size, size), interpolation=cv2.INTER_AREA)
    # use all three colour channels: cross-channel structure carries signal
    planes = [img[:, :, i].astype(np.float32) for i in range(3)]
    planes.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32))

    vec = []
    for kname, k in KERNELS.items():
        for p in planes:
            res = cv2.filter2D(p, cv2.CV_32F, k)
            q = np.clip(np.round(res / 2.0), -T, T).astype(np.int32) + T
            vec.append(_cooc(q, 0))
            vec.append(_cooc(q, 1))
    return np.concatenate(vec)


DIM = len(KERNELS) * 4 * 2 * BINS
