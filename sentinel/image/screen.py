"""Is this a photograph at all, or a picture of a screen?

Every signal in the image pipeline is a statement about a *capture pipeline*:
sensor noise residuals, lens dispersion, demosaicing, JPEG history. A screenshot
has none of that. The phone's compositor rendered it, so the residual statistics
describe a display buffer and the calibration -- fitted on photographs -- has no
idea what it is looking at.

Left ungated, the system does not notice. Run the 17 social-media screenshots in
this project's evaluation set through the photo pipeline and it returns
LEANS_SYNTHETIC on three of them and LEANS_AUTHENTIC on four, with coverage 1.00
and no indication that the question was meaningless. That is the worst failure
mode available to a forensics tool: a confident answer to a question it cannot
answer.

So this is a **gate, not a signal**. It does not contribute evidence to the
fused score. When it fires, the image verdict is refused outright, because there
is no defensible number to report. Someone who screenshots a dating profile and
asks "is this AI?" deserves "I can't tell from a screenshot, send me the file"
rather than a fabricated probability.

Three structural cues, all scale-free, none of which depend on EXIF (which is
trivially stripped):

``flat_region_fraction``
    Fraction of 8x8 tiles with essentially zero variance. UI is full of
    perfectly uniform fills -- backgrounds, chat bubbles, bars. Photographs
    almost never contain a truly flat 8x8 patch, because sensors have noise.

``palette_ratio``
    Distinct colours as a fraction of pixels, on a nearest-neighbour resample so
    interpolation cannot invent shades. Interfaces are drawn from a small
    palette; photographs are not.

``uniform_run``
    90th-percentile longest horizontal run of near-constant pixels, as a
    fraction of width. UI dividers and bubble edges produce rows that are
    constant across most of the screen. This is the strongest of the three: 16
    of 17 screenshots score above 0.9, and no photograph measured here exceeds
    0.402.

The rule is deliberately biased toward *not* firing. Wrongly refusing to analyse
a real photograph is a worse failure than analysing a screenshot, so the
thresholds were chosen for precision:

    on 17 screenshots and 80 photographs (both classes of both corpora),
    recall 16/17 (94%), false positives 0/80.

Those thresholds are fitted to that data and are the honest limit of the claim:
this is validated against phone screenshots of chat and social apps, not against
scans, photographs-of-screens, or heavily graphic-designed images. The one miss
is a screenshot whose content is mostly a single large photograph, which is the
expected boundary case -- and arguably the one where running the photo pipeline
does the least harm.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# --- thresholds, fitted on 17 screenshots + 80 photographs (see module docstring) ---
TILE = 8
FLAT_VAR_MAX = 1.0        # a tile this uniform is not sensor output
UNIFORM_RUN_FIRE = 0.60   # highest photograph measured: 0.402
FLAT_FIRE = 0.27          # highest photograph measured below the palette guard: 0.255
PALETTE_FIRE = 0.22       # lowest photograph measured: 0.216
RUN_NORM_WIDTH = 1024     # normalise wide images so the run measure is scale-free
RUN_TOLERANCE = 1         # grey levels; JPEG never yields exactly-equal neighbours


@dataclass(frozen=True)
class ScreenReport:
    """Why the gate did or did not fire. Every number is reported, not just the verdict."""

    is_screen_capture: bool
    flat_region_fraction: float
    palette_ratio: float
    uniform_run: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "is_screen_capture": self.is_screen_capture,
            "flat_region_fraction": round(self.flat_region_fraction, 4),
            "palette_ratio": round(self.palette_ratio, 4),
            "uniform_run": round(self.uniform_run, 4),
            "reason": self.reason,
        }


def _flat_region_fraction(grey: np.ndarray) -> float:
    h, w = grey.shape
    if h < TILE or w < TILE:
        return 0.0
    hh, ww = (h // TILE) * TILE, (w // TILE) * TILE
    tiles = (
        grey[:hh, :ww]
        .reshape(hh // TILE, TILE, ww // TILE, TILE)
        .swapaxes(1, 2)
        .reshape(-1, TILE * TILE)
        .astype(np.float32)
    )
    return float((tiles.var(axis=1) < FLAT_VAR_MAX).mean())


def _palette_ratio(bgr: np.ndarray) -> float:
    # INTER_NEAREST: resampling must not blend colours into existence.
    small = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
    distinct = len(np.unique(small.reshape(-1, 3), axis=0))
    return distinct / (256 * 256)


def _uniform_run(grey: np.ndarray) -> float:
    if grey.shape[1] > RUN_NORM_WIDTH + 176:
        height = max(1, int(grey.shape[0] * RUN_NORM_WIDTH / grey.shape[1]))
        grey = cv2.resize(grey, (RUN_NORM_WIDTH, height), interpolation=cv2.INTER_AREA)
    h, w = grey.shape
    if w < 2:
        return 0.0
    same = np.abs(np.diff(grey.astype(np.int16), axis=1)) <= RUN_TOLERANCE
    longest = np.zeros(h, dtype=np.float32)
    for i in range(h):
        row = same[i]
        if not row.any():
            continue
        # Run-length boundaries via the padded-diff trick.
        edges = np.flatnonzero(np.diff(np.r_[0, row.view(np.int8), 0]))
        longest[i] = (edges[1::2] - edges[::2]).max()
    return float(np.percentile(longest / w, 90))


def detect(bgr: np.ndarray) -> ScreenReport:
    """Classify an already-decoded BGR image as screen capture or photograph."""
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    flat = _flat_region_fraction(grey)
    palette = _palette_ratio(bgr if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR))
    run = _uniform_run(grey)

    if run >= UNIFORM_RUN_FIRE:
        return ScreenReport(
            True, flat, palette, run,
            f"{run:.0%} of image width is a single near-constant run of pixels on a "
            f"typical row; photographs measured here never exceed 40%",
        )
    if flat >= FLAT_FIRE and palette <= PALETTE_FIRE:
        return ScreenReport(
            True, flat, palette, run,
            f"{flat:.0%} of 8x8 tiles are perfectly uniform and the image is drawn from "
            f"an unusually small palette ({palette:.0%} distinct colours) -- both are "
            f"properties of rendered interfaces, not of sensor output",
        )
    return ScreenReport(
        False, flat, palette, run,
        "no screen-capture structure detected; treated as a photograph",
    )
