"""The screen-capture gate: it must refuse screenshots and never refuse photographs.

The asymmetry is the point. Wrongly analysing a screenshot produces a confident
number about a question that has no answer; wrongly refusing a photograph makes
the tool useless on its actual job. These tests encode that priority: the
false-positive tests are the strict ones.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from sentinel.image import screen
from sentinel.image.pipeline import analyse_image


def _encode(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    assert ok
    return buf.tobytes()


def _synthetic_ui(width: int = 1080, height: int = 2340) -> np.ndarray:
    """A crude chat interface: flat fills, full-width dividers, small palette."""
    img = np.full((height, width, 3), 245, np.uint8)
    img[:90] = 30                                  # status bar
    for top in range(200, height - 200, 260):      # message bubbles
        colour = (220, 200, 120) if (top // 260) % 2 else (235, 235, 235)
        cv2.rectangle(img, (60, top), (width - 60, top + 170), colour, -1)
        cv2.line(img, (0, top + 200), (width, top + 200), (215, 215, 215), 2)
    return img


def _synthetic_photograph(size: int = 512) -> np.ndarray:
    """Smooth gradients plus sensor-like noise: no flat tiles, no uniform runs."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = 90 + 70 * np.sin(xx / 47.0) + 50 * np.cos(yy / 31.0)
    img = np.stack([base + rng.normal(0, 9, (size, size)) for _ in range(3)], axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_flags_a_rendered_interface():
    report = screen.detect(_synthetic_ui())
    assert report.is_screen_capture
    assert report.reason


def test_does_not_flag_a_noisy_photograph():
    report = screen.detect(_synthetic_photograph())
    assert not report.is_screen_capture


def test_pipeline_refuses_a_screenshot_rather_than_guessing():
    result = analyse_image(_encode(_synthetic_ui()))
    verdict = result["verdict"]
    assert verdict["label"] == "INCONCLUSIVE_NOT_A_PHOTOGRAPH"
    # The critical assertion: no number is invented.
    assert verdict["p_synthetic"] is None
    assert verdict["coverage"] == 0.0
    assert result["explanation"]["caveats"], "a refusal must explain itself"


def test_pipeline_still_scores_a_photograph():
    result = analyse_image(_encode(_synthetic_photograph()))
    assert result["verdict"]["label"] != "INCONCLUSIVE_NOT_A_PHOTOGRAPH"
    assert result["verdict"]["p_synthetic"] is not None


def test_media_block_always_reports_the_gate_measurements():
    """Whether or not it fires, the numbers behind the decision are exposed."""
    for image in (_synthetic_ui(), _synthetic_photograph()):
        block = analyse_image(_encode(image))["media"]["screen_capture"]
        assert set(block) == {
            "is_screen_capture",
            "flat_region_fraction",
            "palette_ratio",
            "uniform_run",
            "reason",
        }


@pytest.mark.parametrize("shape", [(64, 64, 3), (5, 5, 3), (300, 12, 3)])
def test_detector_survives_degenerate_shapes(shape):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, shape, dtype=np.uint8)
    report = screen.detect(img)
    assert isinstance(report.is_screen_capture, bool)


def test_a_uniform_image_is_treated_as_a_screen_not_a_photo():
    """Degenerate but important: a blank image has no capture evidence either."""
    assert screen.detect(np.full((400, 400, 3), 200, np.uint8)).is_screen_capture
