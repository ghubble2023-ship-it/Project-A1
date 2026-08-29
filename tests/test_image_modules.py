"""Image modules must degrade to 'I could not measure that', never to a crash."""

import cv2
import numpy as np
import pytest

from sentinel.detect import FaceEyeDetector, shared_detector
from sentinel.image import ocular, optics, provenance, spectral
from sentinel.image.pipeline import analyse_image, decode


def encode(bgr, quality=92):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


@pytest.fixture
def noise_512():
    rng = np.random.default_rng(1234)
    return rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)


def test_detector_backend_is_available():
    """Guards the OpenCV 5 regression: cascades are no longer bundled."""
    det = shared_detector()
    assert det.available, (
        "no face detector; install opencv-python-headless<5 or set "
        "SENTINEL_YUNET_MODEL"
    )


def test_ocular_abstains_without_a_face(noise_512):
    rep = ocular.analyse(noise_512)
    assert rep.error is None
    assert all(not s.available for s in rep.signals)
    assert all(s.unavailable_reason for s in rep.signals)


def test_spectral_abstains_on_undersized_image():
    tiny = np.zeros((64, 64, 3), np.uint8)
    rep = spectral.analyse(tiny)
    assert all(not s.available for s in rep.signals)
    assert "smaller than" in rep.signals[0].unavailable_reason


def test_spectral_measures_full_size_image(noise_512):
    rep = spectral.analyse(noise_512)
    keys = {s.key for s in rep.signals if s.available}
    assert keys == {
        "upsampling_peak_prominence",
        "jpeg_blockiness",
        "residual_energy",
        "residual_kurtosis",
    }


def test_spectral_is_resolution_independent():
    """The same content at two sizes must not change the spectral verdict much.

    Resolution is the dominant confounder in this problem, so this is a
    regression test against accidentally reintroducing it.
    """
    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, (1024, 1024, 3), dtype=np.uint8)
    small = cv2.resize(base, (512, 512), interpolation=cv2.INTER_AREA)
    a = {s.key: s.value for s in spectral.analyse(base).signals}
    b = {s.key: s.value for s in spectral.analyse(small).signals}
    # Prominence is a ratio against the local median, so it should stay in the
    # same ballpark even though pixel counts differ 4x.
    assert abs(a["upsampling_peak_prominence"] - b["upsampling_peak_prominence"]) < 3.0


def test_optics_abstains_on_flat_image():
    flat = np.full((512, 512, 3), 128, np.uint8)
    rep = optics.analyse(flat)
    assert all(not s.available for s in rep.signals)
    assert "usable chromatic edges" in rep.signals[0].unavailable_reason


def test_provenance_reports_absent_metadata(noise_512):
    rep = provenance.analyse(encode(noise_512))
    completeness = rep.signal("exif_capture_completeness")
    assert completeness.available
    assert completeness.value == 0.0
    assert any("No EXIF" in n for n in rep.notes)


def test_provenance_survives_a_corrupt_container():
    rep = provenance.analyse(b"not an image at all")
    assert rep.error
    assert all(not s.available for s in rep.signals)


def test_decode_rejects_garbage():
    with pytest.raises(ValueError):
        decode(b"\x00\x01\x02garbage")


def test_pipeline_runs_all_modules_and_isolates_failure(noise_512):
    result = analyse_image(encode(noise_512))
    modules = {m["module"] for m in result["modules"]}
    assert modules == {"ocular", "optics", "spectral", "provenance"}
    assert "verdict" in result and "explanation" in result
