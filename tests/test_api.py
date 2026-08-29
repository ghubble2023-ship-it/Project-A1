"""HTTP contract, including the shim the existing front-ends depend on."""

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sentinel.api import app

client = TestClient(app)


def jpeg_bytes(dim=512, seed=3):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (dim, dim, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_health_reports_detector_and_calibration():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["face_detector"] in {"haar", "yunet"}
    assert body["calibrated_signals"]


def test_calibration_is_publicly_inspectable():
    """Secret thresholds are unreviewable thresholds."""
    body = client.get("/calibration").json()
    assert body["signals"]
    for entry in body["signals"].values():
        assert "auc" in entry and "weight" in entry


def test_analyze_image_returns_verdict_and_explanation():
    r = client.post(
        "/analyze/image", files={"file": ("a.jpg", io.BytesIO(jpeg_bytes()), "image/jpeg")}
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"media", "verdict", "explanation", "modules"}
    assert 0.0 <= body["verdict"]["p_synthetic"] <= 1.0


def test_analyze_image_rejects_non_image():
    r = client.post(
        "/analyze/image", files={"file": ("a.txt", io.BytesIO(b"hello"), "text/plain")}
    )
    assert r.status_code == 400


def test_analyze_image_rejects_empty_upload():
    r = client.post(
        "/analyze/image", files={"file": ("a.jpg", io.BytesIO(b""), "image/jpeg")}
    )
    assert r.status_code == 400


def test_analyze_chat_scores_a_funnel():
    r = client.post(
        "/analyze/chat",
        json={
            "messages": [
                {"sender": "suspect", "text": "This is the fraud department, read back the 6-digit code."},
                {"sender": "suspect", "text": "Install AnyDesk so we can secure your account."},
            ]
        },
    )
    assert r.status_code == 200
    assert r.json()["risk_band"] in {"HIGH", "CRITICAL"}


def test_analyze_chat_rejects_unknown_suspect():
    r = client.post(
        "/analyze/chat",
        json={"messages": [{"sender": "a", "text": "hi"}], "suspect": "nobody"},
    )
    assert r.status_code == 400


def test_legacy_ocular_endpoint_keeps_its_shape():
    """The Next.js console and Android client were written against this."""
    r = client.post(
        "/forensics/ocular-comparator",
        files={"file": ("a.jpg", io.BytesIO(jpeg_bytes()), "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    assert "ocular_consistency_forensics" in body
    dakota = body["dakota_report"]
    assert dakota["analyst_persona"] == "Dakota"
    assert isinstance(dakota["flagged_discrepancies"], list)
    assert isinstance(dakota["detailed_physical_breakdown"], list)
    assert dakota["caveats"]
