"""Backend tests for Project AI Hardened Core v1 scanner API."""
import os
import pytest
import requests

BASE_URL = (os.environ.get("EXPO_PUBLIC_BACKEND_URL") or os.environ.get("EXPO_BACKEND_URL", "https://mobile-launch-4895.preview.emergentagent.com")).rstrip("/")
API = f"{BASE_URL}/api"


@pytest.fixture(scope="module")
def s():
    sess = requests.Session()
    sess.headers.update({"Content-Type": "application/json"})
    return sess


# ---------- health ----------
def test_health(s):
    r = s.get(f"{API}/")
    assert r.status_code == 200
    assert r.json().get("engine") == "Project AI Hardened Core v1"


# ---------- text scan fixtures ----------
def test_text_clean(s):
    r = s.post(f"{API}/scan/text", json={"chat_messages": "Hey, how was your day? Want to grab coffee this weekend?"})
    assert r.status_code == 200
    d = r.json()
    assert d["risk_level"] == "LOW"
    assert d["category_count"] == 0


def test_text_medium(s):
    r = s.post(f"{API}/scan/text", json={"chat_messages": "My love I miss you so much. Can we talk on telegram?"})
    assert r.status_code == 200
    d = r.json()
    assert d["risk_level"] == "MEDIUM", d
    assert d["category_count"] == 2, d


def test_text_high(s):
    r = s.post(f"{API}/scan/text", json={"chat_messages": "My love you are my soulmate. Please move to signal. I am deployed overseas and my camera is broken. Are you there? Why no reply?"})
    assert r.status_code == 200
    d = r.json()
    assert d["risk_level"] == "HIGH", d
    assert d["category_count"] >= 3, d


def test_text_money_only(s):
    r = s.post(f"{API}/scan/text", json={"chat_messages": "Hey, can you send me $50 on Cash App real quick? I need it."})
    assert r.status_code == 200
    d = r.json()
    assert d["money_request_present"] is True
    assert d["risk_level"] == "LOW"


# ---------- photo ----------
def test_photo_no_flags(s):
    r = s.post(f"{API}/scan/photo", json={})
    assert r.status_code == 200
    d = r.json()
    assert d["flag_count"] == 0
    assert d["photo_status"] == "No clear issues detected"


def test_photo_5_flags_good(s):
    payload = {
        "missing_or_wrong_contact_shadow": True,
        "glasses_frame_issue": True,
        "lighting_direction_mismatch": True,
        "unnatural_skin_texture": True,
        "edge_or_hair_blending_failure": True,
        "observation_quality": "good",
    }
    r = s.post(f"{API}/scan/photo", json=payload)
    d = r.json()
    assert d["flag_count"] >= 3
    assert "Suspected photo" in d["photo_status"]


def test_photo_3_flags_poor(s):
    payload = {
        "missing_or_wrong_contact_shadow": True,
        "glasses_frame_issue": True,
        "lighting_direction_mismatch": True,
        "observation_quality": "poor",
    }
    r = s.post(f"{API}/scan/photo", json=payload)
    d = r.json()
    assert "Suspected photo" not in d["photo_status"]
    assert "review recommended" in d["photo_status"]


# ---------- profile ----------
def test_profile_clean(s):
    r = s.post(f"{API}/scan/profile", json={
        "follower_count": 450, "following_count": 380, "post_count": 120,
        "photo_count": 15, "account_age_days": 600,
        "bio": "Photographer based in Oregon. Love hiking.",
        "they_contacted_first": False,
    })
    d = r.json()
    assert d["profile_concern"] == "LOW", d


def test_profile_elevated(s):
    r = s.post(f"{API}/scan/profile", json={
        "follower_count": 28, "following_count": 1850, "post_count": 1,
        "photo_count": 2, "account_age_days": 22, "bio": "hi",
        "they_contacted_first": True,
    })
    d = r.json()
    assert d["profile_concern"] == "ELEVATED", d
    assert d["signal_count"] >= 3


# ---------- phone ----------
def test_phone_mismatch(s):
    r = s.post(f"{API}/scan/phone", json={
        "phone_number": "+15415550100", "default_region": "US", "claimed_location": "Chicago"
    })
    d = r.json()
    assert d["valid"] is True
    assert d["inferred_area_code"] == "541"
    assert any("Oregon" in f and "Chicago" in f for f in d["flags"]), d


# ---------- full scan ----------
def test_full_scan_high(s):
    r = s.post(f"{API}/scan/full", json={
        "chat_messages": "My love you are my soulmate. Please move to signal. I am deployed overseas and my camera is broken. Are you there? Why no reply?",
        "follower_count": 28, "following_count": 1850, "post_count": 1,
        "photo_count": 2, "account_age_days": 22, "bio": "hi",
        "they_contacted_first": True,
        "phone_number": "+15415550100", "claimed_location": "Chicago",
    })
    d = r.json()
    assert d["overall_risk"] == "HIGH"
    assert d["text"]["risk_level"] == "HIGH"
    assert d["profile"]["profile_concern"] == "ELEVATED"
    assert d["phone"] is not None
    assert d["engine"] == "Project AI Hardened Core v1"


def test_full_scan_empty(s):
    r = s.post(f"{API}/scan/full", json={})
    d = r.json()
    assert d["overall_risk"] == "LOW"
    assert d["phone"] is None


# ---------- history CRUD ----------
def test_history_crud(s):
    # clear
    s.delete(f"{API}/scans")
    payload = {
        "scan_type": "text",
        "input_summary": "TEST_summary",
        "result": {"risk_level": "LOW"},
        "overall_risk": "LOW",
    }
    r = s.post(f"{API}/scans", json=payload)
    assert r.status_code == 200
    rec = r.json()
    sid = rec["id"]
    assert "_id" not in rec

    r = s.get(f"{API}/scans")
    assert r.status_code == 200
    lst = r.json()
    assert any(x["id"] == sid for x in lst)
    for x in lst:
        assert "_id" not in x

    r = s.get(f"{API}/scans/{sid}")
    assert r.status_code == 200 and r.json()["id"] == sid

    r = s.delete(f"{API}/scans/{sid}")
    assert r.status_code == 200 and r.json()["deleted"] == 1

    r = s.get(f"{API}/scans/{sid}")
    assert r.status_code == 404

    # clear all
    s.post(f"{API}/scans", json=payload)
    r = s.delete(f"{API}/scans")
    assert r.status_code == 200
