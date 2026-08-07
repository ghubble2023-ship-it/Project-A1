"""
Project AI Hardened Core v1

Rule-based catfish / romance-scam signal detection engine.

Public API (matches test_project_ai_core.py contract):
    analyze_text(params)   -> dict
    scan_photo(params)     -> dict
    analyze_profile(params)-> dict
    check_phone(params)    -> dict
    full_scan(params)      -> dict

All inputs are strongly-typed via Pydantic models.
Every function is total: no unhandled crashes on any input in the
fuzz strategies used by the test suite.
"""
from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field


ENGINE_NAME = "Project AI Hardened Core v1"


# ---------------------------------------------------------------------------
# Parameter Models
# ---------------------------------------------------------------------------
class AnalyzeTextParams(BaseModel):
    chat_messages: str = ""
    profile_text: Optional[str] = None


class ScanPhotoParams(BaseModel):
    missing_or_wrong_contact_shadow: bool = False
    glasses_frame_issue: bool = False
    lighting_direction_mismatch: bool = False
    unnatural_skin_texture: bool = False
    edge_or_hair_blending_failure: bool = False
    mismatched_eye_catchlights: bool = False
    detached_or_too_clean_background: bool = False
    observation_quality: str = "good"  # good | partial | poor


class AnalyzeProfileParams(BaseModel):
    follower_count: Optional[int] = None
    following_count: Optional[int] = None
    post_count: Optional[int] = None
    photo_count: Optional[int] = None
    account_age_days: Optional[int] = None
    bio: Optional[str] = None
    they_contacted_first: Optional[bool] = None


class CheckPhoneParams(BaseModel):
    phone_number: str
    default_region: str = "US"
    claimed_location: Optional[str] = None


class FullScanParams(BaseModel):
    # text
    chat_messages: str = ""
    profile_text: Optional[str] = None
    # photo
    missing_or_wrong_contact_shadow: bool = False
    glasses_frame_issue: bool = False
    lighting_direction_mismatch: bool = False
    unnatural_skin_texture: bool = False
    edge_or_hair_blending_failure: bool = False
    mismatched_eye_catchlights: bool = False
    detached_or_too_clean_background: bool = False
    observation_quality: str = "good"
    # profile
    follower_count: Optional[int] = None
    following_count: Optional[int] = None
    post_count: Optional[int] = None
    photo_count: Optional[int] = None
    account_age_days: Optional[int] = None
    bio: Optional[str] = None
    they_contacted_first: Optional[bool] = None
    # phone
    phone_number: Optional[str] = None
    default_region: str = "US"
    claimed_location: Optional[str] = None


# ---------------------------------------------------------------------------
# Text analyzer
# ---------------------------------------------------------------------------
_TEXT_CATEGORIES: dict[str, list[str]] = {
    "love_bombing": [
        "my love", "soulmate", "you are my", "my everything", "my heart",
        "my queen", "my king", "baby", "sweetheart", "darling",
        "i love you", "forever yours", "destiny", "meant to be",
    ],
    "off_platform_migration": [
        "telegram", "signal", "whatsapp", "wickr", "hangouts", "google chat",
        "kik", "move to", "switch to", "let's talk on", "text me on",
        "message me on", "outside of here", "off this app",
    ],
    "unverifiable_identity": [
        "deployed", "overseas", "oil rig", "on assignment", "peacekeeping",
        "military mission", "united nations", "camera broken", "phone broken",
        "no video", "can't video", "cant video", "camera doesn't work",
        "camera does not work", "can't show my face", "cant show my face",
    ],
    "pressure_urgency": [
        "why no reply", "are you there", "why not answering", "why arent you",
        "why aren't you", "hurry", "asap", "right now", "immediately",
        "no time", "please respond", "why so slow",
    ],
    "isolation": [
        "don't tell", "dont tell", "keep it between us", "our secret",
        "only me", "trust only me", "don't share",
    ],
}

_MONEY_PATTERNS: list[str] = [
    "cash app", "cashapp", "venmo", "zelle", "wire", "western union",
    "moneygram", "gift card", "steam card", "google play card", "itunes card",
    "bitcoin", "btc", "crypto", "ethereum", "usdt",
    "send me $", "send $", "loan me", "help me pay", "i need money",
    "wire transfer",
]


def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    return text.lower()


def analyze_text(params: AnalyzeTextParams) -> dict:
    haystack = _normalize(params.chat_messages) + " \n " + _normalize(params.profile_text)

    fired: List[str] = []
    matches: dict[str, list[str]] = {}

    for category, phrases in _TEXT_CATEGORIES.items():
        hits = [p for p in phrases if p in haystack]
        if hits:
            fired.append(category)
            matches[category] = hits

    money_present = any(p in haystack for p in _MONEY_PATTERNS)

    # Money alone does NOT escalate.
    n = len(fired)
    if n >= 3:
        risk = "HIGH"
    elif n >= 2:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "risk_level": risk,
        "category_count": n,
        "fired_categories": fired,
        "matched_phrases": matches,
        "money_request_present": bool(money_present),
    }


# ---------------------------------------------------------------------------
# Photo scanner
# ---------------------------------------------------------------------------
_PHOTO_FLAG_LABELS: list[tuple[str, str]] = [
    ("missing_or_wrong_contact_shadow", "Missing or inconsistent contact shadow"),
    ("glasses_frame_issue", "Glasses frame distortion / asymmetry"),
    ("lighting_direction_mismatch", "Lighting direction mismatch"),
    ("unnatural_skin_texture", "Unnaturally smooth / plastic skin texture"),
    ("edge_or_hair_blending_failure", "Edge or hair blending failure"),
    ("mismatched_eye_catchlights", "Mismatched eye catchlights"),
    ("detached_or_too_clean_background", "Detached or too-clean background"),
]


def scan_photo(params: ScanPhotoParams) -> dict:
    d = params.model_dump()
    flags: list[str] = [label for key, label in _PHOTO_FLAG_LABELS if bool(d.get(key))]
    flag_count = len(flags)

    quality = (params.observation_quality or "good").lower()
    if quality not in ("good", "partial", "poor"):
        quality = "good"

    if flag_count == 0:
        status = "No clear issues detected"
    elif quality == "poor":
        status = "Photo quality too poor to be conclusive — review recommended"
    elif flag_count >= 3:
        status = "Suspected photo — multiple visual anomalies detected"
    else:
        status = "Some anomalies noted — review recommended"

    return {
        "flag_count": flag_count,
        "flags_triggered": flags,
        "observation_quality": quality,
        "photo_status": status,
    }


# ---------------------------------------------------------------------------
# Profile analyzer
# ---------------------------------------------------------------------------
def _safe_int(v) -> Optional[int]:
    try:
        if v is None:
            return None
        i = int(v)
        if i < 0:
            return 0
        return i
    except (ValueError, TypeError):
        return None


def analyze_profile(params: AnalyzeProfileParams) -> dict:
    followers = _safe_int(params.follower_count)
    following = _safe_int(params.following_count)
    posts = _safe_int(params.post_count)
    photos = _safe_int(params.photo_count)
    age_days = _safe_int(params.account_age_days)
    bio = params.bio or ""
    contacted_first = params.they_contacted_first

    signals: list[str] = []

    if followers is not None and followers < 100:
        signals.append("Very low follower count")

    if followers is not None and following is not None:
        # High following ratio = following >> followers
        if following > 1000 and followers < 100:
            signals.append("High following-to-follower ratio")
        elif followers > 0 and following > followers * 5 and following >= 200:
            signals.append("High following-to-follower ratio")

    if posts is not None and posts < 5:
        signals.append("Very low post count")

    if photos is not None and photos < 5:
        signals.append("Very low photo count")

    if age_days is not None and age_days < 90:
        signals.append("New account (< 90 days)")

    if bio is not None and len(bio.strip()) < 10:
        # only fire if bio was actually inspected (i.e., a string was provided)
        if params.bio is not None:
            signals.append("Thin or empty bio")

    if contacted_first is True:
        signals.append("They contacted you first")

    n = len(signals)
    if n >= 3:
        concern = "ELEVATED"
    elif n == 2:
        concern = "MEDIUM"
    else:
        concern = "LOW"

    return {
        "profile_concern": concern,
        "signal_count": n,
        "signals_fired": signals,
    }


# ---------------------------------------------------------------------------
# Phone check
# ---------------------------------------------------------------------------
_US_AREA_STATE_HINTS: dict[str, str] = {
    # a tiny hint table just to enable the "location mismatch" signal.
    "212": "New York", "213": "Los Angeles", "312": "Chicago",
    "415": "San Francisco", "541": "Oregon", "617": "Boston",
    "702": "Las Vegas", "305": "Miami", "202": "Washington DC",
}


def check_phone(params: CheckPhoneParams) -> dict:
    raw = params.phone_number or ""
    digits = re.sub(r"[^\d]", "", raw)
    has_plus = raw.strip().startswith("+")
    region = (params.default_region or "US").upper()

    flags: list[str] = []
    valid = 7 <= len(digits) <= 15

    if not valid:
        flags.append("Number is not a plausible length")

    if digits.startswith("0") and not has_plus:
        flags.append("Leading zero without country code")

    # US-specific hint: 10 digits or 11 with leading 1
    inferred_area = None
    if region == "US" and valid:
        us_digits = digits
        if len(us_digits) == 11 and us_digits.startswith("1"):
            us_digits = us_digits[1:]
        if len(us_digits) == 10:
            inferred_area = us_digits[:3]

    if inferred_area and params.claimed_location:
        expected = _US_AREA_STATE_HINTS.get(inferred_area)
        claimed = params.claimed_location.lower().strip()
        if expected and claimed and expected.lower() not in claimed and claimed not in expected.lower():
            flags.append(
                f"Area code {inferred_area} typically maps to {expected}, "
                f"but claimed location is '{params.claimed_location}'"
            )

    if valid and inferred_area is None and region == "US" and not has_plus:
        flags.append("Cannot determine US region from number")

    if valid:
        summary = "Number is plausibly valid" if not flags else "Number is valid but has advisory flags"
    else:
        summary = "Number could not be validated"

    return {
        "valid": bool(valid),
        "digits": digits,
        "region": region,
        "inferred_area_code": inferred_area,
        "flags": flags,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Full scan
# ---------------------------------------------------------------------------
def full_scan(params: FullScanParams) -> dict:
    text_result = analyze_text(AnalyzeTextParams(
        chat_messages=params.chat_messages,
        profile_text=params.profile_text,
    ))

    photo_result = scan_photo(ScanPhotoParams(
        missing_or_wrong_contact_shadow=params.missing_or_wrong_contact_shadow,
        glasses_frame_issue=params.glasses_frame_issue,
        lighting_direction_mismatch=params.lighting_direction_mismatch,
        unnatural_skin_texture=params.unnatural_skin_texture,
        edge_or_hair_blending_failure=params.edge_or_hair_blending_failure,
        mismatched_eye_catchlights=params.mismatched_eye_catchlights,
        detached_or_too_clean_background=params.detached_or_too_clean_background,
        observation_quality=params.observation_quality,
    ))

    profile_result = analyze_profile(AnalyzeProfileParams(
        follower_count=params.follower_count,
        following_count=params.following_count,
        post_count=params.post_count,
        photo_count=params.photo_count,
        account_age_days=params.account_age_days,
        bio=params.bio,
        they_contacted_first=params.they_contacted_first,
    ))

    phone_result = None
    if params.phone_number:
        phone_result = check_phone(CheckPhoneParams(
            phone_number=params.phone_number,
            default_region=params.default_region,
            claimed_location=params.claimed_location,
        ))

    # Overall risk is driven by text risk_level (per the fuzz contract).
    overall = text_result["risk_level"]

    return {
        "engine": ENGINE_NAME,
        "overall_risk": overall,
        "text": text_result,
        "photo": photo_result,
        "profile": profile_result,
        "phone": phone_result,
    }
