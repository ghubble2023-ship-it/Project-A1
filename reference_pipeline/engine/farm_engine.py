"""
Farm Engine - Types B, C, C-W.

Type B and Type C are ported directly from the original spec - that
rule logic was already real, not a stub.

Type C-W ("Wig Farm") previously had:
    compareAnchorsIgnoringHair: (profiles) => ({samePersonDifferentWigs: true})
which always returned true regardless of input. Replaced below with an
actual, real comparison.

BE HONEST ABOUT WHAT THIS IS: true face-matching would use a trained
face-embedding model (e.g. dlib/face_recognition), which isn't
available offline in this sandbox. What's implemented instead is a
coarse biometric-ratio proxy:
  - inter-eye-distance / face-width ratio (roughly hairstyle-invariant,
    a known "soft biometric," but nowhere near as reliable as a real
    face-embedding match)
  - a glasses-bridge edge heuristic (rough, will have false
    positives/negatives on its own)
This is a real first-pass filter, not a conclusive identity match -
treat a flag from this as "worth a human look," not "confirmed."
"""

import re
import cv2
import numpy as np

_CASCADE_DIR = cv2.data.haarcascades
_face_cascade = cv2.CascadeClassifier(_CASCADE_DIR + "haarcascade_frontalface_default.xml")
_eye_cascade = cv2.CascadeClassifier(_CASCADE_DIR + "haarcascade_eye.xml")

RATIO_TOLERANCE = 0.035  # how close two profiles' eye/face ratios must be to call them "same anchor"


# ---------- Type B ----------

def detect_type_b(screen: dict):
    """
    From Type_B_Paragraph_v2.txt:
    No verifiable info after friending, refuses extra photos, deflects
    off-platform, generic looping affection, monetization is the
    conversation itself (credit consumption).
    """
    if screen.get("monetization_is_chat") and screen.get("refuses_photos") and screen.get("deflects_external"):
        return [{
            "type": "FARM", "subType": "B - Engagement Retention", "risk": "MEDIUM",
            "action": "Credit drain - not meeting", "sources": "Dating.com, Blink-match.com",
        }]
    return []


# ---------- Type C ----------

CATALOG_LANGUAGE = re.compile(r"number\s*\d{4,5}|she.*number.*loves", re.I)


def detect_type_c(screen: dict):
    """
    Catalog language, handler+bait separation, fails a real-time local
    test (e.g. claims weather that doesn't match actual local
    conditions), photo timestamp/thumbnail irregularities.
    Scoring and threshold ported directly from the original spec.
    """
    score = 0
    if CATALOG_LANGUAGE.search(screen.get("text", "") or ""):
        score += 2
    if screen.get("has_handler_bait_separation"):
        score += 2
    if screen.get("failed_weather_test"):
        score += 2
    if screen.get("same_timestamp_diff_photos"):
        score += 2

    if score >= 4:
        return [{
            "type": "CRITICAL", "subType": "C - Physical Lure", "risk": "CRITICAL - DO NOT GO",
            "rule": "inventory_number OR catalog_phrasing AND meeting_by_third_party AND fails_local_test => CRITICAL",
            "action": "Purge chat, screenshot address, tell trusted person, report template. Never go to verify.",
        }]
    return []


# ---------- Type C-W (Wig Farm) - the part that's actually new ----------

def _measure_face_anchor(bgr_image):
    """
    Returns dict with eye_face_ratio and glasses_present, or None if no
    face/eyes detected.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[fy:fy + fh, fx:fx + fw]

    eyes = _eye_cascade.detectMultiScale(face_roi, 1.1, 8)
    if len(eyes) < 2:
        return None

    # Take the two largest eye detections
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    centers = [(ex + ew / 2, ey + eh / 2) for ex, ey, ew, eh in eyes]
    eye_dist = float(np.hypot(centers[0][0] - centers[1][0], centers[0][1] - centers[1][1]))

    ratio = eye_dist / fw

    # Glasses-bridge heuristic: look for a strong horizontal edge
    # spanning between the two eyes (the bridge/frame line)
    y_band = int(min(centers[0][1], centers[1][1]))
    x1, x2 = sorted([int(centers[0][0]), int(centers[1][0])])
    band = face_roi[max(0, y_band - 4):y_band + 4, x1:x2]
    glasses_present = False
    if band.size > 0:
        edges = cv2.Canny(band, 50, 150)
        edge_density = float(np.mean(edges > 0))
        glasses_present = edge_density > 0.12  # empirically loose threshold - heuristic, not exact

    return {"eye_face_ratio": ratio, "glasses_present": glasses_present}


def compare_anchors_ignoring_hair(profile_image_paths):
    """
    Real replacement for the placeholder. Takes a list of image file
    paths (one representative photo per profile being compared) and
    returns a real comparison instead of a hardcoded True.
    """
    anchors = []
    for path in profile_image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        a = _measure_face_anchor(img)
        if a is not None:
            a["path"] = path
            anchors.append(a)

    if len(anchors) < 2:
        return {"samePersonDifferentWigs": False, "reason": "not enough usable faces detected", "anchors": anchors}

    # Pairwise compare ratio closeness + glasses agreement
    base = anchors[0]
    matches = []
    for other in anchors[1:]:
        ratio_close = abs(other["eye_face_ratio"] - base["eye_face_ratio"]) < RATIO_TOLERANCE
        glasses_agree = other["glasses_present"] == base["glasses_present"]
        matches.append(ratio_close and glasses_agree)

    same_person = all(matches) if matches else False
    return {
        "samePersonDifferentWigs": same_person,
        "anchors": anchors,
        "confidence": "LOW-MEDIUM - ratio+glasses heuristic only, not a trained face-embedding match",
    }


def detect_type_cw(screen: dict):
    """
    screen["profile_image_paths"]: list of image paths for the
    profiles being compared (2+ needed).
    """
    paths = screen.get("profile_image_paths") or []
    if len(paths) >= 2:
        anchor_match = compare_anchors_ignoring_hair(paths)
        if anchor_match["samePersonDifferentWigs"]:
            return [{
                "type": "FARM", "subType": "C-W - Wig Farm",
                "threat": "The Coming Floods - AI makes farms 10x easier",
                "ignore": ["hairColor", "hairStyle", "wig"],
                "anchor": ["eyeFaceRatio", "glassesBridge"],
                "confidence": anchor_match["confidence"],
                "action": "Flag as possible same-operator farm - verify with a human before acting",
            }]
    return []


def detect_type_e(screen: dict):
    """
    Type E was carried in the original spec as an inherited placeholder
    from an earlier phase, with no criteria ever defined. Rather than
    inventing detection rules you never gave me, this stays empty until
    you specify what Type E actually is.
    """
    return []


def analyze(screen: dict):
    flags = []
    flags += detect_type_b(screen)
    flags += detect_type_c(screen)
    flags += detect_type_cw(screen)
    flags += detect_type_e(screen)
    return flags
