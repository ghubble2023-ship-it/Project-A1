"""
Project AI - Eyes of Greg - restored long-press architecture.

Mirrors the original onLongPress flow:
  visual + text + profile + farm + gravity -> fusion -> show result -> delete image, log hash+score only

This is the ORIGINAL single-scan-on-demand design (long press a photo/
profile/conversation -> get an answer), NOT the continuous live-screen-
capture + multi-device UDP cluster from the "Project AI rdy" handoff
doc - that's a different architecture and isn't part of this restore.
"""

import cv2

from engine import gravity_check, text_engine, profile_engine, farm_engine, visual_engine, fusion_scorer, privacy
from engine.stolen_image_db import LocalHashDB, sha256_of_bytes


def long_press_scan(image_path: str, ocr_text: str = "", profile_meta: dict = None,
                     farm_context: dict = None, stolen_db: LocalHashDB = None):
    """
    image_path: path to the photo/screenshot the user long-pressed
    ocr_text: text extracted from the screen (pass real OCR output -
              see run_real_ocr() below for a working example using
              tesseract, which IS installed and real in this sandbox)
    profile_meta: dict for profile_engine (see profile_engine.py docstring)
    farm_context: dict for farm_engine (see farm_engine.py docstring)
    stolen_db: a LocalHashDB instance (created once, reused across scans)
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"could not read image: {image_path}"}

    with open(image_path, "rb") as f:
        raw_bytes = f.read()
    image_hash = sha256_of_bytes(raw_bytes)

    results = []

    # 1. Visual engine (ELA, skin uniformity, catchlight, stolen-image hash)
    results.append(visual_engine.analyze(img, stolen_db=stolen_db, image_path_for_hash_lookup=image_path))

    # 2. Text engine
    results.append(text_engine.analyze(ocr_text))

    # 3. Profile engine
    results.append(profile_engine.analyze(profile_meta or {}))

    # 4. Farm engine (Type B/C/C-W/E)
    results.append(farm_engine.analyze(farm_context or {}))

    # 5. Gravity check
    face_angle, face_bbox = gravity_check.detect_face_angle(img)
    if face_angle is not None:
        bg_angle = gravity_check.detect_background_angle(img, exclude_bbox=face_bbox)
        if bg_angle is not None:
            delta = abs(bg_angle - abs(face_angle))
            if delta > gravity_check.TILT_THRESHOLD_DEG and abs(face_angle) < gravity_check.FACE_LEVEL_MAX_DEG:
                results.append([{
                    "type": "CRITICAL", "name": "GravityCheck",
                    "reason": f"Background tilt {delta:.1f}° while face level - re-photographed/AI composite",
                    "weight": 0.95,
                }])
            else:
                results.append([])
        else:
            results.append([])
    else:
        results.append([])

    risk = fusion_scorer.fuse(results)

    # Privacy: log hash+score only, never the image or OCR text
    privacy.log_only(image_hash, risk["score"], risk["verdict"])
    # NOTE: not calling secure_delete on the user's original photo here -
    # that's for a TEMP working copy the OS capture created, not the
    # user's own photo library file. Wire this to whatever temp buffer
    # your Android capture step creates.

    return risk


def run_real_ocr(image_path: str) -> str:
    """
    Real OCR using tesseract (installed and working in this sandbox) -
    use this to get real ocr_text instead of typing it in by hand.
    """
    import pytesseract
    img = cv2.imread(image_path)
    return pytesseract.image_to_string(img)
