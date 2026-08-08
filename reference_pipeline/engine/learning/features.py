"""
Feature extraction for the teachable layer.

Pulls together every REAL signal we've built so far (gravity check,
skin uniformity, ELA, catchlight) into one numeric feature vector per
photo. This is what makes the system teachable instead of hardcoded:
instead of me guessing "delta > 8deg = fake," a real classifier learns
which features actually separate fake from real once you feed it
labeled examples - and it can weigh multiple weak signals together
(the whole point of "Eyes of Greg" - no single tell is proof, but six
weak tells together might be).

Honest limitation up front: several of these features depend on face/
eye detection succeeding (see gravity_check.py, visual_engine.py docs).
On real photos that fails often - glasses, off-angle heads, etc. Missing
values are a normal, expected part of this data, not a bug - handled
via imputation in train_classifier.py, not by pretending they're 0.
"""

import cv2
import numpy as np

from engine import gravity_check, visual_engine

FEATURE_NAMES = [
    "gravity_face_angle",
    "gravity_bg_angle",
    "gravity_delta",
    "ela_mean_error",
    "ela_max_error",
    "skin_uniformity_score",
    "catchlight_position_diff",
    "catchlight_color_diff",
]


def extract_features(image_path: str) -> dict:
    """
    Returns a dict with every key in FEATURE_NAMES. Values are floats
    or None (None = signal unavailable for this photo, e.g. no face
    detected - NOT the same as 0, and must not be treated as 0).
    """
    img = cv2.imread(image_path)
    if img is None:
        return {name: None for name in FEATURE_NAMES}

    features = {name: None for name in FEATURE_NAMES}

    # --- Gravity check ---
    face_angle, face_bbox = gravity_check.detect_face_angle(img)
    features["gravity_face_angle"] = face_angle
    if face_bbox is not None:
        bg_angle = gravity_check.detect_background_angle(img, exclude_bbox=face_bbox)
        features["gravity_bg_angle"] = bg_angle
        if face_angle is not None and bg_angle is not None:
            features["gravity_delta"] = abs(bg_angle - abs(face_angle))

    # --- ELA (always computable, no face needed) ---
    mean_err, max_err = visual_engine.error_level_analysis(img)
    features["ela_mean_error"] = mean_err
    features["ela_max_error"] = max_err

    # --- Skin uniformity + catchlight (need a detected face) ---
    face_bbox2 = visual_engine._detect_face_bbox(img)
    if face_bbox2 is not None:
        skin = visual_engine.skin_uniformity_score(img, face_bbox2)
        features["skin_uniformity_score"] = skin

        cl = visual_engine.catchlight_consistency(img, face_bbox2)
        if cl is not None:
            features["catchlight_position_diff"] = cl["position_diff"]
            features["catchlight_color_diff"] = cl["color_diff"]

    return features


def features_to_vector(features: dict):
    """Ordered list matching FEATURE_NAMES, for feeding to sklearn."""
    return [features.get(name) for name in FEATURE_NAMES]
