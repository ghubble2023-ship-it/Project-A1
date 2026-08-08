"""
Visual Engine - forensic image analysis.

IMPORTANT HONESTY NOTE, read before trusting this module:

The original spec had checkTooPerfectFlat, checkEmotionallyEmpty,
checkHairUnnatural, checkSkinEven, checkReflectionError, checkWaxFigure
all defined like this:

    checkTooPerfectFlat: (img) => [{type:"greg_tell", tell:"...", weight: 0.8}]

Notice the `img` parameter is never used - every one of these fired
unconditionally on every single photo, regardless of content. That's
a hidden stub dressed up as a rule.

Below, THREE of those six are replaced with real, computable signals:
  - skin texture uniformity  (real proxy for "too perfect / too flat / too even")
  - eye catchlight consistency (real proxy for "reflection error")
  - ELA (error level analysis) (real proxy for "photoshop/composite")

The other THREE - emotionally empty eyes, wax-figure quality, and
unnatural hair - do not have a reliable lightweight offline heuristic.
Faking one would be exactly as dishonest as the original stub, just
dressed up in more code. They're left as explicit not-yet-automated
items below rather than silently always-firing or silently dropped.
Getting these right for real would need a trained model (facial
expression / material classification), which needs training data this
sandbox doesn't have.

Deepfake detection ("checkDeepfakeSignals" in the original) is NOT a
solved problem with hand-written heuristics - real deepfake detectors
are trained neural networks. What's here is a forensic heuristic
composite (ELA + skin uniformity), useful as a first-pass signal, not
a substitute for a trained classifier.
"""

import cv2
import numpy as np
import tempfile
import os

_CASCADE_DIR = cv2.data.haarcascades
_face_cascade = cv2.CascadeClassifier(_CASCADE_DIR + "haarcascade_frontalface_default.xml")
_eye_cascade = cv2.CascadeClassifier(_CASCADE_DIR + "haarcascade_eye.xml")


# ---------- Real signal #1: Error Level Analysis (photoshop/composite proxy) ----------

def error_level_analysis(bgr_image, quality=90):
    """
    Resaves the image at a known JPEG quality and diffs against the
    original. Regions that were edited/composited after the original
    save tend to show a different error level than untouched regions -
    a well-established, genuinely real forensic technique (not a proof
    of manipulation on its own, but a real signal).
    Returns mean_error (float) and a normalized error map.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, bgr_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        resaved = cv2.imread(tmp_path)
    finally:
        os.remove(tmp_path)

    diff = cv2.absdiff(bgr_image, resaved).astype(np.float32)
    error_map = diff.mean(axis=2)
    mean_error = float(error_map.mean())
    max_error = float(error_map.max())
    return mean_error, max_error


# ---------- Real signal #2: skin texture uniformity ----------

def skin_uniformity_score(bgr_image, face_bbox):
    """
    Real, computable proxy for 'too perfect / too flat / too even'.
    Natural skin has fine, irregular micro-texture (pores, small
    variations). Over-smoothed or synthetic skin shows unusually LOW
    local variance. Measures Laplacian variance in cheek/forehead
    patches within the detected face box.
    Lower score = more suspiciously smooth.
    """
    x1, y1, x2, y2 = face_bbox
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    face = gray[y1:y2, x1:x2]
    if face.size == 0:
        return None

    h, w = face.shape
    # Sample forehead (top-center) and both cheeks (mid-left, mid-right)
    patches = [
        face[int(h * 0.10):int(h * 0.25), int(w * 0.35):int(w * 0.65)],   # forehead
        face[int(h * 0.45):int(h * 0.65), int(w * 0.10):int(w * 0.30)],   # left cheek
        face[int(h * 0.45):int(h * 0.65), int(w * 0.70):int(w * 0.90)],   # right cheek
    ]
    variances = [cv2.Laplacian(p, cv2.CV_64F).var() for p in patches if p.size > 0]
    if not variances:
        return None
    return float(np.mean(variances))


# ---------- Real signal #3: eye catchlight consistency ----------

def catchlight_consistency(bgr_image, face_bbox):
    """
    Real proxy for 'reflection error'. Finds the brightest point
    (the catchlight/reflection) in each eye and compares position
    (normalized within the eye box) and color. In a single real photo
    taken in one lighting setup, catchlights are usually consistent in
    position and color between the two eyes. Composited/generated
    faces sometimes mismatch this.
    Returns None if fewer than 2 eyes detected.
    """
    x1, y1, x2, y2 = face_bbox
    gray_full = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    face_gray = gray_full[y1:y2, x1:x2]

    eyes = _eye_cascade.detectMultiScale(face_gray, 1.1, 8)
    if len(eyes) < 2:
        return None
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

    face_color = bgr_image[y1:y2, x1:x2]
    catchlights = []
    for ex, ey, ew, eh in eyes:
        eye_patch_gray = face_gray[ey:ey + eh, ex:ex + ew]
        eye_patch_color = face_color[ey:ey + eh, ex:ex + ew]
        if eye_patch_gray.size == 0:
            continue
        _, max_val, _, max_loc = cv2.minMaxLoc(eye_patch_gray)
        norm_pos = (max_loc[0] / ew, max_loc[1] / eh)
        color_at_max = eye_patch_color[max_loc[1], max_loc[0]].tolist()
        catchlights.append({"norm_pos": norm_pos, "brightness": max_val, "color": color_at_max})

    if len(catchlights) < 2:
        return None

    pos_diff = float(np.hypot(
        catchlights[0]["norm_pos"][0] - catchlights[1]["norm_pos"][0],
        catchlights[0]["norm_pos"][1] - catchlights[1]["norm_pos"][1],
    ))
    color_diff = float(np.linalg.norm(
        np.array(catchlights[0]["color"]) - np.array(catchlights[1]["color"])
    ))
    return {"position_diff": pos_diff, "color_diff": color_diff, "catchlights": catchlights}


def _detect_face_bbox(bgr_image):
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return (fx, fy, fx + fw, fy + fh)


def analyze(bgr_image, stolen_db=None, image_path_for_hash_lookup=None):
    """
    Full visual engine pass. Returns flags in the same shape as the
    original spec used elsewhere in the pipeline.
    """
    flags = []
    face_bbox = _detect_face_bbox(bgr_image)

    # --- ELA / composite check ---
    mean_err, max_err = error_level_analysis(bgr_image)
    if max_err > 25.0:  # threshold is a starting point, not empirically tuned yet
        flags.append({
            "type": "forensics", "tell": "ELA", "weight": 0.5,
            "mean_error": round(mean_err, 2), "max_error": round(max_err, 2),
            "note": "Elevated error-level variance - worth a closer look, not proof of editing.",
        })

    if face_bbox is not None:
        # --- skin uniformity ("too perfect flat / too even") ---
        skin_score = skin_uniformity_score(bgr_image, face_bbox)
        if skin_score is not None and skin_score < 15.0:  # starting threshold, needs real-photo calibration
            flags.append({
                "type": "greg_tell", "tell": "skin unusually smooth/uniform", "weight": 0.6,
                "skin_variance_score": round(skin_score, 2),
            })

        # --- catchlight consistency ("reflection error") ---
        cl = catchlight_consistency(bgr_image, face_bbox)
        if cl is not None and (cl["position_diff"] > 0.35 or cl["color_diff"] > 80):
            flags.append({
                "type": "greg_tell", "tell": "reflection/catchlight mismatch between eyes", "weight": 0.7,
                "position_diff": round(cl["position_diff"], 3), "color_diff": round(cl["color_diff"], 1),
            })

    # --- stolen image check (real perceptual hash lookup, not a stub) ---
    if stolen_db is not None and image_path_for_hash_lookup is not None:
        match = stolen_db.lookup(image_path_for_hash_lookup)
        flags.append({"type": "stolenCheck", "hashMatch": match})

    # --- honestly not automated yet - see module docstring ---
    flags.append({
        "type": "not_automated",
        "tells": ["emotionally_empty_eyes", "wax_figure_quality", "hair_unnatural"],
        "reason": "No reliable lightweight offline heuristic for these - needs a trained model or your own review.",
    })

    return flags
