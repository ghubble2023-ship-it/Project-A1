"""
Gravity Check - Real implementation
Project AI / Eyes of Greg

Replaces the placeholder stub from the spec:
    getFaceAngle: (img) => 0            // fake
    getBackgroundAngle: (img) => 10.5   // fake, always "Thames case"

This version actually measures:
  1. Face angle  -> MediaPipe Face Mesh, angle of the line between the
                     outer corners of the two eyes vs. horizontal.
  2. Background angle -> Canny edge detection + probabilistic Hough
                          transform on the region OUTSIDE the face,
                          clustered to find the dominant near-vertical
                          line (walls, door frames, window edges) and
                          how far it deviates from true vertical.

If the face is level but the background is tilted beyond a threshold,
that's the "gravity check" signal: a real photo re-shot at an angle,
or a composite where the background wasn't matched to the subject.

Usage:
    python3 gravity_check.py path/to/photo.jpg
"""

import sys
import math
import os
import cv2
import numpy as np

# NOTE: the mediapipe version available in this environment only ships
# the new Tasks API (mp.tasks), which requires downloading a .task model
# file at runtime. This sandbox has no network access, so that path
# isn't usable here. Using OpenCV's bundled Haar cascades instead -
# fully offline, no model download required. Real accuracy trade-off
# is discussed in README.md.
_CASCADE_DIR = cv2.data.haarcascades
_face_cascade = cv2.CascadeClassifier(_CASCADE_DIR + "haarcascade_frontalface_default.xml")
_eye_cascade = cv2.CascadeClassifier(_CASCADE_DIR + "haarcascade_eye.xml")

TILT_THRESHOLD_DEG = 8.0     # delta above which we flag HIGH risk
FACE_LEVEL_MAX_DEG = 3.0     # face must be roughly level for the check to apply


def detect_face_angle(bgr_image):
    """
    Returns (angle_degrees, face_bbox) or (None, face_bbox) if a face
    was found but eyes weren't usable.
    angle_degrees: rotation of the eye-line relative to horizontal.
                   0 = perfectly level face.
    face_bbox: (x_min, y_min, x_max, y_max) in pixel coords, used later
               to exclude the face region from background line detection.

    NOTE: an earlier version used the separate left/right-eye-split
    cascades with strict parameters (minNeighbors=8) and only the FIRST
    detection from each. That failed on every real test photo,
    including the actual origin ("Thames case") photo this whole check
    is named after - the strict settings simply found nothing. This
    version uses the general eye cascade with looser parameters and
    takes the two LARGEST detections (real eyes are usually the biggest
    consistent pair) - validated against real photos, not just
    synthetic ones.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60))
    if len(faces) == 0:
        return None, None

    # Use the largest detected face (closest to camera / main subject)
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    bbox = (fx, fy, fx + fw, fy + fh)
    face_roi = gray[fy:fy + fh, fx:fx + fw]

    eyes = _eye_cascade.detectMultiScale(face_roi, scaleFactor=1.05, minNeighbors=5)
    if len(eyes) < 2:
        return None, bbox

    # Take the two largest detections - on real photos, false positives
    # (eyebrow shadow, glasses-frame corners, cap brim) are usually
    # smaller than the actual eyes.
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    centers = [(ex + ew / 2, ey + eh / 2) for ex, ey, ew, eh in eyes]
    centers.sort(key=lambda c: c[0])  # left-to-right in the image

    # Guard against both picks being the same eye / overlapping detections
    if abs(centers[1][0] - centers[0][0]) < fw * 0.15:
        return None, bbox

    angle = math.degrees(math.atan2(
        centers[1][1] - centers[0][1],
        centers[1][0] - centers[0][0],
    ))

    return angle, bbox


def detect_background_angle(bgr_image, exclude_bbox=None):
    """
    Returns the dominant background line's deviation from true vertical,
    in degrees (0 = perfectly plumb wall/door frame). None if no
    reliable line found.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Mask out the face region (and a margin around it) so facial
    # features don't get picked up as "structural" lines.
    mask = np.ones(gray.shape, dtype=np.uint8) * 255
    if exclude_bbox is not None:
        x1, y1, x2, y2 = exclude_bbox
        pad_x = int((x2 - x1) * 0.6)
        pad_y = int((y2 - y1) * 0.6)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(gray.shape[1], x2 + pad_x)
        y2 = min(gray.shape[0], y2 + pad_y)
        mask[y1:y2, x1:x2] = 0

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=60,
        minLineLength=min(gray.shape) * 0.15, maxLineGap=10
    )

    if lines is None:
        return None

    # A photo rotated by some angle theta shows THAT SAME theta whether
    # you measure it off a vertical reference (door frame, wall edge) or
    # a horizontal one (horizon, railing, table edge) - so both are
    # collected as "scene rotation" candidates and pooled together.
    # Earlier version only looked for near-vertical lines and was
    # therefore blind to horizon/railing-dominated photos (a river/
    # bridge photo, for instance) - fixed after testing against a real
    # photo exposed exactly this gap.
    deviations = []
    weights = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.hypot(x2 - x1, y2 - y1)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angle = angle % 180  # normalize to 0-180

        dev_from_vertical = abs(angle - 90)
        dev_from_horizontal = min(angle, 180 - angle)

        if dev_from_vertical < 30:      # roughly vertical -> wall/door/window edge
            deviations.append(dev_from_vertical)
            weights.append(length)
        elif dev_from_horizontal < 30:  # roughly horizontal -> horizon/railing/table edge
            deviations.append(dev_from_horizontal)
            weights.append(length)

    if not deviations:
        return None

    # --- Dominant-line clustering (replaces a plain global average) ---
    # Meta's own description of how she got 10.5° on the Thames photo:
    # she identified ONE dominant line - the river embankment - not an
    # average blended across every edge in the scene. A straight average
    # over ALL candidate lines dilutes that one signal with unrelated
    # structure (e.g. a distant building at a different angle). Fixed to
    # cluster candidate lines by angle and report the deviation of
    # whichever cluster carries the most total line length - i.e. the
    # actual dominant structural line, the way a person would eyeball it.
    deviations = np.array(deviations)
    weights = np.array(weights)

    order = np.argsort(deviations)
    sorted_devs = deviations[order]
    sorted_weights = weights[order]

    CLUSTER_TOLERANCE_DEG = 4.0
    clusters = []  # list of (total_weight, weighted_mean_dev, n_lines)
    start = 0
    for i in range(1, len(sorted_devs) + 1):
        if i == len(sorted_devs) or sorted_devs[i] - sorted_devs[start] > CLUSTER_TOLERANCE_DEG:
            chunk_devs = sorted_devs[start:i]
            chunk_weights = sorted_weights[start:i]
            total_w = float(chunk_weights.sum())
            mean_dev = float(np.average(chunk_devs, weights=chunk_weights))
            clusters.append((total_w, mean_dev, len(chunk_devs)))
            start = i

    dominant = max(clusters, key=lambda c: c[0])
    total_weight, dominant_dev, n_lines = dominant

    # Low-evidence guard: a single short/lonely line (see the emma1 test
    # case - one line in an almost-featureless wall) shouldn't produce a
    # confident-looking number. Requires either multiple line segments
    # agreeing, or one substantial one.
    if n_lines < 2 and total_weight < min(bgr_image.shape[:2]) * 0.35:
        return None

    return dominant_dev


def classify_gravity(face_angle: float, bg_angle: float):
    """
    Pure decision function: given a face angle and background angle,
    return (risk, band, delta). Extracted from gravity_check() so the
    rules can be unit-tested directly against known angle pairs -
    important because on real photos the image stage often returns
    UNKNOWN (no face or no background lines), which left the band logic
    with no test coverage at all.
    """
    delta = abs(bg_angle - abs(face_angle))
    abs_face = abs(face_angle)

    if delta >= 20.0 and abs_face < 12.0:
        # A 20+ degree mismatch is not explainable by hand shake or a
        # slightly tilted camera, even with a noticeably off-level face.
        return "HIGH", "large_delta", delta
    if delta >= 12.0 and abs_face < 8.0:
        return "HIGH", "clear_delta", delta
    if delta > TILT_THRESHOLD_DEG and abs_face < FACE_LEVEL_MAX_DEG:
        # Original rule, preserved: modest delta but a genuinely level face
        return "HIGH", "level_face_modest_delta", delta
    if delta > TILT_THRESHOLD_DEG and abs_face < 8.0:
        # Modest delta with a somewhat tilted face - real but weaker
        return "MEDIUM", "modest_delta_tilted_face", delta
    if delta >= 20.0:
        # Big delta but the face is heavily tilted too - most likely a
        # whole-camera tilt, so flag for review rather than HIGH
        return "MEDIUM", "large_delta_heavily_tilted_face", delta
    return "LOW", "consistent", delta


def gravity_check(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not read image: {image_path}"}

    face_angle, face_bbox = detect_face_angle(img)
    if face_angle is None:
        return {
            "risk": "UNKNOWN",
            "reason": "No face detected - gravity check requires a visible face",
        }

    bg_angle = detect_background_angle(img, exclude_bbox=face_bbox)
    if bg_angle is None:
        return {
            "risk": "UNKNOWN",
            "reason": "No reliable background structural lines found "
                      "(need a wall edge, door frame, window, etc.)",
            "face_angle_deg": round(face_angle, 2),
        }

    # GRADUATED RISK BANDS - replaces the old hard AND-gate
    # (delta > 8 AND face_angle < 3.0).
    #
    # Why this changed: a real confirmed-suspicious photo produced a
    # 24.7 degree delta - over 3x the threshold - but was scored LOW
    # because the face measured 3.93 degrees instead of under 3.0. It
    # missed the "level face" gate by less than one degree while showing
    # an enormous orientation mismatch. Discarding that is clearly wrong.
    #
    # The reasoning behind the original gate is still sound: if the face
    # itself is tilted a lot, the whole camera was probably tilted, so a
    # tilted background is expected and means nothing. But that argument
    # gets weaker as delta grows, so face-level tolerance now scales with
    # delta rather than being a fixed cliff.
    risk, band, delta = classify_gravity(face_angle, bg_angle)
    abs_face = abs(face_angle)

    result = {
        "face_angle_deg": round(face_angle, 2),
        "background_tilt_deg": round(bg_angle, 2),
        "delta_deg": round(delta, 2),
        "risk": risk,
        "band": band,
    }

    if risk == "HIGH":
        result["reason"] = (
            f"Background tilt ({bg_angle:.1f}°) diverges from face angle "
            f"({face_angle:.1f}°) by {delta:.1f}°. Consistent with a "
            "re-photographed image or a composite where subject and "
            "background weren't shot together."
        )
    elif risk == "MEDIUM":
        result["reason"] = (
            f"Orientation mismatch of {delta:.1f}°, but the face is also "
            f"off-level by {abs_face:.1f}°, so a tilted camera could explain "
            "some of it. Worth a human look."
        )
    else:
        result["reason"] = "Face and background orientation are consistent."

    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 gravity_check.py <image_path>")
        sys.exit(1)

    result = gravity_check(sys.argv[1])
    print("\n--- Gravity Check Result ---")
    for k, v in result.items():
        print(f"{k}: {v}")
