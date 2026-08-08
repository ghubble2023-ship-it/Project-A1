"""
Eye closeup engine - for extreme close-up eye photos where there's no
full face to detect (the existing visual_engine.catchlight_consistency
requires a face bbox first, which won't exist in a pure iris/pupil
crop).

Real technique here: catchlight (the bright reflection in the eye)
analysis is a genuinely established forensic/CGI-detection method -
real eyes reflect real light sources, and the reflection's shape,
count, and position follow real optical rules. Composited or
AI-generated eyes sometimes show catchlights that are the wrong shape,
inconsistent between two eyes in the same photo, or physically odd
(e.g. a perfectly soft circular highlight with no environmental
structure in it, when a real catchlight of a window or screen often
shows some internal structure - visible window mullions, screen
edges, etc, exactly like the window reflections visible in some of
Greg's own real eye photos).

This does NOT detect "fake" on its own - it extracts real, measurable
properties. What separates real from fake (if anything) still needs to
be learned from labeled examples, same as everywhere else in this
project.
"""

import cv2
import numpy as np


def detect_iris_and_pupil(bgr_image):
    """
    Returns dict with pupil circle (x, y, r), or None if no reliable
    dark, circular region found. Works directly on an eye close-up
    crop - no face detection needed.

    Uses contour-based dark-blob detection with a circularity filter,
    not Hough circles. Found via testing across 5 real photos at very
    different zoom levels (extreme iris-fills-frame close-ups vs a
    wider single-eye crop) that Hough circles needed different radius
    ranges for each and had no reliable way to pick the right one
    automatically - it kept either missing the real pupil entirely or
    latching onto a large spurious circle (a lighting gradient, a
    UI element). Thresholding for genuinely dark regions + filtering by
    how circular the resulting blob's outline actually is generalizes
    across zoom level without needing a fixed radius range.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    h, w = gray.shape

    # Mask out the top-left corner: real test data (Greg's online
    # screenshots) had a rounded back-arrow UI button there, which was
    # dark enough to compete with the real pupil - found via testing.
    corner_h, corner_w = int(h * 0.20), int(w * 0.20)
    gray[0:corner_h, 0:corner_w] = 255

    # The pupil is genuinely one of the darkest regions in an eye
    # photo regardless of zoom level - use a percentile-based
    # threshold so it adapts to each photo's own brightness range
    # instead of a fixed cutoff.
    dark_cutoff = np.percentile(gray, 8)
    _, mask = cv2.threshold(gray, dark_cutoff, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    img_area = h * w
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.0008 or area > img_area * 0.5:
            continue  # too tiny to be a real pupil, or suspiciously huge
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.40:  # real pupils are fairly round even if eyelids clip them
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        candidates.append((cx, cy, radius, area, circularity))

    if not candidates:
        return None

    # Prefer the largest sufficiently-circular dark blob - the pupil is
    # usually the single biggest genuinely-dark round region in an eye
    # closeup; smaller dark specs (eyelashes, shadows) are noise.
    best = max(candidates, key=lambda c: c[3])

    return {"pupil": {"x": int(best[0]), "y": int(best[1]), "r": int(best[2])}}


def analyze_catchlight_shape(bgr_image, pupil):
    """
    Finds bright spots (catchlights) within/near the pupil+iris region
    and characterizes them: count, total area, largest blob's
    elongation (aspect ratio - real point-source reflections are often
    fairly compact; window/screen reflections can show real structure).
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    x, y, r = pupil["x"], pupil["y"], pupil["r"]

    pad = int(r * 2.2)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(gray.shape[1], x + pad), min(gray.shape[0], y + pad)
    region = gray[y1:y2, x1:x2]

    if region.size == 0:
        return None

    # RELATIVE threshold, not a hard absolute floor - found via testing
    # that a real (if lower-contrast/dimmer) photo had a genuine
    # catchlight that never crossed a fixed floor of 200. Basing the
    # cutoff on this region's own brightest pixel instead means it
    # still finds the catchlight in a dim photo, while still requiring
    # it to be meaningfully brighter than the surrounding iris.
    region_max = float(region.max())
    thresh_val = max(np.percentile(region, 97), region_max * 0.75)
    _, bright_mask = cv2.threshold(region, thresh_val, 255, cv2.THRESH_BINARY)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_mask)
    blobs = []
    for i in range(1, n_labels):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 2:
            continue
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(bw, bh) / max(1, min(bw, bh))
        cx, cy = centroids[i]
        blobs.append({
            "area": int(area),
            "aspect_ratio": round(float(aspect), 2),
            "position_in_region": (round(float(cx / region.shape[1]), 3), round(float(cy / region.shape[0]), 3)),
        })

    blobs.sort(key=lambda b: b["area"], reverse=True)

    return {
        "n_catchlight_blobs": len(blobs),
        "total_bright_area": sum(b["area"] for b in blobs),
        "largest_blob": blobs[0] if blobs else None,
        "all_blobs": blobs,
    }


def analyze_eye_image(image_path):
    """
    Full analysis for one eye closeup image. If two eyes are both
    visible and both get a pupil detection, also returns a direct
    between-eye comparison (position/aspect symmetry) - real signal,
    same idea as visual_engine.catchlight_consistency but not
    dependent on a face detector.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"could not read {image_path}"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Try whole-image detection first (single eye fills the frame)
    result = detect_iris_and_pupil(img)
    eyes_found = []
    if result is not None:
        eyes_found.append(result["pupil"])

    # If the image is wide (roughly 2x tall as it is wide's inverse -
    # i.e. two-eyes-side-by-side framing), also try left half / right
    # half separately, since HoughCircles above may only have picked
    # up one of the two.
    two_eye_layout = (w / h) > 1.4
    if two_eye_layout:
        left_half = img[:, :w // 2]
        right_half = img[:, w // 2:]
        left_result = detect_iris_and_pupil(left_half)
        right_result = detect_iris_and_pupil(right_half)
        eyes_found = []
        if left_result is not None:
            eyes_found.append(left_result["pupil"])
        if right_result is not None:
            p = right_result["pupil"]
            p["x"] += w // 2  # offset back into full-image coordinates
            eyes_found.append(p)

    if not eyes_found:
        return {"error": "no pupil/iris detected", "two_eye_layout": two_eye_layout}

    catchlights = []
    for pupil in eyes_found:
        cl = analyze_catchlight_shape(img, pupil)
        catchlights.append(cl)

    output = {
        "n_eyes_detected": len(eyes_found),
        "two_eye_layout": two_eye_layout,
        "catchlights": catchlights,
    }

    if len(catchlights) == 2 and all(c is not None for c in catchlights):
        c0, c1 = catchlights
        output["between_eye_comparison"] = {
            "blob_count_match": c0["n_catchlight_blobs"] == c1["n_catchlight_blobs"],
            "area_ratio": round(
                (c0["total_bright_area"] + 1) / (c1["total_bright_area"] + 1), 2
            ),
        }

    return output


_face_cascade_for_eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_eye_cascade_for_eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def analyze_face_eyes(image_path):
    """
    Entry point for a normal FULL-FACE photo (not a tight eye-only
    crop) - locates each eye first via the face+eye Haar cascades
    (same approach as farm_engine/visual_engine), then runs pupil/
    catchlight analysis inside each located eye box.

    Added after testing exposed a real gap: analyze_eye_image()'s
    whole-image dark-blob search works well on a tight eye crop, but
    on a full-face photo it just finds whatever the single largest
    dark circular region in the WHOLE frame is - which turned out to
    be a shadow near the shoulder/clothing, not either eye at all.
    This locates the eye region first instead of searching blind.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"could not read {image_path}"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    faces = _face_cascade_for_eyes.detectMultiScale(gray_eq, 1.05, 4, minSize=(60, 60))
    if len(faces) == 0:
        return {"error": "no face detected"}
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    face_roi_gray = gray_eq[fy:fy + fh, fx:fx + fw]

    eyes = _eye_cascade_for_eyes.detectMultiScale(face_roi_gray, 1.05, 5)
    if len(eyes) < 1:
        return {"error": "face found but no eyes located within it"}

    # Keep the largest 2 (avoids small false positives like nostril shadow)
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

    results = []
    for (ex, ey, ew, eh) in eyes:
        # Crop with generous padding so the iris/pupil detector has
        # context, in full-image coordinates
        pad = int(max(ew, eh) * 0.3)
        x1 = max(0, fx + ex - pad)
        y1 = max(0, fy + ey - pad)
        x2 = min(img.shape[1], fx + ex + ew + pad)
        y2 = min(img.shape[0], fy + ey + eh + pad)
        eye_crop = img[y1:y2, x1:x2]

        pupil = detect_iris_and_pupil(eye_crop)
        if pupil is None:
            results.append({"error": "no pupil found in this eye crop", "bbox": (x1, y1, x2, y2)})
            continue
        cl = analyze_catchlight_shape(eye_crop, pupil["pupil"])
        results.append({"pupil": pupil["pupil"], "catchlight": cl, "bbox": (x1, y1, x2, y2)})

    output = {"n_eyes_located": len(eyes), "eyes": results}

    valid = [r for r in results if "catchlight" in r and r["catchlight"] is not None]
    if len(valid) == 2:
        c0, c1 = valid[0]["catchlight"], valid[1]["catchlight"]
        output["between_eye_comparison"] = {
            "blob_count_match": c0["n_catchlight_blobs"] == c1["n_catchlight_blobs"],
            "area_ratio": round((c0["total_bright_area"] + 1) / (c1["total_bright_area"] + 1), 2),
        }

    return output
