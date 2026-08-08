"""
Pupil shape analysis.

WHY THIS EXISTS: irregular pupil geometry is a genuinely documented
tell for GAN/AI-generated faces - generators don't enforce the
physical constraint that a human pupil is a near-circle, so synthetic
faces often show oval, lumpy, or spiky pupil outlines. Published
detection work has used exactly this.

WHY IT IS DANGEROUS, AND WHY THAT MATTERS MORE:
Real human eyes are ALSO irregular for real medical reasons -
traumatic injury, surgical iridectomy, coloboma, corectopia,
synechiae, and more. Greg supplied a real photo of a real person whose
pupil is deformed from injury and reads as "spiky," precisely to test
this. That is a HARD NEGATIVE: if a pupil-shape rule flags him, the
rule is not detecting AI, it is discriminating against people with eye
injuries and conditions.

So this module measures shape. It deliberately does NOT return a
fake/real verdict. Any threshold must be learned from a labeled set
that INCLUDES real deformed eyes on the "real" side, or it will
produce exactly the false accusation described above.
"""

import cv2
import numpy as np


def _pupil_mask(bgr_image, dark_percentile=6):
    """
    Builds a pupil mask that is NOT destroyed by the catchlight.

    Root cause found by testing against real photos: the specular
    reflection (catchlight) sits INSIDE the pupil and is bright, so a
    plain darkness threshold punches a hole straight through the pupil
    and often splits it into fragments. Shape measured on those
    fragments is meaningless - every real eye scored as wildly
    irregular, and one man's injury-deformed pupil actually scored MORE
    circular than his normal one.

    Fix: threshold for dark pixels, then (a) add back bright specular
    pixels that fall inside the pupil's neighbourhood, and (b) fill
    remaining interior holes. What's left is the pupil OUTLINE, which
    is what the shape metrics are supposed to be about.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    cutoff = np.percentile(gray, dark_percentile)
    _, dark = cv2.threshold(gray, cutoff, 255, cv2.THRESH_BINARY_INV)

    # Specular highlights: the genuinely brightest pixels in the frame.
    bright_cutoff = max(np.percentile(gray, 99.0), float(gray.max()) * 0.8)
    _, bright = cv2.threshold(gray, bright_cutoff, 255, cv2.THRESH_BINARY)

    # Only count highlights that are actually surrounded by pupil-dark
    # area, so a highlight elsewhere in the frame (skin, sclera, an
    # eyelid) doesn't get merged in. Dilating the dark mask gives the
    # pupil's neighbourhood.
    near_pupil = cv2.dilate(dark, np.ones((15, 15), np.uint8), iterations=2)
    bright_in_pupil = cv2.bitwise_and(bright, near_pupil)

    mask = cv2.bitwise_or(dark, bright_in_pupil)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Fill interior holes so the contour traces the outer boundary only
    filled = mask.copy()
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)

    return filled


def analyze_pupil_shape(image_path_or_image, dark_percentile=6):
    """
    Returns real geometric measurements of the darkest blob (pupil
    candidate) in a TIGHT iris-centered crop:

      circularity        4*pi*A/P^2   1.0 = perfect circle
      ellipse_aspect     major/minor axis of best-fit ellipse
                         1.0 = round, >1.2 = visibly oval
      solidity           area / convex-hull area
                         < ~0.95 means concave notches (spikes)
      n_spikes           count of significant convexity defects -
                         this is the direct measure of "spiky"
      max_spike_depth_frac  deepest defect as a fraction of radius

    Returns None if no usable dark blob is found. Does NOT judge
    real vs fake - see module docstring.
    """
    if isinstance(image_path_or_image, str):
        img = cv2.imread(image_path_or_image)
        if img is None:
            return None
    else:
        img = image_path_or_image

    mask = _pupil_mask(img, dark_percentile)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = img.shape[0] * img.shape[1]
    cand = [c for c in contours if cv2.contourArea(c) > img_area * 0.005]
    if not cand:
        return None

    # Selecting the LARGEST blob turned out to be wrong on real photos:
    # the eyelash / upper-lid shadow band is often as large as or larger
    # than the pupil, and being elongated it produced aspect ratios of
    # 4-5 that had nothing to do with pupil geometry.
    #
    # Anchor on the specular catchlight instead: physically, the
    # reflection sits inside the pupil/iris, so the blob containing (or
    # closest to) the brightest highlight is the pupil. This criterion
    # is independent of shape, so it does not bias the roundness
    # measurement it feeds - important, since selecting "the roundest
    # blob" would be circular reasoning.
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray_full, (5, 5), 0)
    _, _, _, bright_loc = cv2.minMaxLoc(gray_blur)

    def anchor_distance(contour):
        inside = cv2.pointPolygonTest(contour, (float(bright_loc[0]), float(bright_loc[1])), True)
        # pointPolygonTest returns +dist inside, -dist outside
        return -inside  # smaller is better: inside beats outside

    cnt = min(cand, key=anchor_distance)

    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    if perim == 0:
        return None
    circularity = 4 * np.pi * area / (perim ** 2)

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else None

    ellipse_aspect = None
    if len(cnt) >= 5:
        (_, _), (ax1, ax2), _ = cv2.fitEllipse(cnt)
        if min(ax1, ax2) > 0:
            ellipse_aspect = max(ax1, ax2) / min(ax1, ax2)

    _, radius = cv2.minEnclosingCircle(cnt)

    # Convexity defects = the actual "spikes"/notches
    # NOTE: an earlier version re-sorted the hull indices descending,
    # which made cv2.convexityDefects raise internally and silently
    # return 0 spikes for EVERY image - contradicting the solidity
    # numbers. Caught by cross-checking the two measures against each
    # other. cv2.convexHull already returns indices in the required
    # order, so it must be passed through unmodified.
    n_spikes = 0
    max_depth_frac = 0.0
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if hull_idx is not None and len(hull_idx) > 3:
        try:
            defects = cv2.convexityDefects(cnt, hull_idx)
        except cv2.error:
            defects = None
        if defects is not None and radius > 0:
            for i in range(defects.shape[0]):
                depth = defects[i, 0, 3] / 256.0
                frac = depth / radius
                if frac > 0.08:  # ignore pixel-level jitter
                    n_spikes += 1
                    max_depth_frac = max(max_depth_frac, frac)

    return {
        "circularity": round(float(circularity), 3),
        "ellipse_aspect": round(float(ellipse_aspect), 3) if ellipse_aspect else None,
        "solidity": round(float(solidity), 3) if solidity else None,
        "n_spikes": n_spikes,
        "max_spike_depth_frac": round(float(max_depth_frac), 3),
        "pupil_radius_px": round(float(radius), 1),
        "verdict": None,
        "verdict_note": (
            "Intentionally no verdict. Real eye injuries and conditions "
            "produce the same irregular geometry as AI artifacts. A "
            "threshold here must be learned from labeled data that "
            "includes real deformed eyes, or it will falsely accuse "
            "real people."
        ),
    }
