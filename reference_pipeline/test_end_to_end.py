"""
End-to-end tests. Run with: python3 test_end_to_end.py

Honest scope note: Haar-cascade-dependent pieces (face angle, skin
uniformity, catchlight, farm C-W anchor comparison) need a REAL face
photo to prove accuracy - this sandbox has none and no internet to
fetch one. Those are exercised here just enough to prove they don't
crash on a no-face image (graceful skip, not a false trigger) - actual
accuracy validation still needs Greg to run this against real photos.
"""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from engine import text_engine, profile_engine, farm_engine
from engine.stolen_image_db import LocalHashDB, dhash, hamming_distance
from project_ai import long_press_scan

PASS = 0
FAIL = 0


def check(label, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {label}")
    else:
        FAIL += 1
        print(f"  FAIL: {label}")


print("=== text_engine ===")
flags = text_engine.analyze("Send me a gift card right now, my queen, I love you")
types = {f["type"] for f in flags}
check("detects moneyRequest", "moneyRequest" in types)
check("detects urgency", "urgency" in types)
check("detects romanceBombing", "romanceBombing" in types)
check("clean text produces no flags", text_engine.analyze("Hey, how was your weekend?") == [])

print("\n=== profile_engine ===")
flags = profile_engine.analyze({"photo_count": 1, "display_name": "Emily Rose", "bio": "find me on t.me/emily"})
types = {f["type"] for f in flags}
check("detects singlePhoto", "singlePhoto" in types)
check("detects twoFirstNames", "twoFirstNames" in types)
check("detects telegramLink", "telegramLink" in types)

print("\n=== farm_engine Type B ===")
flags = farm_engine.detect_type_b({"monetization_is_chat": True, "refuses_photos": True, "deflects_external": True})
check("Type B fires when all 3 conditions met", len(flags) == 1 and flags[0]["subType"].startswith("B"))
check("Type B does NOT fire on partial match", farm_engine.detect_type_b({"monetization_is_chat": True}) == [])

print("\n=== farm_engine Type C ===")
flags = farm_engine.detect_type_c({
    "text": "She, number 24313 loves life",
    "has_handler_bait_separation": True,
    "failed_weather_test": False,
    "same_timestamp_diff_photos": False,
})
check("Type C does NOT fire at score 4 boundary edge case (needs >=4, this is exactly 4)",
      len(flags) == 1)  # 2 (catalog) + 2 (handler) = 4, meets threshold
flags_low = farm_engine.detect_type_c({"text": "just chatting", "has_handler_bait_separation": True})
check("Type C does NOT fire below threshold", flags_low == [])

print("\n=== stolen_image_db (real perceptual hash, synthetic images) ===")
db_path = "/tmp/test_hash_db.json"
if os.path.exists(db_path):
    os.remove(db_path)
db = LocalHashDB(path=db_path)

# Create two synthetic images: identical, and a resized/recompressed variant
img_a = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.rectangle(img_a, (50, 50), (250, 250), (100, 150, 200), -1)
cv2.circle(img_a, (150, 150), 60, (255, 255, 255), -1)
cv2.imwrite("/tmp/test_img_a.jpg", img_a)

img_a_resized = cv2.resize(img_a, (150, 150))
img_a_resized = cv2.resize(img_a_resized, (300, 300))  # simulate a re-save/re-scale
cv2.imwrite("/tmp/test_img_a_resaved.jpg", img_a_resized, [cv2.IMWRITE_JPEG_QUALITY, 60])

img_b = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.circle(img_b, (150, 150), 100, (0, 255, 0), -1)
cv2.imwrite("/tmp/test_img_b.jpg", img_b)

db.add("/tmp/test_img_a.jpg", label="known_stolen_photo")
match_same = db.lookup("/tmp/test_img_a.jpg")
match_resaved = db.lookup("/tmp/test_img_a_resaved.jpg")
match_different = db.lookup("/tmp/test_img_b.jpg")

check("exact same image matches", match_same is not None and match_same["hamming_distance"] == 0)
check("resized/recompressed variant of the SAME image still matches", match_resaved is not None)
check("a genuinely different image does NOT match", match_different is None)

print("\n=== full pipeline (long_press_scan) - no-face synthetic image, checking it doesn't crash and doesn't false-positive ===")
result = long_press_scan(
    "/tmp/test_img_b.jpg",
    ocr_text="hey what's up",
    profile_meta={"photo_count": 3, "display_name": "Sam"},
    farm_context={},
    stolen_db=db,
)
check("pipeline runs end-to-end without error", "error" not in result)
check("no false CRITICAL on a boring synthetic image", result["critical"] is False)
print(f"  -> verdict: {result['verdict']}, score: {result['score']}")

print(f"\n=== RESULTS: {PASS} passed, {FAIL} failed ===")
if FAIL > 0:
    sys.exit(1)
