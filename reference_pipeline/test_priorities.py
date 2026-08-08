"""
Tests for the Dakota-priority work (July 29-30 session).

Run: python3 test_priorities.py

Covers:
  1. Gravity check graduated bands (fixes the Bella brittleness bug)
  2. Opener move-sequence matching + false-positive resistance
  3. Profile social-graph rules (follow ratio, posts, engagement)
  4. Privacy fix in LocalHashDB
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from engine.gravity_check import classify_gravity
from engine.opener_matcher import score_opener
from engine.profile_engine import analyze as profile_analyze
from engine.stolen_image_db import LocalHashDB

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


print("=== 1. Gravity check graduated bands ===")
# Real measured values from actual photos
cases = [
    (0.0, 0.0, "LOW", "perfectly level"),
    (2.0, 3.0, "LOW", "normal hand shake"),
    (-3.6, 2.4, "LOW", "Thames actual measured values"),
    (-0.6, 10.1, "HIGH", "Horse actual - confirmed fake"),
    (3.93, 28.6, "HIGH", "Bella actual - THE BUG CASE"),
    (2.9, 11.0, "HIGH", "just inside old gate"),
    (3.1, 11.2, "MEDIUM", "just outside old gate - was wrongly LOW"),
    (15.0, 16.0, "LOW", "whole camera tilted, bg agrees"),
    (25.0, 26.0, "LOW", "heavy camera tilt, bg agrees"),
    (30.0, 55.0, "MEDIUM", "big delta but face heavily tilted"),
    (5.0, 20.0, "HIGH", "clear delta, mildly tilted face"),
    (10.0, 32.0, "HIGH", "large delta, face 10deg"),
    (13.0, 40.0, "MEDIUM", "large delta but face 13deg"),
]
for face, bg, expected, desc in cases:
    risk, band, delta = classify_gravity(face, bg)
    check(f"{desc}: {risk} (want {expected})", risk == expected)

print("\n=== 2a. Opener matching - real confirmed scam openers should score ===")
scam_openers = [
    ("Hello my name is Bella. How are you doing today?",
     {"unknown_sender": True, "unsolicited_first_contact": True, "photo_attached_unprompted": True}),
    ("How are you doing. Where are you located",
     {"unknown_sender": True, "unsolicited_first_contact": True}),
    ("Hello darling. How are you doing?",
     {"unknown_sender": True, "unsolicited_first_contact": True, "platform_safety_warning": True}),
    ("Hello. How are you doing. Thanks for the follow back I really appreciate it",
     {"unknown_sender": True, "unsolicited_first_contact": True, "platform_safety_warning": True}),
    ("Where are you from dear?",
     {"unknown_sender": True, "unsolicited_first_contact": True}),
]
for text, ctx in scam_openers:
    r = score_opener(text, context=ctx, recipient_name="Greg")
    check(f"'{text[:40]}...' -> {r['level']} (want MEDIUM or HIGH)",
          r["level"] in ("MEDIUM", "HIGH"))

print("\n=== 2b. Opener matching - legitimate messages must NOT flag ===")
# These are the false positives found during development. A stranger
# with a real, checkable reason for contact must not be flagged.
legit = [
    "Hi, this is Dave from the fingerjoint shop - your order is ready for pickup.",
    "Hello, I am reaching out about the truck you listed for sale. Is it still available?",
    "Hey, how are you? We met at the graduation last week - Sarah gave me your number.",
    "Good morning, this is Dr. Miller office confirming your appointment.",
]
for text in legit:
    r = score_opener(text, context={"unknown_sender": True, "unsolicited_first_contact": True},
                     recipient_name="Greg")
    check(f"legit stranger '{text[:38]}...' -> {r['level']} (want LOW)", r["level"] == "LOW")

print("\n=== 2c. Opener matching - no context means never actionable ===")
r = score_opener("Hello, how are you doing today?", context={}, recipient_name="Greg")
check(f"text pattern alone -> {r['level']} (want INFO, not actionable)", r["level"] == "INFO")

print("\n=== 3. Profile social-graph rules (real measured values) ===")
# Real confirmed scam profiles
angela = profile_analyze({"display_name": "Angela Peterson", "following": 2061,
                          "followers": 97, "post_count": 1, "likes_total": 18})
angela_types = {f["type"] for f in angela}
check("angela.peterson41 (21x ratio) fires followRatioImbalance",
      "followRatioImbalance" in angela_types)
check("angela.peterson41 (1 post) fires almostNoPosts", "almostNoPosts" in angela_types)
check("angela.peterson41 (18 likes/97 followers) fires lowEngagement",
      "lowEngagementForFollowerCount" in angela_types)

emmie = profile_analyze({"following": 9997, "followers": 1960, "account_suspended": True})
check("suspended account fires platformSuspendedAccount",
      "platformSuspendedAccount" in {f["type"] for f in emmie})

kate = profile_analyze({"display_name": "Kate Alicee", "following": 838, "followers": 831})
check("kate.alice670 (1.0x ratio) does NOT fire follow-ratio rule",
      "followRatioImbalance" not in {f["type"] for f in kate})

popular = profile_analyze({"display_name": "Sam Ortiz", "following": 300, "followers": 4000,
                           "post_count": 210, "likes_total": 50000})
check("popular real-shaped account does NOT fire social-graph rules",
      not {"followRatioImbalance", "almostNoPosts",
           "lowEngagementForFollowerCount"} & {f["type"] for f in popular})

check("empty meta returns no flags", profile_analyze({}) == [])

print("\n=== 4. Privacy: LocalHashDB must not retain source paths by default ===")
p = "/tmp/priv_test_suite.json"
if os.path.exists(p):
    os.remove(p)
import numpy as np
import cv2
img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.circle(img, (50, 50), 30, (200, 200, 200), -1)
cv2.imwrite("/tmp/priv_test_img.jpg", img)

db = LocalHashDB(path=p)
db.add("/tmp/priv_test_img.jpg", label="t")
keys = set(json.load(open(p))[0].keys())
check(f"default entry keys {sorted(keys)} exclude source_path", "source_path" not in keys)
check("default entry still has the hash (matching still works)", "phash" in keys)

os.remove(p)
db2 = LocalHashDB(path=p, retain_source_paths=True)
db2.add("/tmp/priv_test_img.jpg", label="t")
check("dev mode CAN opt into source paths",
      "source_path" in json.load(open(p))[0])
os.remove(p)

print(f"\n=== RESULTS: {PASS} passed, {FAIL} failed ===")
if FAIL:
    sys.exit(1)
