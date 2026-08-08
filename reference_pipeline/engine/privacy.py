"""
Privacy layer - real hash-only logging, replacing:
    secureDelete: (bitmap) => { /* overwrite memory */ },
    logOnly: (obj) => { /* hash, score, timestamp only */ }
which did nothing at all.
"""

import json
import os
import time

LOG_PATH = os.path.join(os.path.dirname(__file__), "scan_log.json")


def log_only(image_sha256: str, score: float, verdict: str):
    """Appends hash+score+timestamp only - never the image itself, never OCR'd text."""
    entry = {"hash": image_sha256, "score": score, "verdict": verdict, "t": time.time()}
    log = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            log = json.load(f)
    log.append(entry)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    return entry


def secure_delete(temp_image_path: str):
    """
    Real best-effort overwrite-then-delete for a temp file (this sandbox's
    filesystem doesn't guarantee physical overwrite the way a real device's
    storage driver would, but this is the correct logical implementation -
    port directly to the platform's file API on-device).
    """
    if os.path.exists(temp_image_path):
        size = os.path.getsize(temp_image_path)
        with open(temp_image_path, "r+b") as f:
            f.write(os.urandom(size))
        os.remove(temp_image_path)
        return True
    return False
