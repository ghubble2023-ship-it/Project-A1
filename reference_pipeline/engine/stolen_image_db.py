"""
Stolen-image detection: real perceptual hash + real local database.

Replaces these placeholders from the original spec:
    function sha256(b){ return "hash"; }
    const localHashDB = { lookup: async (img) => { return null; } };

A cryptographic hash (sha256) is the WRONG tool for "is this the same
photo, possibly re-compressed/resized/cropped" - a single changed
pixel changes a sha256 hash completely. What you actually want for
stolen-photo matching is a perceptual hash, which stays similar for
visually similar images. Implemented below (dHash - difference hash),
no external package needed, plus a real local JSON store with
Hamming-distance matching so near-duplicates (recompressed, lightly
cropped, resized) are still caught.
"""

import json
import os
import hashlib
import cv2
import numpy as np

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "hash_db.json")
MATCH_HAMMING_THRESHOLD = 8  # out of 64 bits - looser catches more near-dupes, but more false positives


def dhash(bgr_image, hash_size=8):
    """
    Difference hash: resize to (hash_size+1) x hash_size, compare
    adjacent pixels. Returns an integer bit string as hex.
    Robust to resizing, mild recompression, and minor color shifts -
    NOT robust to mirroring or heavy cropping (documented limitation).
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return format(value, f"0{hash_size * hash_size}x")


def sha256_of_bytes(raw_bytes: bytes) -> str:
    """Real sha256 (for the log-only audit trail, not for photo matching)."""
    return hashlib.sha256(raw_bytes).hexdigest()


def hamming_distance(hex_a: str, hex_b: str) -> int:
    int_a = int(hex_a, 16)
    int_b = int(hex_b, 16)
    return bin(int_a ^ int_b).count("1")


class LocalHashDB:
    """
    Real local JSON-backed store, not a stub that always returns None.

    PRIVACY (fixed during review): this previously stored the full
    source file path of every image added, which conflicts with the
    project's non-negotiable rule that only hash + verdict + timestamp
    persist. A file path is identifying data about the user's device and
    library, and it does not need to persist for matching to work -
    the perceptual hash alone is sufficient.

    Source-path retention is now OFF by default. It can be enabled
    explicitly for local development/debugging, but should never be on
    in a shipped build.
    """

    def __init__(self, path: str = DEFAULT_DB_PATH, retain_source_paths: bool = False):
        self.path = path
        self.retain_source_paths = retain_source_paths
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                self.entries = json.load(f)
        else:
            self.entries = []

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.entries, f, indent=2)

    def add(self, image_path: str, label: str = ""):
        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"could not read {image_path}"}
        entry = {"phash": dhash(img), "label": label}
        if self.retain_source_paths:
            # Dev/debug only - see class docstring. Never in a shipped build.
            entry["source_path"] = image_path
        self.entries.append(entry)
        self._save()
        return entry

    def lookup(self, image_path: str):
        """
        Real replacement for the placeholder. Returns the closest match
        under the Hamming distance threshold, or None if nothing close
        enough is stored yet.
        """
        img = cv2.imread(image_path)
        if img is None:
            return None
        query_hash = dhash(img)

        best = None
        best_dist = None
        for entry in self.entries:
            dist = hamming_distance(query_hash, entry["phash"])
            if best_dist is None or dist < best_dist:
                best, best_dist = entry, dist

        if best is not None and best_dist <= MATCH_HAMMING_THRESHOLD:
            return {"match": best, "hamming_distance": best_dist}
        return None
