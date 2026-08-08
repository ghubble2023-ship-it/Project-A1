"""
Extract the reference-pipeline feature vector for every photo in the
uploaded training corpus and write it to features.json.

Runs the REAL code from /app/reference_pipeline/engine (gravity_check,
visual_engine) - no reimplementation, no shortcuts.
"""
import os
import sys
import json
import glob
from multiprocessing import Pool

REF = "/app/reference_pipeline"
sys.path.insert(0, REF)

from engine.learning.features import extract_features, FEATURE_NAMES  # noqa: E402

DATASETS = {
    "fake": "/app/datasets/fake",
    "real": "/app/datasets/real",
}
OUT = "/app/ml/features.json"


def list_images(root):
    files = []
    for ext in ("jpg", "jpeg", "png", "webp"):
        files += glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)
        files += glob.glob(os.path.join(root, "**", f"*.{ext.upper()}"), recursive=True)
    return sorted(f for f in files if "__MACOSX" not in f)


def work(args):
    path, label = args
    try:
        feats = extract_features(path)
    except Exception as e:  # noqa: BLE001
        return {"image_path": path, "label": label, "error": str(e),
                "features": {k: None for k in FEATURE_NAMES}}
    return {"image_path": path, "label": label, "features": feats}


if __name__ == "__main__":
    jobs = []
    for label, root in DATASETS.items():
        imgs = list_images(root)
        print(f"{label}: {len(imgs)} images")
        jobs += [(p, label) for p in imgs]

    with Pool(os.cpu_count()) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(work, jobs, chunksize=4), 1):
            rows.append(row)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    with open(OUT, "w") as f:
        json.dump(rows, f)
    print(f"wrote {len(rows)} rows -> {OUT}")

    # --- honest coverage report: how often each feature is actually available ---
    print("\n--- FEATURE COVERAGE (non-null count / total) ---")
    for name in FEATURE_NAMES:
        line = [name.ljust(26)]
        for label in ("fake", "real"):
            sub = [r for r in rows if r["label"] == label]
            n = sum(1 for r in sub if r["features"].get(name) is not None)
            line.append(f"{label}: {n}/{len(sub)}")
        print("  ".join(line))
