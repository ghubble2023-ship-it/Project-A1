"""
Extract spectral (frequency-domain) features for every photo in the
training corpus and write them to features_spectral.json.

Recreated after the fork corruption lost the original script; output
format matches what train_combined.py / train_final.py expect:
  [{"image_path": ..., "label": ..., "features": {...}}, ...]
"""
import os
import sys
import json
import glob
from multiprocessing import Pool

import cv2

sys.path.insert(0, "/app/ml")
import spectral_features as sf  # noqa: E402

DATASETS = {
    "fake": "/app/datasets/fake",
    "real": "/app/datasets/real",
}
OUT = "/app/ml/features_spectral.json"


def list_images(root):
    files = []
    for ext in ("jpg", "jpeg", "png", "webp"):
        files += glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)
        files += glob.glob(os.path.join(root, "**", f"*.{ext.upper()}"), recursive=True)
    return sorted(f for f in files if "__MACOSX" not in f)


def work(args):
    path, label = args
    try:
        img = cv2.imread(path)
        if img is None:
            raise ValueError("unreadable image")
        feats = sf.extract(img)
    except Exception as e:  # noqa: BLE001
        return {"image_path": path, "label": label, "error": str(e),
                "features": {k: None for k in sf.FEATURE_NAMES}}
    return {"image_path": path, "label": label, "features": feats}


if __name__ == "__main__":
    jobs = []
    for label, root in DATASETS.items():
        imgs = list_images(root)
        print(f"{label}: {len(imgs)} images")
        jobs += [(p, label) for p in imgs]

    with Pool(os.cpu_count()) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(work, jobs, chunksize=8), 1):
            rows.append(row)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    with open(OUT, "w") as f:
        json.dump(rows, f)
    errs = sum(1 for r in rows if "error" in r)
    print(f"wrote {len(rows)} rows ({errs} errors) -> {OUT}")
