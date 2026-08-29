"""Builds the two dataset views the evaluation needs.

The labelled folders that ship with this project are badly confounded: the
authentic set is 256x256 aligned face thumbnails, and the synthetic set is
640-1280px full-scene selfies. Resolution, JPEG history and framing all
correlate perfectly with the label, so *any* statistic will separate them and
almost none of that separation is about authenticity.

Two views are therefore produced:

``raw``
    The files as given. Useful only as a demonstration of how badly a naive
    evaluation overstates itself.

``controlled``
    Both classes put through the identical pipeline: detect the face, crop a
    fixed-margin box around it, resample to 256x256, re-encode as JPEG at one
    fixed quality. After this the two classes differ in content, not in
    capture pipeline, and a score above chance means something.

Any honest headline number for this system comes from ``controlled``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel.detect import shared_detector  # noqa: E402

TARGET_DIM = 256
JPEG_QUALITY = 88
FACE_MARGIN = 0.45  # fraction of face size added on each side


def controlled_crop(bgr: np.ndarray) -> np.ndarray | None:
    """Face-centred square crop, resampled to a fixed size. None if no face."""
    det = shared_detector()
    face = det.detect(bgr)
    if face is None:
        return None

    h, w = bgr.shape[:2]
    side = int(max(face.w, face.h) * (1.0 + 2 * FACE_MARGIN))
    ccx = face.x + face.w / 2.0
    ccy = face.y + face.h / 2.0
    x0 = int(round(ccx - side / 2.0))
    y0 = int(round(ccy - side / 2.0))

    # Reflect-pad rather than clamp, so the face stays centred even when the
    # crop runs off the edge. Clamping would shift the optical centre and
    # corrupt the chromatic-aberration test.
    pad = max(0, -x0, -y0, x0 + side - w, y0 + side - h)
    if pad > 0:
        bgr = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        x0 += pad
        y0 += pad

    crop = bgr[y0 : y0 + side, x0 : x0 + side]
    if crop.size == 0:
        return None
    interp = cv2.INTER_AREA if side > TARGET_DIM else cv2.INTER_LANCZOS4
    return cv2.resize(crop, (TARGET_DIM, TARGET_DIM), interpolation=interp)


def build(src_dirs: dict[str, str], out_root: str) -> dict:
    manifest = {"raw": [], "controlled": [], "skipped": []}

    for label, src in src_dirs.items():
        for view in ("raw", "controlled"):
            os.makedirs(os.path.join(out_root, view, label), exist_ok=True)

        for name in sorted(os.listdir(src)):
            path = os.path.join(src, name)
            if not os.path.isfile(path):
                continue
            data = open(path, "rb").read()
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                manifest["skipped"].append({"path": path, "why": "undecodable"})
                continue

            raw_out = os.path.join(out_root, "raw", label, name)
            with open(raw_out, "wb") as fh:
                fh.write(data)
            manifest["raw"].append({"path": raw_out, "label": label, "source": path})

            crop = controlled_crop(bgr)
            if crop is None:
                manifest["skipped"].append({"path": path, "why": "no face detected"})
                continue
            ctrl_out = os.path.join(
                out_root, "controlled", label, os.path.splitext(name)[0] + ".jpg"
            )
            cv2.imwrite(ctrl_out, crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            manifest["controlled"].append(
                {"path": ctrl_out, "label": label, "source": path}
            )

    with open(os.path.join(out_root, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--authentic", required=True, help="directory of authentic images")
    ap.add_argument("--synthetic", required=True, help="directory of synthetic images")
    ap.add_argument("--out", required=True, help="output dataset root")
    args = ap.parse_args()

    m = build(
        {"authentic": args.authentic, "synthetic": args.synthetic}, args.out
    )
    for view in ("raw", "controlled"):
        counts: dict[str, int] = {}
        for row in m[view]:
            counts[row["label"]] = counts.get(row["label"], 0) + 1
        print(f"{view:12s} {counts}")
    print(f"skipped      {len(m['skipped'])}")
    for s in m["skipped"]:
        print(f"  - {os.path.basename(s['path'])}: {s['why']}")


if __name__ == "__main__":
    main()
