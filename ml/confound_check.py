"""
Confound check: is the fake/real split distinguishable by JPEG encoding
alone? If so, any classifier trained on ELA is partly learning
"different encoder", not "AI-generated face".

Compares, per class:
  - file size (bytes)
  - JPEG quantization tables (hashed -> how many distinct tables, and
    whether fake/real share them)
  - estimated JPEG quality from the luma table
"""
import glob
import hashlib
import statistics
from collections import Counter

from PIL import Image

DATASETS = {"fake": "/app/datasets/fake", "real": "/app/datasets/real"}


def imgs(root):
    fs = glob.glob(root + "/**/*.jpg", recursive=True)
    return sorted(f for f in fs if "__MACOSX" not in f)


def qtable_sig(im):
    qt = getattr(im, "quantization", None)
    if not qt:
        return "none"
    blob = b"".join(bytes(v) for _, v in sorted(qt.items()))
    return hashlib.sha1(blob).hexdigest()[:10]


def luma_mean(im):
    qt = getattr(im, "quantization", None)
    if not qt or 0 not in qt:
        return None
    return statistics.mean(qt[0])


summary = {}
for label, root in DATASETS.items():
    files = imgs(root)
    sizes, lumas, sigs, modes = [], [], Counter(), Counter()
    for f in files:
        im = Image.open(f)
        im.load()
        sizes.append(__import__("os").path.getsize(f))
        sigs[qtable_sig(im)] += 1
        modes[im.mode] += 1
        lm = luma_mean(im)
        if lm is not None:
            lumas.append(lm)
    summary[label] = {"sigs": sigs, "sizes": sizes, "lumas": lumas, "modes": modes}
    print(f"\n=== {label} ({len(files)} files) ===")
    print(f"  mode: {dict(modes)}")
    print(f"  file size bytes: mean={statistics.mean(sizes):.0f} "
          f"median={statistics.median(sizes):.0f} "
          f"min={min(sizes)} max={max(sizes)}")
    if lumas:
        print(f"  luma qtable mean: mean={statistics.mean(lumas):.2f} "
              f"median={statistics.median(lumas):.2f} "
              f"distinct_tables={len(sigs)}")
    print(f"  top qtable signatures: {sigs.most_common(3)}")

fs, rs = set(summary["fake"]["sigs"]), set(summary["real"]["sigs"])
shared = fs & rs
print("\n--- VERDICT ---")
print(f"qtable signatures shared between classes: {len(shared)} "
      f"(fake-only {len(fs - rs)}, real-only {len(rs - fs)})")
if not shared:
    print("WARNING: the two classes use COMPLETELY DIFFERENT JPEG encoders.")
    print("         ELA-based features will be partly learning the encoder,")
    print("         not the face. Treat ELA weights with suspicion.")
else:
    print("OK: classes overlap in JPEG encoding, so ELA is not a pure giveaway.")

fsz, rsz = summary["fake"]["sizes"], summary["real"]["sizes"]
overlap_lo = max(min(fsz), min(rsz))
overlap_hi = min(max(fsz), max(rsz))
print(f"file-size range overlap: {overlap_lo}..{overlap_hi} bytes "
      f"({'overlapping' if overlap_hi > overlap_lo else 'DISJOINT - strong confound'})")
