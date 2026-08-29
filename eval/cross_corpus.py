"""Does a calibration fitted on one corpus survive contact with another?

This is the question that decides whether any of this is real. Leave-one-out
inside a single corpus tells you the signals separate *these* images. It says
nothing about whether they separate images from a different generator, a
different camera, or a different collection process -- and in AI-image
forensics that is exactly where methods die.

So: fit on corpus A, score corpus B, never letting a single image from B touch
the calibration. Report the drop. A method that scores 0.95 within-corpus and
0.55 across is a method that learned the corpus, and saying so plainly is more
useful than shipping the 0.95.

Both corpora are additionally re-fitted in the pooled direction, because a
signal that flips sign between corpora is worse than a weak one: it will
confidently mislabel whichever population it was not fitted on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_eval import auc, extract, fit_calibration, summarise  # noqa: E402
from sentinel.types import sigmoid  # noqa: E402


def score_with(cal, rows: list[dict]) -> tuple[list[float], list[str]]:
    scores, labels = [], []
    for row in rows:
        total = cal.prior_log_odds
        for key, val in row["values"].items():
            entry = cal.entries.get(key)
            if entry is None or entry.weight == 0.0:
                continue
            total += entry.llr(float(val)) * entry.weight
        scores.append(sigmoid(total))
        labels.append(row["label"])
    return scores, labels


def direction_table(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Per-signal AUC in each corpus, flagging sign flips."""
    def per_signal(rows):
        out = {}
        keys = {k for r in rows for k in r["values"]}
        for k in keys:
            pos = [r["values"][k] for r in rows if r["label"] == "synthetic" and k in r["values"]]
            neg = [r["values"][k] for r in rows if r["label"] == "authentic" and k in r["values"]]
            if len(pos) >= 3 and len(neg) >= 3:
                out[k] = {"auc": round(auc(pos, neg), 3), "n": len(pos) + len(neg)}
        return out

    a, b = per_signal(rows_a), per_signal(rows_b)
    table = {}
    for k in sorted(set(a) | set(b)):
        ea, eb = a.get(k), b.get(k)
        row = {
            "auc_train_corpus": ea["auc"] if ea else None,
            "auc_test_corpus": eb["auc"] if eb else None,
        }
        if ea and eb:
            # A flip means the signal points at synthetic in one corpus and at
            # authentic in the other. Those are actively dangerous.
            row["sign_flip"] = (ea["auc"] - 0.5) * (eb["auc"] - 0.5) < 0
            row["stable"] = (
                not row["sign_flip"]
                and abs(ea["auc"] - 0.5) >= 0.12
                and abs(eb["auc"] - 0.5) >= 0.12
            )
        table[k] = row
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True, help="view directory to fit on")
    ap.add_argument("--test", required=True, help="view directory to score")
    ap.add_argument("--train-name", default="train")
    ap.add_argument("--test-name", default="test")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    train_rows = extract(args.train)
    test_rows = extract(args.test)

    cal, train_table = fit_calibration(train_rows, {"view": args.train_name})

    # Sanity floor: how the training corpus scores under its own calibration.
    # This is optimistic by construction and is printed only for contrast.
    in_scores, in_labels = score_with(cal, train_rows)
    in_sample = summarise(in_scores, in_labels)

    out_scores, out_labels = score_with(cal, test_rows)
    transfer = summarise(out_scores, out_labels)

    directions = direction_table(train_rows, test_rows)
    flips = [k for k, v in directions.items() if v.get("sign_flip")]
    stable = [k for k, v in directions.items() if v.get("stable")]

    print(f"\n=== transfer: fit on {args.train_name} -> score {args.test_name} ===")
    print(f"train corpus: {len(train_rows)} images, test corpus: {len(test_rows)} images\n")
    print(f"{'signal':32s} {'AUC ' + args.train_name[:9]:>14s} {'AUC ' + args.test_name[:9]:>14s}  note")
    for k, v in sorted(
        directions.items(),
        key=lambda kv: -abs((kv[1]['auc_train_corpus'] or 0.5) - 0.5),
    ):
        note = ""
        if v.get("sign_flip"):
            note = "SIGN FLIP -- direction reverses between corpora"
        elif v.get("stable"):
            note = "stable"
        ta = v["auc_train_corpus"]
        te = v["auc_test_corpus"]
        print(
            f"{k:32s} {('%.3f' % ta) if ta is not None else '   -  ':>14s} "
            f"{('%.3f' % te) if te is not None else '   -  ':>14s}  {note}"
        )

    print(f"\nin-sample on {args.train_name} (optimistic, for contrast):")
    print(f"  AUC {in_sample['auc']}  acc {in_sample['best_operating_point']['accuracy']}")
    print(f"\nTRANSFER to {args.test_name} (the number that matters):")
    print(json.dumps(transfer, indent=2))
    print(f"\nsignals stable across both corpora: {stable or 'none'}")
    print(f"signals that reverse direction:     {flips or 'none'}")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "train": args.train_name,
                    "test": args.test_name,
                    "n_train": len(train_rows),
                    "n_test": len(test_rows),
                    "in_sample_train": in_sample,
                    "transfer_to_test": transfer,
                    "per_signal_direction": directions,
                    "stable_signals": stable,
                    "sign_flipped_signals": flips,
                },
                fh,
                indent=2,
            )
        print(f"\nwrote report -> {args.report}")


if __name__ == "__main__":
    main()
