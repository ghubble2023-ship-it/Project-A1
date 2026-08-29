"""Measures every signal, fits the calibration, and reports honest accuracy.

What this does, in order:

1. Extract every signal from every image in a dataset view.
2. Report, per signal, how often it could be measured at all and what AUC it
   achieves on its own. This is the table that decides which physics actually
   earns its place, rather than which physics sounds most convincing.
3. Fit the shipping calibration (per-signal Gaussians under each hypothesis).
4. Report the **leave-one-out cross-validated** accuracy of the fused score.
   Fitting and scoring on the same images would be circular; with a set this
   small, leave-one-out is the honest way to get a number.

Feature extraction happens once and is reused across all LOO folds, so the
cost is one pass over the images regardless of fold count.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel.fusion import Calibration, SignalCalibration  # noqa: E402
from sentinel.image.pipeline import run_modules  # noqa: E402
from sentinel.types import sigmoid  # noqa: E402

MIN_PER_CLASS = 5  # a signal needs this many measurements per class to fit


def auc(pos: list[float], neg: list[float]) -> float:
    """Rank-based AUC (Mann-Whitney U), ties counted as half."""
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def extract(view_dir: str) -> list[dict]:
    rows = []
    for label in sorted(os.listdir(view_dir)):
        sub = os.path.join(view_dir, label)
        if not os.path.isdir(sub):
            continue
        for name in sorted(os.listdir(sub)):
            path = os.path.join(sub, name)
            data = open(path, "rb").read()
            reports = run_modules(data)
            values = {
                s.key: s.value
                for r in reports
                for s in r.signals
                if s.available and s.value is not None
            }
            rows.append({"path": path, "label": label, "values": values})
    return rows


def fit_calibration(rows: list[dict], meta: dict) -> tuple[Calibration, dict]:
    """Fit per-signal Gaussians and report each signal's standalone AUC."""
    by_signal: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"authentic": [], "synthetic": []}
    )
    for row in rows:
        for key, val in row["values"].items():
            by_signal[key][row["label"]].append(float(val))

    entries: dict[str, SignalCalibration] = {}
    table = {}
    n_auth = sum(1 for r in rows if r["label"] == "authentic")
    n_syn = sum(1 for r in rows if r["label"] == "synthetic")

    for key, groups in sorted(by_signal.items()):
        a, s = groups["authentic"], groups["synthetic"]
        cov = (len(a) + len(s)) / max(1, len(rows))
        signal_auc = auc(s, a)  # AUC of "higher value => synthetic"
        table[key] = {
            "n_authentic": len(a),
            "n_synthetic": len(s),
            "coverage": round(cov, 3),
            "auc": round(signal_auc, 3),
            "median_authentic": round(float(np.median(a)), 5) if a else None,
            "median_synthetic": round(float(np.median(s)), 5) if s else None,
        }
        if len(a) < MIN_PER_CLASS or len(s) < MIN_PER_CLASS:
            table[key]["fitted"] = False
            table[key]["why"] = "too few measurements in one class"
            continue

        # A tiny floor on sd keeps a degenerate (constant) signal from
        # producing an infinitely confident likelihood ratio.
        sd_a = max(float(np.std(a)), 1e-4 * (abs(float(np.mean(a))) + 1.0))
        sd_s = max(float(np.std(s)), 1e-4 * (abs(float(np.mean(s))) + 1.0))
        entries[key] = SignalCalibration(
            key=key,
            mu_authentic=float(np.mean(a)),
            sd_authentic=sd_a,
            mu_synthetic=float(np.mean(s)),
            sd_synthetic=sd_s,
            auc=signal_auc,
            n_authentic=len(a),
            n_synthetic=len(s),
        )
        table[key]["fitted"] = True
        table[key]["weight"] = round(entries[key].weight, 3)

    prior = 0.0
    if n_auth and n_syn:
        # Deliberately NOT the dataset base rate: this folder is 3:1, which is
        # an artefact of collection, not of the world. A neutral prior keeps
        # the score driven by evidence.
        prior = 0.0

    cal = Calibration(entries=entries, prior_log_odds=prior, meta=meta)
    return cal, table


def loo_scores(rows: list[dict], meta: dict) -> tuple[list[float], list[str]]:
    """Leave-one-out fused score for every row."""
    scores, labels = [], []
    for i in range(len(rows)):
        train = rows[:i] + rows[i + 1 :]
        cal, _ = fit_calibration(train, meta)
        held = rows[i]
        total = cal.prior_log_odds
        for key, val in held["values"].items():
            entry = cal.entries.get(key)
            if entry is None or entry.weight == 0.0:
                continue
            total += entry.llr(float(val)) * entry.weight
        scores.append(sigmoid(total))
        labels.append(held["label"])
    return scores, labels


def summarise(scores: list[float], labels: list[str]) -> dict:
    pos = [s for s, l in zip(scores, labels) if l == "synthetic"]
    neg = [s for s, l in zip(scores, labels) if l == "authentic"]
    a = auc(pos, neg)

    best = {"threshold": 0.5, "accuracy": 0.0}
    for t in [i / 100.0 for i in range(1, 100)]:
        tp = sum(1 for s in pos if s >= t)
        tn = sum(1 for s in neg if s < t)
        acc = (tp + tn) / max(1, len(pos) + len(neg))
        if acc > best["accuracy"]:
            best = {
                "threshold": round(t, 2),
                "accuracy": round(acc, 4),
                "sensitivity": round(tp / max(1, len(pos)), 4),
                "specificity": round(tn / max(1, len(neg)), 4),
            }
    return {
        "n_authentic": len(neg),
        "n_synthetic": len(pos),
        "auc": round(a, 4),
        "mean_score_authentic": round(float(np.mean(neg)), 4) if neg else None,
        "mean_score_synthetic": round(float(np.mean(pos)), 4) if pos else None,
        "best_operating_point": best,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--view", required=True, help="dataset view directory")
    ap.add_argument("--name", default="", help="label for this view in the report")
    ap.add_argument("--write-calibration", default="", help="path to write calibration.json")
    ap.add_argument("--report", default="", help="path to write the JSON report")
    args = ap.parse_args()

    name = args.name or os.path.basename(args.view.rstrip("/"))
    rows = extract(args.view)
    meta = {"view": name, "n_images": len(rows), "source": args.view}

    cal, table = fit_calibration(rows, meta)
    scores, labels = loo_scores(rows, meta)
    fused = summarise(scores, labels)

    print(f"\n=== view: {name}  ({len(rows)} images) ===")
    print(f"{'signal':32s} {'cov':>5s} {'nA':>4s} {'nS':>4s} {'AUC':>6s} {'wt':>5s}  medians A -> S")
    for key, t in sorted(table.items(), key=lambda kv: -abs(kv[1]["auc"] - 0.5)):
        wt = t.get("weight", 0.0) if t.get("fitted") else 0.0
        print(
            f"{key:32s} {t['coverage']:5.2f} {t['n_authentic']:4d} "
            f"{t['n_synthetic']:4d} {t['auc']:6.3f} {wt:5.2f}  "
            f"{t['median_authentic']} -> {t['median_synthetic']}"
        )

    print(f"\nfused, leave-one-out cross-validated:")
    print(json.dumps(fused, indent=2))

    if args.write_calibration:
        cal.meta.update({"fused_loo": fused, "per_signal": table})
        cal.dump(args.write_calibration)
        print(f"\nwrote calibration -> {args.write_calibration}")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(
                {"view": name, "per_signal": table, "fused_loo": fused},
                fh,
                indent=2,
            )
        print(f"wrote report      -> {args.report}")


if __name__ == "__main__":
    main()
