"""Fits and validates a calibration for the conversation signals.

The conversation path shipped as an openly-declared rubric because no labelled
transcript corpus existed. Given one, the same machinery the image path uses
applies unchanged: extract signals, measure each one's standalone AUC, fit
per-hypothesis Gaussians, and cross-validate the fused score leave-one-out.

Input layout -- one directory per label, each containing conversation files::

    corpus/
      scam/     one .json or .txt transcript per conversation
      benign/

JSON transcripts are ``[{"sender": ..., "text": ...}, ...]`` or
``{"messages": [...]}``. Plain-text transcripts are parsed by
``eval.ingest_chats``, which handles the common phone-export formats.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest_chats import load_transcript  # noqa: E402
from run_eval import auc, fit_calibration, summarise  # noqa: E402
from sentinel.text import classifier, stylometry  # noqa: E402
from sentinel.types import sigmoid  # noqa: E402

#: Directory names mapped onto the two hypotheses the fusion layer expects.
LABEL_MAP = {
    "scam": "synthetic",
    "fraud": "synthetic",
    "fake": "synthetic",
    "benign": "authentic",
    "normal": "authentic",
    "real": "authentic",
    "ham": "authentic",
}


def signals_for(messages: list[dict], suspect: str | None) -> dict[str, float]:
    scan = classifier.scan(messages, suspect)
    idx = scan["suspect_message_indices"]
    if not idx:
        return {}
    texts = [messages[i].get("text", "") or "" for i in idx]
    playbooks = classifier.score_playbooks(scan["hits"])
    velocity = classifier.escalation_velocity(scan["hits"], len(idx))
    sigs = classifier.to_signals(playbooks, velocity) + stylometry.analyse(texts)
    return {s.key: float(s.value) for s in sigs if s.available and s.value is not None}


def extract_corpus(root: str, suspect: str | None) -> list[dict]:
    rows = []
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub):
            continue
        label = LABEL_MAP.get(name.lower())
        if label is None:
            print(f"  skipping unlabelled directory {name!r}", file=sys.stderr)
            continue
        for fn in sorted(os.listdir(sub)):
            path = os.path.join(sub, fn)
            if not os.path.isfile(path):
                continue
            try:
                messages = load_transcript(path)
            except Exception as exc:
                print(f"  skipping {fn}: {exc}", file=sys.stderr)
                continue
            if not messages:
                continue
            values = signals_for(messages, suspect)
            if not values:
                continue
            rows.append(
                {"path": path, "label": label, "values": values, "n_messages": len(messages)}
            )
    return rows


def loo(rows: list[dict], meta: dict) -> tuple[list[float], list[str]]:
    scores, labels = [], []
    for i in range(len(rows)):
        cal, _ = fit_calibration(rows[:i] + rows[i + 1 :], meta)
        total = cal.prior_log_odds
        for key, val in rows[i]["values"].items():
            entry = cal.entries.get(key)
            if entry is None or entry.weight == 0.0:
                continue
            total += entry.llr(float(val)) * entry.weight
        scores.append(sigmoid(total))
        labels.append(rows[i]["label"])
    return scores, labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, help="root with per-label subdirectories")
    ap.add_argument(
        "--suspect",
        default=None,
        help="sender id to analyse; omit to analyse every message",
    )
    ap.add_argument("--write-calibration", default="")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    rows = extract_corpus(args.corpus, args.suspect)
    if not rows:
        print("No usable transcripts found.", file=sys.stderr)
        raise SystemExit(1)

    meta = {"view": "conversations", "n": len(rows), "source": args.corpus}
    cal, table = fit_calibration(rows, meta)
    scores, labels = loo(rows, meta)
    fused = summarise(scores, labels)

    n_scam = sum(1 for r in rows if r["label"] == "synthetic")
    print(f"\n=== conversation calibration ({len(rows)} transcripts, {n_scam} scam) ===")
    print(f"{'signal':32s} {'cov':>5s} {'nB':>4s} {'nS':>4s} {'AUC':>6s} {'wt':>5s}  medians benign -> scam")
    for key, t in sorted(table.items(), key=lambda kv: -abs(kv[1]["auc"] - 0.5)):
        wt = t.get("weight", 0.0) if t.get("fitted") else 0.0
        print(
            f"{key:32s} {t['coverage']:5.2f} {t['n_authentic']:4d} {t['n_synthetic']:4d} "
            f"{t['auc']:6.3f} {wt:5.2f}  {t['median_authentic']} -> {t['median_synthetic']}"
        )

    print("\nfused, leave-one-out cross-validated:")
    print(json.dumps(fused, indent=2))

    if args.write_calibration:
        cal.meta.update({"fused_loo": fused, "per_signal": table})
        cal.dump(args.write_calibration)
        print(f"\nwrote calibration -> {args.write_calibration}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump({"per_signal": table, "fused_loo": fused}, fh, indent=2)
        print(f"wrote report      -> {args.report}")


if __name__ == "__main__":
    main()
