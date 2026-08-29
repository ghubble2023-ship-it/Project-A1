"""Command-line entry point: ``python -m sentinel``."""

from __future__ import annotations

import argparse
import json
import sys

from .fusion import Calibration
from .image.pipeline import analyse_image
from .text.pipeline import analyse_conversation


def _print_image_human(result: dict) -> None:
    v = result["verdict"]
    e = result["explanation"]
    print(f"\n  verdict     {v['label']}")
    print(f"  p(synthetic) {v['p_synthetic']:.3f}    evidence coverage {v['coverage']:.0%}")
    print(f"\n  {e['summary']}\n")

    for title, items in (
        ("Points to synthetic", e["evidence_for_synthetic"]),
        ("Points to authentic", e["evidence_for_authentic"]),
    ):
        if not items:
            continue
        print(f"  {title}:")
        for it in items:
            print(f"    - {it['headline']}")
            for line in _wrap(it["finding"], 74):
                print(f"        {line}")
        print()

    if e["not_measured"]:
        print("  Not measured:")
        for nm in e["not_measured"]:
            print(f"    - {nm['label']}: {nm['reason']}")
        print()
    print("  Caveats:")
    for c in e["caveats"]:
        for line in _wrap(c, 74):
            print(f"    {line}")
    print()


def _print_text_human(result: dict) -> None:
    print(f"\n  risk        {result['risk_band']}  ({result['risk_score']:.2f})")
    print(f"  playbooks   {', '.join(result['identified_playbooks'])}")
    print(f"  components  {result['scoring']['components']}")
    print(f"  method      {result['scoring']['method']}")
    print(f"\n  {result['escalation']['note']}\n")

    if result["matched_phrases"]:
        print("  Matched phrases:")
        for m in result["matched_phrases"]:
            print(f"    - [msg {m['message_index']}] {m['finding']}: {m['quote']!r}")
        print()
    if result["identifiers"]:
        print("  Identifiers seen:")
        for k, vals in result["identifiers"].items():
            print(f"    - {k}: {', '.join(vals)}")
        print()
    print("  What to do:")
    for a in result["what_to_do"]:
        for line in _wrap(a, 74):
            print(f"    {line}")
    print()


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="sentinel",
        description="Physics-based media forensics and scam-conversation analysis.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    img = sub.add_parser("image", help="analyse an image file")
    img.add_argument("path")
    img.add_argument("--calibration", default=None)
    img.add_argument("--json", action="store_true", help="emit raw JSON")

    txt = sub.add_parser("chat", help="analyse a conversation JSON file")
    txt.add_argument(
        "path",
        help='JSON: [{"sender": "suspect", "text": "..."}, ...] or {"messages": [...]}',
    )
    txt.add_argument("--suspect", default="suspect", help="sender id to analyse")
    txt.add_argument("--all-senders", action="store_true", help="ignore sender filter")
    txt.add_argument("--json", action="store_true", help="emit raw JSON")

    args = ap.parse_args(argv)

    if args.command == "image":
        with open(args.path, "rb") as fh:
            data = fh.read()
        cal = Calibration.load(args.calibration)
        if not cal.entries:
            print(
                "warning: no calibration loaded; every signal will abstain.\n"
                "         run eval/run_eval.py --write-calibration first.",
                file=sys.stderr,
            )
        result = analyse_image(data, cal)
        print(json.dumps(result, indent=2)) if args.json else _print_image_human(result)
        return 0

    with open(args.path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    messages = payload["messages"] if isinstance(payload, dict) else payload
    suspect = None if args.all_senders else args.suspect
    result = analyse_conversation(messages, suspect)
    print(json.dumps(result, indent=2)) if args.json else _print_text_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
