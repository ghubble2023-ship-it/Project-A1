"""Turns exported chat logs into the message list the analyser expects.

Phone and desktop exports all say the same thing in different punctuation, so
this normalises the common shapes rather than demanding one:

* **JSON** -- either a bare list of messages, or ``{"messages": [...]}``. Also
  understands Telegram's desktop export, where a message's ``text`` can be a
  list of runs rather than a string.
* **WhatsApp text export** -- ``[12/03/24, 19:04:11] Alice: hi`` and the
  unbracketed ``12/03/24, 19:04 - Alice: hi`` variant. Continuation lines
  belong to the previous message; system lines with no sender are dropped.
* **Generic ``Sender: text``** -- one message per line.
* **CSV** -- any file with sender-ish and text-ish column headers.

Everything comes out as ``[{"sender": str, "text": str, "ts": str|None}]``.
When a file has exactly two participants the more prolific one is *not*
assumed to be the suspect; that call belongs to the caller, because guessing
it wrong silently analyses the victim instead.
"""

from __future__ import annotations

import csv
import json
import os
import re

#: [12/03/2024, 19:04:11] Alice: text     (iOS / some Android exports)
WA_BRACKET = re.compile(
    r"^\[(?P<ts>[^\]]{6,40})\]\s*(?P<sender>[^:]{1,64}?):\s?(?P<text>.*)$"
)
#: 12/03/2024, 19:04 - Alice: text        (classic Android export)
WA_DASH = re.compile(
    r"^(?P<ts>\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4},?\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp][Mm])?)"
    r"\s*-\s*(?P<sender>[^:]{1,64}?):\s?(?P<text>.*)$"
)
#: Alice: text
PLAIN = re.compile(r"^(?P<sender>[A-Za-z0-9 _.\-'()+]{1,40}):\s(?P<text>.+)$")

SENDER_KEYS = ("sender", "from", "author", "name", "speaker", "user", "who")
TEXT_KEYS = ("text", "message", "body", "content", "msg")
TS_KEYS = ("ts", "timestamp", "time", "date", "datetime")


def _first_key(row: dict, candidates: tuple[str, ...]) -> str | None:
    lowered = {k.lower().strip(): k for k in row if isinstance(k, str)}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    return None


def _flatten_telegram_text(value) -> str:
    """Telegram splits a message into styled runs; join them back up."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for run in value:
            if isinstance(run, str):
                parts.append(run)
            elif isinstance(run, dict):
                parts.append(str(run.get("text", "")))
        return "".join(parts)
    return "" if value is None else str(value)


def from_json(payload) -> list[dict]:
    if isinstance(payload, dict):
        payload = payload.get("messages", payload.get("chats", []))
    out = []
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        sk = _first_key(item, SENDER_KEYS)
        tk = _first_key(item, TEXT_KEYS)
        if tk is None:
            continue
        text = _flatten_telegram_text(item[tk]).strip()
        if not text:
            continue
        tsk = _first_key(item, TS_KEYS)
        out.append(
            {
                "sender": str(item.get(sk, "unknown")) if sk else "unknown",
                "text": text,
                "ts": str(item[tsk]) if tsk else None,
            }
        )
    return out


def from_text(raw: str) -> list[dict]:
    messages: list[dict] = []
    for line in raw.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        m = WA_BRACKET.match(line) or WA_DASH.match(line)
        if m:
            messages.append(
                {
                    "sender": m.group("sender").strip(),
                    "text": m.group("text").strip(),
                    "ts": m.group("ts").strip(),
                }
            )
            continue
        m = PLAIN.match(line)
        if m and messages is not None:
            messages.append(
                {"sender": m.group("sender").strip(), "text": m.group("text").strip(), "ts": None}
            )
            continue
        if messages:
            # Wrapped continuation of the previous message.
            messages[-1]["text"] = (messages[-1]["text"] + " " + line.strip()).strip()
    return [m for m in messages if m["text"]]


def from_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return from_json(rows)


def load_transcript(path: str) -> list[dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return from_csv(path)
    raw = open(path, "r", encoding="utf-8", errors="replace").read()
    if ext == ".json" or raw.lstrip()[:1] in "[{":
        try:
            return from_json(json.loads(raw))
        except json.JSONDecodeError:
            pass
    return from_text(raw)


def participants(messages: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in messages:
        counts[m["sender"]] = counts.get(m["sender"], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect how a transcript parses.")
    ap.add_argument("path")
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    msgs = load_transcript(args.path)
    print(f"{len(msgs)} messages; participants: {participants(msgs)}")
    for m in msgs[: args.show]:
        print(f"  [{m['ts'] or '-'}] {m['sender']}: {m['text'][:90]}")
