"""Convert chat export formats into the corpus structure.

Handles common phone export formats (WhatsApp, SMS, etc) and plain text.
Input: one transcript file per chat
Output: structured JSON for corpus ingestion
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def parse_whatsapp_export(text: str) -> list[dict]:
    """Parse WhatsApp chat export format.

    Handles formats like:
    [13/07, 05:45] sender: message
    or
    7/13/20, 05:45 - sender: message
    """
    messages = []
    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Pattern: [date, time] sender: text or date, time - sender: text
        match = re.match(r'[\[\[]?[\d/\-]+,?\s+[\d:]+[\]\]]?\s*[-]?\s*(.+?):\s*(.+)', line)
        if match:
            sender, text = match.groups()
            messages.append({
                "sender": sender.strip(),
                "text": text.strip()
            })
        # Fallback: just split on colon
        elif ":" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                messages.append({
                    "sender": parts[0].strip(),
                    "text": parts[1].strip()
                })

    return messages


def parse_sms_export(text: str) -> list[dict]:
    """Parse simple SMS/text export (sender: message per line)."""
    messages = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        sender, text = line.split(":", 1)
        messages.append({
            "sender": sender.strip(),
            "text": text.strip()
        })
    return messages


def load_transcript(path: str | Path) -> list[dict]:
    """Load transcript from file, auto-detecting format."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    # Try WhatsApp first (has timestamps)
    if re.search(r'\d{1,2}[/\-]\d{1,2}', text):
        return parse_whatsapp_export(text)

    # Fallback to simple sender: message format
    return parse_sms_export(text)


def main():
    """Example: convert a transcript file."""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python parse_transcripts.py <input.txt> [output.json]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else str(input_path).replace(".txt", ".json")

    messages = load_transcript(input_path)
    Path(output_path).write_text(json.dumps(messages, indent=2), encoding="utf-8")
    print(f"Converted {len(messages)} messages to {output_path}")


if __name__ == "__main__":
    main()
