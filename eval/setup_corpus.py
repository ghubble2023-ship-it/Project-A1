"""Set up corpus directory structure and validate transcripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from parse_transcripts import load_transcript


def setup_corpus(root: Path) -> None:
    """Create subdirectories for scam/benign classifications."""
    for label in ("scam", "benign"):
        (root / label).mkdir(parents=True, exist_ok=True)
    print(f"Corpus structure created at {root}")
    print("  scam/     — romance fraud, investment scams, etc.")
    print("  benign/   — ordinary conversations")


def validate_corpus(root: Path) -> None:
    """Check corpus and report coverage."""
    root = Path(root)
    if not root.exists():
        print(f"Corpus not found: {root}")
        return

    for label in ("scam", "benign"):
        subdir = root / label
        if not subdir.exists():
            continue
        files = list(subdir.glob("*.json")) + list(subdir.glob("*.txt"))
        if not files:
            print(f"  {label:8s}: empty")
            continue

        total_messages = 0
        for f in files:
            try:
                messages = load_transcript(f)
                total_messages += len(messages)
            except Exception as e:
                print(f"    ⚠ {f.name}: {e}")

        print(f"  {label:8s}: {len(files):3d} files, ~{total_messages:4d} messages")


def main():
    root = Path("corpus")
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])

    if not root.exists():
        setup_corpus(root)
    print("\nCurrent corpus state:")
    validate_corpus(root)

    if (root / "scam").exists() and len(list((root / "scam").glob("*.json"))) == 0:
        print("\n→ Next: add .json or .txt transcript files to corpus/scam/ and corpus/benign/")
        print("  Then run: python eval/run_text_eval.py --corpus corpus/")


if __name__ == "__main__":
    main()
