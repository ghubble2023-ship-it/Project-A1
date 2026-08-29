"""Is the person on the other end typing, or is a script?

Three things separate a human correspondent from a boiler-room script:

* **Length rhythm.** People write bursts of wildly different lengths. Scripted
  senders emit paragraphs of near-identical length because they are pasting.
* **Verbatim repetition.** Humans rephrase. Scripts resend. Near-duplicate
  message pairs are the strongest single tell here.
* **Register mismatch.** Marketing-grade prose ("unparalleled opportunity",
  "esteemed client") inside a casual chat means the text was written for
  someone else first.

The uniformity test needs several messages before it means anything, so it
abstains below that -- one long message is not evidence of automation.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ..types import SYNTHETIC_HIGH, SYNTHETIC_LOW, Signal

MIN_MESSAGES = 4

FORMAL_REGISTER = re.compile(
    r"\b(?:esteemed|kindly (?:be )?(?:advised|note)|unparalleled|"
    r"lucrative opportunity|dear (?:sir|madam|valued|esteemed)|"
    r"i am reaching out|at your earliest convenience|"
    r"rest assured|utmost (?:importance|priority))",
    re.IGNORECASE,
)


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def analyse(texts: list[str]) -> list[Signal]:
    clean = [t for t in texts if t and t.strip()]
    if len(clean) < MIN_MESSAGES:
        reason = f"needs {MIN_MESSAGES} messages from this sender, got {len(clean)}"
        return [
            Signal.missing("style_length_uniformity", "Message-length uniformity", reason),
            Signal.missing("style_repetition", "Near-duplicate message rate", reason),
            Signal.missing("style_formal_register", "Marketing register in casual chat", reason),
        ]

    lengths = [len(t.split()) for t in clean]
    mean_len = sum(lengths) / len(lengths)
    sd = math.sqrt(sum((l - mean_len) ** 2 for l in lengths) / len(lengths))
    # Coefficient of variation: scale-free, so a chatty human and a terse one
    # are judged on rhythm rather than verbosity.
    cv = sd / mean_len if mean_len > 0 else 0.0

    norm = [_normalise(t) for t in clean]
    pairs = 0
    dupes = 0
    for i in range(len(norm)):
        for j in range(i + 1, len(norm)):
            pairs += 1
            if _jaccard(norm[i], norm[j]) >= 0.8:
                dupes += 1
    repetition = dupes / pairs if pairs else 0.0

    formal_hits = sum(1 for t in clean if FORMAL_REGISTER.search(t))
    formal_rate = formal_hits / len(clean)

    return [
        Signal(
            key="style_length_uniformity",
            label="Message-length uniformity",
            # Low variation => scripted, so low is the suspicious direction.
            value=round(cv, 4),
            unit="coefficient of variation",
            direction=SYNTHETIC_LOW,
            context={"mean_words": round(mean_len, 1), "messages": len(clean)},
        ),
        Signal(
            key="style_repetition",
            label="Near-duplicate message rate",
            value=round(repetition, 4),
            unit="fraction of message pairs",
            direction=SYNTHETIC_HIGH,
            context={"duplicate_pairs": dupes, "pairs": pairs},
        ),
        Signal(
            key="style_formal_register",
            label="Marketing register in casual chat",
            value=round(formal_rate, 4),
            unit="fraction of messages",
            direction=SYNTHETIC_HIGH,
            context={"messages_hit": formal_hits},
        ),
    ]
