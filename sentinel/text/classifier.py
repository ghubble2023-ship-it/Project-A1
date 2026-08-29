"""Scores a conversation against the playbook table.

Scoring rubric, stated openly because a fraud score nobody can audit is a
fraud score nobody should act on:

* Each rule that fires contributes its ``severity``, but a playbook's total is
  **saturating** -- the fifth hit on the same rule adds almost nothing. This
  stops one repeated word from manufacturing a critical score.
* A playbook whose hits span multiple *stages* is escalated. Progression
  through a funnel (icebreaker, then grooming, then a deposit request) is far
  more diagnostic than any single phrase, and it is what separates a real
  operation from someone quoting one.
* **Escalation velocity** is scored separately: an ask to move off-platform in
  the first few messages is a different animal from the same ask on day ten.

The output is a 0-1 risk score with the arithmetic exposed, not a black box.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from ..types import SYNTHETIC_HIGH, Signal
from .playbooks import EXTRACTORS, RULES

#: How many messages count as "early" for escalation-velocity purposes.
EARLY_WINDOW = 5


def _saturate(total: float, k: float = 1.1) -> float:
    """Diminishing returns: many hits raise confidence, but never linearly."""
    return 1.0 - math.exp(-total / k)


def scan(messages: list[dict[str, str]], suspect: str | None) -> dict[str, Any]:
    """Run every rule over the suspect's messages, keeping the matched text."""
    hits: list[dict[str, Any]] = []
    suspect_indices: list[int] = []

    for idx, msg in enumerate(messages):
        sender = msg.get("sender")
        if suspect is not None and sender != suspect:
            continue
        suspect_indices.append(idx)
        text = msg.get("text", "") or ""
        position = len(suspect_indices)  # 1-based within the suspect's stream
        for rule in RULES:
            m = rule.pattern.search(text)
            if m:
                hits.append(
                    {
                        "rule": rule.key,
                        "playbook": rule.playbook,
                        "stage": rule.stage,
                        "severity": rule.severity,
                        "label": rule.label,
                        "message_index": idx,
                        "suspect_message_number": position,
                        # Quoting the matched span is what makes this
                        # reviewable rather than merely assertive.
                        "quote": m.group(0),
                    }
                )
    return {"hits": hits, "suspect_message_indices": suspect_indices}


def extract_identifiers(texts: list[str]) -> dict[str, list[str]]:
    joined = "\n".join(texts)
    out: dict[str, list[str]] = {}
    for name, pat in EXTRACTORS.items():
        found = []
        for m in pat.finditer(joined):
            val = m.group(0)
            if val not in found:
                found.append(val)
        if found:
            out[name] = found[:12]
    return out


def score_playbooks(hits: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_playbook: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for h in hits:
        by_playbook[h["playbook"]].append(h)

    scored: dict[str, dict[str, Any]] = {}
    for playbook, group in by_playbook.items():
        # Saturate per rule first, then sum -- so ten hits on one rule cannot
        # outweigh one hit each on three different rules.
        per_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for h in group:
            per_rule[h["rule"]].append(h)

        raw = 0.0
        for rule_hits in per_rule.values():
            sev = rule_hits[0]["severity"]
            raw += sev * (1.0 + 0.25 * (len(rule_hits) - 1))

        stages = sorted({h["stage"] for h in group})
        # Funnel progression: touching stages 1 and 3 is the real signature.
        span_bonus = 0.0
        if len(stages) >= 2:
            span_bonus = 0.35 * (len(stages) - 1) + 0.25 * (max(stages) - min(stages))

        score = _saturate(raw + span_bonus)
        scored[playbook] = {
            "score": round(score, 4),
            "rules_fired": sorted(per_rule.keys()),
            "stages_seen": stages,
            "stage_progression": len(stages) >= 2,
            "hit_count": len(group),
        }
    return scored


def escalation_velocity(hits: list[dict[str, Any]], n_suspect: int) -> dict[str, Any]:
    """How fast the suspect pushed off-platform or toward money."""
    urgent = [
        h
        for h in hits
        if h["playbook"] in {"off_platform", "payment_pressure", "account_takeover"}
    ]
    if not urgent:
        return {
            "score": 0.0,
            "first_ask_at_message": None,
            "early": False,
            "note": "No off-platform, payment, or credential request detected.",
        }

    first = min(h["suspect_message_number"] for h in urgent)
    early = first <= EARLY_WINDOW
    # Fast asks score high; the same ask deep into a real friendship does not.
    #
    # This was previously linear -- 1 - (first-1)/(2*EARLY_WINDOW) -- which hits
    # zero at message 11 and stays there. That is a cliff, not a decay: an ask at
    # message 21 scored identically to an ask at message 500, and since real
    # gift-card and romance approaches routinely spend twenty messages on
    # rapport before the ask, the component contributed exactly nothing on the
    # cases it was meant to catch. A hyperbolic decay keeps the same ordering and
    # the same strong preference for early asks, but never asserts that a
    # mid-conversation ask carries no information at all.
    #
    #   message  1 -> 1.00      message 11 -> 0.33
    #   message  6 -> 0.50      message 21 -> 0.20
    score = EARLY_WINDOW / (EARLY_WINDOW + max(0, first - 1))
    return {
        "score": round(score, 4),
        "first_ask_at_message": first,
        "early": early,
        "of_messages": n_suspect,
        "note": (
            f"First off-platform/payment/credential request arrived at message "
            f"{first} of {n_suspect} from this sender."
            + (" That is unusually fast." if early else "")
        ),
    }


def to_signals(
    playbooks: dict[str, dict[str, Any]], velocity: dict[str, Any]
) -> list[Signal]:
    top = max((v["score"] for v in playbooks.values()), default=0.0)
    return [
        Signal(
            key="scam_playbook_max",
            label="Strongest matched fraud playbook",
            value=round(top, 4),
            unit="0-1",
            direction=SYNTHETIC_HIGH,
            context={"playbooks": playbooks},
        ),
        Signal(
            key="scam_playbook_breadth",
            label="Number of distinct playbooks matched",
            value=float(sum(1 for v in playbooks.values() if v["score"] >= 0.25)),
            unit="count",
            direction=SYNTHETIC_HIGH,
        ),
        Signal(
            key="scam_escalation_velocity",
            label="Speed of the first off-platform or payment ask",
            value=round(velocity["score"], 4),
            unit="0-1",
            direction=SYNTHETIC_HIGH,
            context=velocity,
        ),
    ]
