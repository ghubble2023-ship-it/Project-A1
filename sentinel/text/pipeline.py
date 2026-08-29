"""End-to-end conversation analysis: score, explain, and say what to do.

Unlike the image path, this scorer is **rule-based and uncalibrated**: no
labelled corpus of scam transcripts ships with the project, so there is no
honest way to fit likelihood ratios the way ``sentinel.fusion`` does for
images. The weights here come from published fraud typologies, and the report
says so rather than dressing a rubric up as a measurement.

What it does not skimp on is the last section. A risk score alone helps
nobody. Someone deep in a romance scam has usually already been told by
friends that it is a scam; what changes the outcome is a concrete next step
they can take right now, tied to what was actually found.
"""

from __future__ import annotations

from typing import Any

from ..types import Signal
from . import classifier, stylometry

#: Weights over the component scores. Escalation velocity is weighted heavily
#: because timing is much harder for an operator to disguise than vocabulary.
WEIGHTS = {
    "playbook": 0.55,
    "velocity": 0.25,
    "breadth": 0.10,
    "style": 0.10,
}

ADVICE: dict[str, list[str]] = {
    "pig_butchering": [
        "Do not deposit funds, however small. The 'test withdrawal that works' "
        "is part of the script; the money is released precisely so that a "
        "larger deposit follows.",
        "A platform you were introduced to by someone you met online is not an "
        "independent platform. Check it against your national regulator's "
        "warning list before anything else.",
    ],
    "task_scam": [
        "Legitimate work never requires you to pay in to get paid out. A "
        "'negative balance' you must clear is the entire mechanism of the scam.",
        "Stop before the next deposit. The amount already in the account is the "
        "hook, not an asset you can recover by adding to it.",
    ],
    "account_takeover": [
        "Never read out a verification code. No real bank, platform, or police "
        "force will ever ask for one -- the code exists specifically to stop "
        "the thing the caller is trying to do.",
        "If remote-access software was installed, disconnect from the internet, "
        "uninstall it, and change passwords from a different device.",
        "Call your bank back on the number printed on your card, not any number "
        "given to you in this conversation.",
    ],
    "recovery_fraud": [
        "Recovery services that charge an upfront fee are re-targeting people "
        "who already lost money. Funds are never recovered this way.",
        "Report to your national fraud reporting service instead; recovery "
        "through them is free.",
    ],
    "payment_pressure": [
        "Gift cards, wire transfers, and crypto are chosen because they cannot "
        "be reversed. A request for one of these is a request you cannot undo.",
    ],
    "coercion": [
        "Being asked to keep a financial matter secret from family or your bank "
        "is a control tactic, not discretion. Tell one person you trust today.",
    ],
    "off_platform": [
        "Moving to Telegram or WhatsApp removes the reporting and moderation "
        "you currently have. Staying on the original platform costs you nothing.",
    ],
    "identity_evasion": [
        "Ask for a live video call at a time you choose. Consistent excuses "
        "about cameras or deployment are the single most reliable tell.",
    ],
}

GENERAL_ADVICE = [
    "Nothing here is proof about any individual. Treat it as a prompt to slow "
    "down and verify, not as a verdict about a person.",
]


def band(score: float) -> str:
    if score >= 0.75:
        return "CRITICAL"
    if score >= 0.5:
        return "HIGH"
    if score >= 0.28:
        return "ELEVATED"
    if score >= 0.12:
        return "LOW"
    return "MINIMAL"


def analyse_conversation(
    messages: list[dict[str, str]], suspect: str | None = "suspect"
) -> dict[str, Any]:
    if not messages:
        raise ValueError("No messages supplied.")

    scan = classifier.scan(messages, suspect)
    hits = scan["hits"]
    suspect_idx = scan["suspect_message_indices"]

    if not suspect_idx:
        raise ValueError(
            f"No messages from sender {suspect!r}. Pass suspect=None to scan "
            f"every message."
        )

    suspect_texts = [messages[i].get("text", "") or "" for i in suspect_idx]

    playbooks = classifier.score_playbooks(hits)
    velocity = classifier.escalation_velocity(hits, len(suspect_idx))
    style_signals = stylometry.analyse(suspect_texts)
    signals: list[Signal] = classifier.to_signals(playbooks, velocity) + style_signals

    top_playbook = max((v["score"] for v in playbooks.values()), default=0.0)
    breadth = sum(1 for v in playbooks.values() if v["score"] >= 0.25)
    breadth_norm = min(1.0, breadth / 3.0)

    style_score = 0.0
    style_parts = 0
    uniformity = next((s for s in style_signals if s.key == "style_length_uniformity"), None)
    repetition = next((s for s in style_signals if s.key == "style_repetition"), None)
    if uniformity and uniformity.available and uniformity.value is not None:
        # Below ~0.35 coefficient of variation reads as pasted text.
        style_score += max(0.0, min(1.0, (0.35 - uniformity.value) / 0.35))
        style_parts += 1
    if repetition and repetition.available and repetition.value is not None:
        style_score += min(1.0, repetition.value * 3.0)
        style_parts += 1
    style_score = style_score / style_parts if style_parts else 0.0

    risk = (
        WEIGHTS["playbook"] * top_playbook
        + WEIGHTS["velocity"] * velocity["score"]
        + WEIGHTS["breadth"] * breadth_norm
        + WEIGHTS["style"] * style_score
    )
    risk = round(min(1.0, risk), 4)

    active = sorted(
        (p for p, v in playbooks.items() if v["score"] >= 0.25),
        key=lambda p: -playbooks[p]["score"],
    )

    advice: list[str] = []
    for p in active:
        for line in ADVICE.get(p, []):
            if line not in advice:
                advice.append(line)
    advice.extend(GENERAL_ADVICE)

    return {
        "risk_score": risk,
        "risk_band": band(risk),
        "scoring": {
            "method": "rule-based rubric (not statistically calibrated)",
            "weights": WEIGHTS,
            "components": {
                "playbook": round(top_playbook, 4),
                "velocity": round(velocity["score"], 4),
                "breadth": round(breadth_norm, 4),
                "style": round(style_score, 4),
            },
        },
        "identified_playbooks": active or ["none_matched"],
        "playbook_detail": playbooks,
        "escalation": velocity,
        "matched_phrases": [
            {
                "message_index": h["message_index"],
                "playbook": h["playbook"],
                "finding": h["label"],
                "quote": h["quote"],
            }
            for h in hits
        ],
        "identifiers": classifier.extract_identifiers(suspect_texts),
        "signals": [s.to_dict() for s in signals],
        "what_to_do": advice,
        "messages_analysed": len(suspect_idx),
    }
