"""
Opener move-sequence matcher.

THE PROBLEM THIS SOLVES: the text engine catches late-stage scam
language (money, urgency, off-platform) but scored ZERO on the actual
opener that started a confirmed scam - "Hello my name is Bella. How
are you doing today?" That's by design on the scammer's side: openers
are engineered to look harmless.

THE INSIGHT (Greg's, confirmed against 9 real openers): the WORDING
varies, the MOVES don't. Exact-string clustering only grouped 4 of 9,
but semantic-move analysis showed greeting + "how are you doing"
appearing in 6/9 and a location question in 3/9 across entirely
unrelated accounts.

So this matches on move-sequence, not phrasing.

IMPORTANT - THIS IS NOT A SCAM VERDICT ON ITS OWN. Millions of
legitimate first messages are "Hi, how are you?" This produces a
*pattern score* that is only meaningful combined with context the
caller supplies (unknown sender, profile tells, unsolicited contact).
See score_opener()'s context handling. Anything else would flag every
polite stranger on earth.

Privacy: pure local regex/string work. No network, no third-party
services, nothing stored by this module.
"""

import re

# Each move is a semantic act, not a fixed phrase. Patterns are
# deliberately loose on wording and tight on intent.
MOVES = {
    "greeting": re.compile(r"\b(hello+|hi+|hey+|hiya|good\s+(morning|afternoon|evening))\b", re.I),
    "how_are_you": re.compile(r"how\s*(?:'?re|\s+are|\s+is)?\s*you(?:\s+doing)?|how\s+was\s+your\s+day", re.I),
    "asks_location": re.compile(r"where\s+(?:are\s+)?you\s+(?:from|located|at|live)|what\s+(?:part|state|city|country)", re.I),
    "self_introduces": re.compile(r"\bmy\s+name\s+is\b|\bthis\s+is\s+\w+\b|\bi'?m\s+[A-Z]\w+", re.I),
    "endearment": re.compile(r"\b(darling|dear|honey|sweetie|sweetheart|babe|love|handsome|beautiful)\b", re.I),
    "thanks_for_follow": re.compile(r"thank(?:s| you)\s+(?:so much\s+)?for\s+(?:the\s+)?(?:follow|add|accept)", re.I),
    "uses_recipient_name": None,  # handled specially - needs the user's name
    "compliment": re.compile(r"\b(you'?re|your)\s+(so\s+)?(cute|pretty|handsome|gorgeous|beautiful|smoking|hot)\b|\bnice\s+(smile|eyes|photo|pic)\b", re.I),
    "asks_marital_status": re.compile(r"\b(are you|you)\s+(single|married|dating|in a relationship)\b", re.I),
    "asks_age": re.compile(r"how\s+old\s+(are\s+)?you|what'?s?\s+your\s+age", re.I),
    "emoji_heavy": re.compile(r"(?:[\U0001F300-\U0001FAFF\u2600-\u27BF]\s*){2,}"),
}

# Move-pair sequences observed repeatedly in the real corpus across
# unrelated accounts. These are the actual templates - derived from
# data, not invented.
OBSERVED_SEQUENCES = [
    (("greeting", "how_are_you"), 0.35, "greeting immediately followed by 'how are you' - most common observed opener shape"),
    (("how_are_you", "asks_location"), 0.55, "wellbeing question paired with an immediate location question - strongest observed pattern"),
    (("greeting", "asks_location"), 0.45, "greeting straight into a location question"),
    (("self_introduces", "how_are_you"), 0.40, "name introduction plus wellbeing question - the confirmed SMS scam shape"),
    (("endearment", "how_are_you"), 0.45, "endearment toward a stranger plus wellbeing question"),
    (("greeting", "compliment"), 0.30, "greeting plus unsolicited compliment"),
]

# Individual moves that carry weight on their own in a FIRST message
STANDALONE_WEIGHTS = {
    "endearment": 0.40,          # calling a total stranger "darling"/"dear"
    "asks_location": 0.35,       # location fishing in message one
    "thanks_for_follow": 0.30,   # observed in the real corpus
    "asks_marital_status": 0.35,
    "compliment": 0.20,
    "asks_age": 0.20,
    "emoji_heavy": 0.10,
}

# EXCULPATORY moves - these REDUCE the score. Added after false-positive
# testing: legitimate strangers ("your order is ready", "about the truck
# you listed", "we met at the graduation") were scoring in the same
# MEDIUM band as real scam openers. The distinguishing feature is that a
# real stranger states a SPECIFIC, CHECKABLE REASON for making contact.
# Scam openers are pure social contact with no stated purpose - that
# absence is itself the signal.
EXCULPATORY = {
    "states_transaction_reason": (
        re.compile(r"\b(your\s+)?(order|delivery|package|invoice|payment|booking|reservation|appointment|prescription)\b", re.I),
        -0.55,
    ),
    "references_a_listing": (
        re.compile(r"\b(you\s+(listed|posted|advertised)|the\s+(ad|listing|posting)|for\s+sale|still\s+available)\b", re.I),
        -0.55,
    ),
    "names_mutual_context": (
        re.compile(r"\bwe\s+met\b|\bgave\s+me\s+your\s+(number|contact)\b|\bfrom\s+(work|church|the\s+shop|school)\b|\byour\s+(brother|sister|friend|cousin|coworker)\b", re.I),
        -0.50,
    ),
    "identifies_organization": (
        re.compile(r"\b(this\s+is|calling\s+from|reaching\s+out\s+(from|on\s+behalf))\b[^.]{0,40}\b(office|clinic|shop|company|store|bank|dealership|department|school|hospital)\b", re.I),
        -0.50,
    ),
    "asks_specific_actionable": (
        re.compile(r"\bare\s+we\s+still\s+on\b|\bwhat\s+time\b|\bconfirm(ing)?\b|\bpickup\b|\bdrop\s+off\b", re.I),
        -0.35,
    ),
}

# Context multipliers - these are what make the score meaningful.
# A generic "hi how are you" from a known contact is nothing.
CONTEXT_WEIGHTS = {
    "unknown_sender": 0.50,
    "platform_safety_warning": 0.60,   # the platform itself flagged it
    "unsolicited_first_contact": 0.35,
    "photo_attached_unprompted": 0.30,
    "no_prior_interaction": 0.25,
}


def detect_moves(text: str, recipient_name: str = None):
    """Returns the ordered list of moves present in the text."""
    if not text:
        return []

    found = []
    for name, pattern in MOVES.items():
        if pattern is None:
            continue
        m = pattern.search(text)
        if m:
            found.append((name, m.start()))

    if recipient_name:
        m = re.search(r"\b" + re.escape(recipient_name) + r"\b", text, re.I)
        if m:
            found.append(("uses_recipient_name", m.start()))

    found.sort(key=lambda t: t[1])
    return [name for name, _ in found]


def score_opener(text: str, context: dict = None, recipient_name: str = None):
    """
    Returns a dict with the moves found, the sequences matched, a
    pattern score, and an honest interpretation.

    context keys (all optional booleans): unknown_sender,
    platform_safety_warning, unsolicited_first_contact,
    photo_attached_unprompted, no_prior_interaction
    """
    context = context or {}
    moves = detect_moves(text, recipient_name)
    move_set = set(moves)

    matched_sequences = []
    pattern_score = 0.0

    for seq, weight, description in OBSERVED_SEQUENCES:
        if all(m in move_set for m in seq):
            # Only count it as a *sequence* if the moves appear in order
            try:
                positions = [moves.index(m) for m in seq]
                in_order = positions == sorted(positions)
            except ValueError:
                in_order = False
            if in_order:
                matched_sequences.append({"sequence": list(seq), "weight": weight, "why": description})
                pattern_score += weight

    standalone_hits = []
    for move, weight in STANDALONE_WEIGHTS.items():
        if move in move_set:
            standalone_hits.append({"move": move, "weight": weight})
            pattern_score += weight

    context_score = 0.0
    context_hits = []
    for key, weight in CONTEXT_WEIGHTS.items():
        if context.get(key):
            context_hits.append({"signal": key, "weight": weight})
            context_score += weight

    # Exculpatory signals: a stated, checkable reason for contact
    exculpatory_score = 0.0
    exculpatory_hits = []
    for name, (pattern, weight) in EXCULPATORY.items():
        m = pattern.search(text or "")
        if m:
            exculpatory_hits.append({"signal": name, "weight": weight, "matched": m.group(0)[:40]})
            exculpatory_score += weight

    total = max(0.0, pattern_score + context_score + exculpatory_score)

    # Interpretation is deliberately conservative. Text pattern alone
    # never earns a high rating - it needs context corroboration.
    if exculpatory_hits and total < 0.8:
        level = "LOW"
        interpretation = (
            "Sender states a specific, checkable reason for contact - the shape "
            "matches a normal legitimate first message, not a templated opener."
        )
    elif context_score == 0:
        interpretation = (
            "Text pattern only, no context supplied. This shape is extremely "
            "common in legitimate first messages - NOT actionable alone."
        )
        level = "INFO"
    elif total >= 1.4:
        level = "HIGH"
        interpretation = "Templated opener shape combined with multiple context red flags."
    elif total >= 0.8:
        level = "MEDIUM"
        interpretation = (
            "Opener matches an observed template shape and context is suspicious. "
            "Note: no specific reason for contact was stated, which is itself "
            "part of the pattern."
        )
    else:
        level = "LOW"
        interpretation = "Weak or partial match."

    return {
        "moves_found": moves,
        "matched_sequences": matched_sequences,
        "standalone_moves": standalone_hits,
        "context_signals": context_hits,
        "exculpatory_signals": exculpatory_hits,
        "pattern_score": round(pattern_score, 2),
        "context_score": round(context_score, 2),
        "exculpatory_score": round(exculpatory_score, 2),
        "total_score": round(total, 2),
        "level": level,
        "interpretation": interpretation,
    }
