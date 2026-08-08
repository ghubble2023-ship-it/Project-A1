"""
Text Engine - romance-scam / bot language detection.
Ported directly from the original engine spec (this piece was already
real logic, not a stub - kept as-is, just given a proper home).
"""

import re

PATTERNS = {
    "urgency":         re.compile(r"urgent|immediately|right now", re.I),
    "moneyRequest":    re.compile(r"gift card|zelle|crypto|itunes|western union|wire transfer", re.I),
    "moveOffPlatform": re.compile(r"telegram|zangi|whatsapp.*private", re.I),
    "locationPushing": re.compile(r"my location|come.*place|address is", re.I),
    "romanceBombing":  re.compile(r"love you.*after (a|2) (message|day)s?|my queen|my king|soulmate", re.I),
}

WEIGHTS = {
    "urgency": 0.6,
    "moneyRequest": 0.9,
    "moveOffPlatform": 0.8,
    "locationPushing": 0.7,
    "romanceBombing": 0.7,
}


def analyze(text: str):
    """Returns a list of flag dicts, same shape as the original spec."""
    if not text:
        return []
    flags = []
    for flag_type, pattern in PATTERNS.items():
        if pattern.search(text):
            flags.append({
                "type": flag_type,
                "weight": WEIGHTS[flag_type],
                "matched_text": pattern.search(text).group(0),
            })
    return flags
