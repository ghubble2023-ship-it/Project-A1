"""
Fusion Scorer - combines all engine flags into one verdict.
Ported directly from the original spec (this logic was already real).
"""


def fuse(all_flags: list):
    """all_flags: list of flag-lists from each engine (visual, text, profile, farm, gravity)."""
    flat = [f for sub in all_flags for f in sub]
    score = sum(f.get("weight", 0.5) for f in flat if f.get("type") not in ("not_automated",))
    critical = any(f.get("type") == "CRITICAL" for f in flat)

    if critical:
        verdict = "HIGH RISK - EYES OF GREG CONFIRMS"
        action = "Do not engage / Do not go"
    elif score > 2:
        verdict = "MEDIUM RISK"
        action = "Review details closely before proceeding"
    else:
        verdict = "LOW"
        action = "No strong signals - stay generally cautious"

    return {"score": round(score, 2), "critical": critical, "flags": flat, "verdict": verdict, "action": action}
