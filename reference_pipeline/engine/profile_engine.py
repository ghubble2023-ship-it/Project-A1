"""
Profile Engine - checks profile metadata for common fake-profile tells.
Original rules ported from the spec; social-graph rules added from real
observed data (see below).

Expected `meta` dict shape:
{
  "photo_count": int,
  "display_name": str,
  "bio": str,
  "repeated_intro": bool,   # you already flagged this line as seen before
  "following": int,         # accounts this profile follows
  "followers": int,         # accounts following this profile
  "post_count": int,
  "likes_total": int,
  "account_suspended": bool  # platform has since suspended it
}
All social-graph keys are optional - rules only fire when present.

SOCIAL-GRAPH RULES - PROVENANCE AND HONEST LIMITS:
Derived from 6 confirmed scam profiles Greg collected on TikTok:

  handle                following  followers  ratio  posts  likes
  angela.peterson41          2061         97  21.2x      1     18
  emmie316                   9997       1960   5.1x      -      -   (later suspended)
  cinderellakate1             704        216   3.3x      -      -
  linda.cox823               4682       1740   2.7x      -      -
  clarasmith0008             1333        517   2.6x      -      -
  kate.alice670               838        831   1.0x      -      -

Five of six above 2.5x. The logic is real: mass-following is how these
accounts acquire targets, and it costs nothing, while genuine followers
have to be earned.

**THE MISSING PIECE - READ BEFORE TRUSTING THESE WEIGHTS:** there is
currently NO baseline measured from known-legitimate accounts. Without
it we don't actually know whether 2.7x is unusual - plenty of real
casual users follow far more accounts than follow them back. The
thresholds below are graduated conservatively for that reason, and the
lowest band is deliberately weak. One confirmed scam profile in the set
sits at 1.0x, so this is a signal, not a test.
"""

import re

TWO_FIRST_NAMES = re.compile(r"^[A-Z][a-z]+ [A-Z][a-z]+$")
TELEGRAM_LINK = re.compile(r"t\.me|zangi", re.I)


def _follow_ratio_flags(meta: dict):
    """Graduated follow-ratio scoring. Returns a list of flags."""
    following = meta.get("following")
    followers = meta.get("followers")
    if not following or followers is None:
        return []

    ratio = following / max(1, followers)

    # Graduated bands - the low band is intentionally weak because no
    # legitimate-account baseline exists yet.
    if ratio >= 15:
        weight, band = 0.65, "extreme"
    elif ratio >= 5:
        weight, band = 0.45, "high"
    elif ratio >= 2.5:
        weight, band = 0.25, "moderate"
    else:
        return []

    return [{
        "type": "followRatioImbalance",
        "band": band,
        "weight": weight,
        "ratio": round(ratio, 1),
        "following": following,
        "followers": followers,
        "note": "Mass-following is cheap; genuine followers are not. No legitimate-account baseline measured yet.",
    }]


def _engagement_flags(meta: dict):
    """Low post count / low engagement relative to follower count."""
    flags = []
    posts = meta.get("post_count")
    followers = meta.get("followers")
    likes = meta.get("likes_total")

    if posts is not None and posts <= 1:
        flags.append({
            "type": "almostNoPosts",
            "weight": 0.45 if posts == 0 else 0.35,
            "post_count": posts,
            "note": "A profile that follows thousands but has posted ~nothing is acquiring targets, not sharing content.",
        })

    # Engagement per follower - a real account with hundreds of
    # followers accumulates likes over time; a harvesting account
    # doesn't, because it has no real audience.
    if likes is not None and followers and followers >= 50:
        per_follower = likes / followers
        if per_follower < 0.25:
            flags.append({
                "type": "lowEngagementForFollowerCount",
                "weight": 0.30,
                "likes_total": likes,
                "followers": followers,
                "likes_per_follower": round(per_follower, 3),
            })

    return flags


def analyze(meta: dict):
    if not meta:
        return []
    flags = []

    if meta.get("photo_count") == 1:
        flags.append({"type": "singlePhoto", "weight": 0.4})

    name = meta.get("display_name", "") or ""
    if TWO_FIRST_NAMES.match(name) and len(name.split()) == 2:
        flags.append({"type": "twoFirstNames", "weight": 0.3, "example": name})

    bio = meta.get("bio", "") or ""
    if TELEGRAM_LINK.search(bio):
        flags.append({"type": "telegramLink", "weight": 0.6})

    if meta.get("repeated_intro"):
        flags.append({"type": "repeatedScript", "weight": 0.5})

    flags += _follow_ratio_flags(meta)
    flags += _engagement_flags(meta)

    # The platform's own verdict is the strongest single signal
    # available - it beats any heuristic here.
    if meta.get("account_suspended"):
        flags.append({
            "type": "platformSuspendedAccount",
            "weight": 0.90,
            "note": "The platform itself removed this account - independent corroboration.",
        })

    return flags
