# Addendum — Dakota Priority Work Completed

Follows on from `STATUS_FOR_GROK.md`. Dakota's Priority 1 and Priority 2
are done. Priority 3 (eye detection) correctly left alone — still
environment-blocked.

**Test totals: 16/16 existing + 33/33 new = 49 passing.**
New suite: `python3 test_priorities.py`

---

## Priority 2 — Gravity Check brittleness: FIXED

The hard AND-gate (`delta > 8° AND face < 3.0°`) is replaced with
graduated bands. The Bella case that exposed this now flags correctly.

Band logic was also **extracted into `classify_gravity(face, bg)`** as a
pure function, so the rules can be unit-tested directly. This mattered:
on real photos the image stage frequently returns UNKNOWN (no face or no
background lines found), which meant the decision rules previously had
**zero test coverage**. Now they have 13 cases.

| face | bg | delta | old result | new result |
|---|---|---|---|---|
| 3.93° | 28.6° | **24.7°** | LOW ❌ | **HIGH** ✅ (Bella — the bug) |
| -0.6° | 10.1° | 9.4° | HIGH | HIGH ✅ (horse, confirmed fake) |
| -3.6° | 2.4° | 1.2° | LOW | LOW ✅ (Thames) |
| 3.1° | 11.2° | 8.1° | LOW ❌ | **MEDIUM** ✅ (was discarded over 0.1°) |
| 15° | 16° | 1.0° | LOW | LOW ✅ (whole camera tilted, bg agrees) |
| 30° | 55° | 25° | LOW | **MEDIUM** ✅ (big delta, but face heavily tilted) |

The original reasoning behind the gate is preserved — if the face is
tilted a lot, the camera probably was too, so a tilted background means
nothing. But that argument weakens as delta grows, so face-level
tolerance now **scales with delta** instead of being a fixed cliff.
Whole-camera-tilt cases still correctly return LOW, so the fix didn't
just loosen everything.

---

## Priority 1 — Text/metadata track: BUILT

### `opener_matcher.py` — move-sequence matching (new)

Matches on semantic **moves** and their order, not exact phrasing.
Sequences were derived from the 9 real corpus openers, not invented.

Results on the real data:

| | Result |
|---|---|
| 9 confirmed scam openers | 6 HIGH, 2 MEDIUM, 1 INFO |
| 4 legitimate strangers | **4 LOW** |

The Bella opener — which scored **zero** in the old text engine — now
scores HIGH (1.40).

**A real false positive was found and fixed mid-build.** First version
scored legitimate strangers at MEDIUM 0.85–1.20, overlapping the scam
MEDIUM range exactly. Testing revealed why: real strangers state a
**specific, checkable reason** for contact ("your order is ready",
"about the truck you listed", "we met at the graduation", "confirming
your appointment"). Scam openers are pure social contact with no stated
purpose — and that *absence* is itself the signal. Added exculpatory
(negative-weight) rules for stated reasons. Clean separation now.

Two deliberate safety properties:
- **Text pattern alone can never exceed INFO.** Without context flags
  (unknown sender, platform warning, unsolicited), the score is capped
  as not-actionable. Millions of legitimate messages are "hi, how are
  you?" — flagging on wording alone would be useless and harmful.
- Context signals carry more weight than wording, and the platform's own
  unknown-sender warning is the single heaviest context signal.

### `profile_engine.py` — social-graph tells (extended)

Added graduated follow-ratio bands, post-count, engagement-per-follower,
and platform-suspension. Weights derived from the 6 real profiles.

| Profile | score | flags |
|---|---|---|
| angela.peterson41 | 1.60 | ratio 21x + 1 post + low engagement |
| emmie316 | 1.35 | ratio 5.1x + platform-suspended |
| cinderellakate1 | 0.55 | ratio 3.3x |
| linda.cox823 | 0.55 | ratio 2.7x |
| clarasmith0008 | 0.25 | ratio 2.6x |
| kate.alice670 | 0.30 | *no ratio flag* (1.0x — correctly not fired) |

**⚠️ REAL FALSE-POSITIVE RISK — needs Greg/Dakota decision.** A
plausible *genuine lurker* account (follows 900, has 60 followers, never
posts, few likes) scores **1.40** — nearly as high as the worst
confirmed scam at 1.60. By these metrics a real lurker and a harvesting
bot are close to indistinguishable. This is exactly why the missing
legitimate-account baseline (Dakota's item #4) is the blocker here, not
an optional nice-to-have. **Do not ship follow-ratio as a strong signal
until that baseline exists.** The bands are graduated conservatively for
this reason, and the lowest band is deliberately weak.

---

## Full-codebase review — two privacy findings

Audited every module for network calls, third-party services, and disk
writes.

**Clean:** zero network calls, zero third-party services, zero API keys
anywhere in the engine. No `requests`, `urllib`, `http`, `socket`, or
any cloud SDK. The Gemini dependency is fully gone from this codebase.
Everything runs locally. Core privacy rule holds.

**Finding 1 — FIXED.** `LocalHashDB.add()` was permanently storing the
full source file path of every image, which conflicts with the
non-negotiable rule that only hash + verdict + timestamp persist. A file
path is identifying data about the user's device and library, and it
isn't needed for matching — the perceptual hash alone is sufficient.
Path retention is now **off by default**, opt-in for dev only, and
covered by a test that asserts the default entry contains no
`source_path`.

**Finding 2 — FLAGGED, needs your call.** `opener_corpus.py` stores
third-party account handles (e.g. `linda.cox823`) in its context field.
For Greg's own local evidence log that's reasonable — it's his research
data about accounts that contacted him. But it means that module is a
**dev/research tool, not shippable as-is**: a shipped build must not
persist other people's handles on a user's device. Documented in the
module. If opener matching ships, it should carry only the derived
move-patterns in `opener_matcher.py`, never the source handles or raw
texts. `labeled_dataset.py` is the same category — dev tool, stores
image paths, don't ship.

---

## Still open (unchanged)

- **Legitimate-account baseline** — now the top blocker, see the
  false-positive risk above
- Eye/pupil segmentation — needs a networked environment
- Two full chat transcripts (opener → gift-card punchline) not yet
  processed; these would show whether the *middle* of the script is as
  templated as the opening
- Category taxonomy rethink (monetization method vs. farm type as
  independent axes)
- Type E undefined; call/audio untouched
- Vision lock and privacy rules: nothing here rebrands or rewrites the
  core. All changes are additions or fixes to existing behaviour, and
  final authority remains with Greg + Dakota.
