# Project AI - Eyes of Greg - Restored Core (Long-Press Architecture)

This restores the ORIGINAL design from your master template - long
press a photo/profile/conversation, get one clear answer, nothing kept
except a hash + score. It does NOT include the continuous live-screen-
capture + 5-device UDP cluster from the "Project AI rdy" handoff doc -
that was a different architecture, not part of this restore.

## Test results: 16/16 passing

```
python3 test_end_to_end.py
=== RESULTS: 16 passed, 0 failed ===
```

## What's real and tested right now

| Piece | Status |
|---|---|
| Gravity check (face angle vs. background tilt) | **Real, tested.** Background-tilt math recovers a known rotation angle within ~0.3° on synthetic test images. |
| Text engine (urgency, money requests, move-off-platform, romance bombing) | **Real, tested.** Was already real logic in your spec - ported as-is. |
| Profile engine (single photo, two-first-names, Telegram link) | **Real, tested.** Same - already real, ported as-is. |
| Farm engine Type B & Type C | **Real, tested.** Same rule logic and thresholds as your spec. |
| Farm engine Type C-W (wig farm) | **Real, newly implemented.** Was a hardcoded `return true` before. Now compares eye-distance/face-width ratio + a glasses-bridge edge heuristic across profile photos. Honest caveat below. |
| Stolen-image check | **Real, newly implemented.** Was `sha256(b){return "hash"}` + a lookup that always returned null. Now a real perceptual hash (dHash) + local JSON database with distance matching - proven to catch a resized/recompressed copy of the same image and correctly ignore a genuinely different one. |
| Photoshop/composite check (ELA) | **Real, newly implemented.** Was a placeholder string. Now real error-level analysis. |
| 3 of 6 "Greg tells" (too-perfect-flat/too-even skin, reflection/catchlight error) | **Real, newly implemented** as skin-texture-uniformity and eye-catchlight-consistency measurements. |
| Privacy layer (hash-only logging, secure delete) | **Real, newly implemented.** Was two empty comment-only functions. |

## Real-photo validation (update - tested against 4 real photos)

Tested against the actual origin "Thames case" photo plus the 3 real
wig-farm profile photos Greg provided.

**Bug found and fixed:** `detect_background_angle` originally only
looked for near-VERTICAL structural lines (door frames, walls) - blind
to near-HORIZONTAL ones (horizons, railings), which is exactly what
dominates a river/bridge photo. Fixed to consider both; re-validated
against synthetic tests for both cases (~0.1-0.2° accuracy either way),
no regression.

**Bug found and fixed:** `detect_face_angle` originally used the
separate left/right-eye-split cascades with strict settings, taking
only the first detection from each. This failed on every real photo
tested, including the Thames-case photo itself. Replaced with the
general eye cascade + taking the two largest detections - now succeeds
on the Thames photo.

**Result on the actual Thames-case photo:** face_angle -3.6°,
background_angle 5.2°, delta 1.6° - below the 8° CRITICAL threshold.
Honest result, not tuned to hit "10.5°." Two possible explanations:
(a) this may not be the exact photo the 10.5° figure came from, or
(b) averaging every detected line in the whole background dilutes the
one specific line a trained eye focuses on (e.g. just the railing
nearest the subject, not also distant building edges that happen to be
level). Worth discussing before tuning further - isolating the single
most prominent nearby line instead of scene-averaging is the likely
next fix, but that's a real design decision, not a threshold tweak.

**Result on the 3 wig-farm photos:** face-angle detection still
returns None on all 3, even after the fix - all three subjects wear
glasses with somewhat off-axis head angles, which Haar cascades handle
poorly. Confirmed, stable limitation, not a bug - it correctly refuses
to guess rather than returning a wrong number.

## Dominant-line clustering (implemented per Meta's described methodology)

Meta's own account of how she got 10.5° on the Thames photo: she
identified ONE dominant line (the river embankment), not an average
across the whole scene. Implemented: candidate lines are now clustered
by angle, and the deviation reported comes from whichever cluster
carries the most total line length (the actual dominant structural
line), not a blend of everything detected. Also added a low-evidence
guard - a single short, lonely line (see emma1 below) no longer
produces a confident-looking number.

Regression check: synthetic vertical (door-frame) and horizontal
(railing/horizon) cases both still recover the known angle to within
~0.1-0.2°. All 16 end-to-end tests still pass.

Real-photo re-test:

| Photo | face_angle | bg_angle | delta |
|---|---|---|---|
| Thames (origin case) | -3.6° | 2.4° | 1.2° - moved further from 10.5°, not closer |
| emma1 | -6.0° | None | low-evidence guard now correctly declines (was a shaky 1-line reading before) |
| emma2 | None | 0.0° | face detection still fails on off-angle look |
| wig-farm x3 | None | None | still fails eye detection - confirmed, stable |

**The honest, important part:** implementing Meta's own described
method did NOT reproduce her 10.5° on the actual photo - it moved
further away. Worth asking her directly: did she run real pixel-level
line-detection code on this image, or describe/estimate the tilt
conversationally? A language model describing an image can produce a
specific, plausible-sounding number without it coming from an actual
measurement - very different from a verified pixel computation, and it
matters for whether "10.5°" is worth calibrating against.

## What's honestly NOT done yet

1. **Face-dependent pieces are now tested against real faces, with
   mixed results (see above).** Gravity check works on a clear,
   glasses-free face; skin uniformity and Type C-W still need a case
   where eye detection actually succeeds to be validated for real.
2. **3 of the 6 original "Greg tells" are left un-automated on
   purpose**: emotionally-empty eyes, wax-figure quality, unnatural
   hair. There's no reliable lightweight offline heuristic for these -
   faking one would just be a fancier version of the same stub
   problem. They need either a trained model or your own review for
   now.
3. **Type C-W's "same person, different wig" match is a coarse
   proxy** (eye-distance ratio + rough glasses detection), not real
   face-recognition. It's a real first-pass filter worth a human
   look, not a confirmed match on its own.
4. **Deepfake detection is a heuristic composite (ELA + skin
   uniformity), not a trained classifier.** A genuinely strong
   deepfake detector needs a trained neural network and labeled
   training data - that's a separate, bigger undertaking.
5. **Call protection (Phase 2 - live number + speech check) wasn't
   touched today.** No offline speech-to-text library is available in
   this sandbox and there's no internet here to install one - that's
   a separate task with a different tool requirement.
6. **This is Python, not Android/Kotlin.** It's a real, tested
   reference implementation proving the logic actually works -
   porting it to a real Android app (Kotlin + a camera/accessibility
   capture step + ML Kit or TFLite where useful) is a real, separate
   engineering step. I can't compile or run an APK in this sandbox.
7. **All thresholds (ELA error cutoff, skin-variance cutoff,
   catchlight mismatch cutoff, ratio tolerance) are starting guesses**,
   same as the original 8°/10.5° gravity numbers - they need
   calibrating against a batch of real confirmed-fake vs. confirmed-
   real photos, not just trusted as-is.

## Files

```
project_ai/
├── project_ai.py              <- long_press_scan() entry point
├── test_end_to_end.py          <- 16 passing tests
└── engine/
    ├── gravity_check.py
    ├── text_engine.py
    ├── profile_engine.py
    ├── farm_engine.py
    ├── visual_engine.py
    ├── stolen_image_db.py
    ├── fusion_scorer.py
    └── privacy.py
```

## How to actually test it yourself

```python
from project_ai import long_press_scan
from engine.stolen_image_db import LocalHashDB

db = LocalHashDB()  # persists to engine/hash_db.json between runs
result = long_press_scan(
    "path/to/photo.jpg",
    ocr_text="...",       # or use project_ai.run_real_ocr(path) for real OCR
    profile_meta={"photo_count": 1, "display_name": "Emily Rose", "bio": "..."},
    farm_context={},
    stolen_db=db,
)
print(result)
```
