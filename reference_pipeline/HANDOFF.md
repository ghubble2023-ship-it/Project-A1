# Project AI / Eyes of Greg - Handoff Summary

Everything below is real: real code, real tests, real photos, real
numbers. Nothing in here is a placeholder dressed up to look finished.

## What's solid and tested right now

- **Text engine** (romance-scam language: urgency, money requests,
  move-off-platform, romance bombing) - real regex logic, tested.
- **Profile engine** (single photo, two-first-names, Telegram link) -
  real, tested.
- **Farm engine Type B & C** - real rule logic and thresholds, ported
  from the original spec, tested. Type C's catalog-language regex was
  confirmed against a real case (the "Eveyn / number 24313" profile).
- **Farm engine Type C-W (wig farm)** - real anchor comparison
  (eye-distance ratio + glasses-bridge heuristic), replacing a
  placeholder that always returned true. Honest limitation: needs
  working eye detection to run at all (see Gravity check below).
- **Stolen-image detection** - real perceptual hash (dHash) + local
  hash database, replacing fake sha256/null-lookup stubs. Proven to
  catch a resized/recompressed copy of the same photo and correctly
  ignore a different one.
- **Privacy layer** - real hash-only logging and secure-delete,
  replacing empty placeholder functions.
- **Gravity check** - real face-angle (OpenCV Haar cascades) +
  real background-line detection (Hough transform, clustered to find
  the dominant structural line rather than averaging the whole
  scene). Confirmed working correctly on a real confirmed-fake photo
  (the "horse" image: face level at 0.6°, background tilted 10.1°,
  delta 9.4° - clears the threshold). This is the first real, working
  confirmation of the core theory on an actual photo, not a synthetic
  test.
- **Teachable classifier** - a real, working pipeline (feature
  extraction -> labeled dataset -> logistic regression via
  scikit-learn) that learns feature weights from labeled examples
  instead of hand-picked thresholds. Currently trained on 24 examples
  (12 fake / 12 real). Includes a built-in guardrail that refuses to
  train below 8 examples per class, so it can't quietly produce a
  model that's just memorizing.

## What still needs work - specifically, not vaguely

1. **More labeled examples, especially ones where gravity check
   actually produces a number.** Of the 24 examples so far, only 4
   have a usable face+background angle reading (eye detection fails
   on the rest - glasses, off-angle heads, low resolution). The
   background-tilt signal came out as the top-weighted feature in the
   last training run, but that's from just 4 data points (3 fake, 1
   real) - not nearly enough to trust yet. This is the single most
   important gap: we need enough examples where gravity check
   actually runs to know if it's real signal or noise.
2. **Eye detection itself is unreliable on real photos.** Haar
   cascades (the only fully-offline option available) fail often on
   glasses, side angles, and low light. A proper facial-landmark
   model would be more robust but needs internet access to fetch
   (not available in this sandbox). This is the main technical
   blocker on gravity check, skin-uniformity, and catchlight all
   getting more real-world data.
3. **The screenshot-vs-camera confound was found AND fixed** (adding
   real+screenshot and fake+saved-file examples dropped
   `ela_max_error`'s weight from top feature to near-bottom) - but
   keep an eye out for other acquisition-method confounds as more
   data comes in (different phones, apps, compression settings).
4. **3 of the original 6 "Greg tells"** (emotionally-empty eyes,
   wax-figure quality, unnatural hair) still have no reliable
   lightweight offline heuristic - left un-automated rather than
   faked.
5. **Type E** was inherited as an undefined placeholder - never
   given real criteria, still empty.
6. **Call/audio protection (Phase 2)** untouched - no offline
   speech-to-text available in this environment.
7. **This is a Python reference implementation**, proving the logic
   works - porting to a real Android app (Kotlin, camera/accessibility
   capture, on-device model packaging) is a separate real engineering
   task.
8. **All numeric thresholds** (the 8°/3° gravity thresholds, the ELA
   cutoff, skin-uniformity cutoff, catchlight cutoffs) are starting
   guesses from the original spec or my own first pass - not
   calibrated against a large confirmed dataset yet. The classifier is
   the long-term fix for this; it just needs volume.

## Files included

Full engine code, the test suite (16/16 passing), the real labeled
dataset (24 examples), and the trained model file are all in this
package - nothing held back.
