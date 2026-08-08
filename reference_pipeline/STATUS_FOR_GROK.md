# Project AI / Eyes of Greg — Status Summary

Prepared for Dakota (Grok). Everything below is real: real code, real
tests, real photos, real numbers. Where something doesn't work, it says
so plainly.

---

## 1. Working and validated

### Rule engines (no models needed, on-device ready)
| Component | Status |
|---|---|
| **Text engine** — urgency, money requests, move-off-platform, romance bombing | Real regex logic, tested |
| **Profile engine** — single photo, two-first-names, Telegram link in bio | Real, tested |
| **Farm engine Type B** — platform engagement-retention bot | Real rule logic, tested |
| **Farm engine Type C** — physical-lure/catalog pattern | Real, tested. Confirmed against a live case: the "Eveyn" profile caption *"She, number 24313…"* trips the catalog-language regex exactly as designed |
| **Privacy layer** — hash-only logging, secure delete | Real implementation (was previously two empty stub functions) |

### Stolen-image detection
Real perceptual hash (dHash) + local JSON hash database. Replaced a
fake `sha256(){return "hash"}` stub and a lookup that always returned
null. **Proven**: matches a resized/recompressed copy of the same
photo, correctly ignores a genuinely different one.

### Gravity check
Real face-angle detection (eye-line via Haar cascades) + real
background-line detection (Canny + Hough transform, clustered to find
the *dominant* structural line rather than averaging the whole scene).

Two real bugs found and fixed by testing on actual photos:
1. Originally only looked for near-**vertical** lines (door frames) —
   blind to near-**horizontal** ones (horizons, railings), which is
   exactly what dominates a river/bridge photo.
2. Original eye-detection method failed on every real photo tested.
   Replaced with a more robust approach.

Validated against synthetic images with known rotation: recovers the
angle to within ~0.1–0.2° for both vertical and horizontal reference
lines.

**First real confirmed hit**: the "horse" photo (confirmed fake) —
face level at 0.6°, background tilted 10.1°, delta 9.4°, clears the 8°
threshold. This is the first end-to-end validation of the core theory
on a real photo.

### Teachable classifier
Real pipeline: feature extraction → labeled dataset → logistic
regression (scikit-learn). Learns its own weights instead of using
hand-picked thresholds. Currently **14 fake / 14 real** labeled photos.

Has a built-in guardrail that refuses to train below 8 examples per
class, so it cannot quietly produce a model that has only memorized
the training set.

**Major win here — a confound was caught and broken.** In the first
training pass, `ela_max_error` (recompression artifacts) came out as
the dominant feature. Problem: at that point *every* fake was a
screenshot and *every* real was a direct camera photo, so the model may
have been learning "is this a screenshot," not "is this fake." After
adding real-photos-as-screenshots and fakes-as-saved-files, that
feature's weight dropped from 1.169 to 0.152 — confirming the confound
was real and is now broken.

### Opener corpus (new today, strongest lead)
Separate corpus for the **text of how scam conversations begin**, with
real TF-IDF + cosine-similarity clustering. Validated on synthetic
data first (correctly grouped four "Hello my name is [X]" variants
while leaving differently-worded ones alone).

**9 real openers logged.** First real clustering result:

Exact-wording clustering found one cluster of 4 across *unrelated*
accounts:
- "How are you doing. Where are you located"
- "Hello darling. How are you doing?"
- "Where are you from"
- "Where are you from dear?"

Semantic-move analysis was stronger than exact wording:

| Move | Frequency |
|---|---|
| Greeting (hello/hi/hey) | 6/9 |
| "How are you doing" | 6/9 |
| Asks location | 3/9 |
| Endearment (darling/dear) | 2/9 |

**Conclusion: the wording varies, the *moves* don't.** Detection should
match on move-sequence, not exact phrasing.

### Follow-ratio profile tell (new today)
Real data from the TikTok batch:

| Handle | Following | Followers | Ratio |
|---|---|---|---|
| angela.peterson41 | 2,061 | 97 | **21.2×** |
| emmie316 | 9,997 | 1,960 | 5.1× |
| cinderellakate1 | 704 | 216 | 3.3× |
| linda.cox823 | 4,682 | 1,740 | 2.7× |
| clarasmith0008 | 1,333 | 517 | 2.6× |
| kate.alice670 | 838 | 831 | 1.0× |

Five of six above 2.5×. Angela at 21× with **1 post and 18 likes** is
textbook. Independent corroboration: emmie316's account has since been
suspended by TikTok.

### Platform safety warning as a signal
Most of these arrived with TikTok's own built-in "Mark this message as
safe? / Report" unknown-sender warning attached. That's a
platform-level flag available *before* any image or text analysis runs
— worth scoring alongside the profile tells.

---

## 2. Needs work — specifically

### Highest priority: the metadata + text track
This is the cheapest and strongest lead, and it needs **no models and
no new environment**. Two things to build:
- Follow-ratio + post-count + likes as real scored features
- Move-sequence matching in the text engine (not exact-phrase matching)

**Blocker on the follow-ratio rule**: we have zero *genuine* accounts
measured for comparison. Without a real baseline we don't know whether
2.7× is actually unusual. Needs a batch of known-legit profiles.

### Text engine has a real coverage gap
The actual opener that started the SMS case — *"Hello my name is
Bella❤️ How are you doing today?"* — triggers **zero** flags in the
current text engine. That is not a bug, it's the design point Greg
identified: openers are deliberately engineered to look harmless, and
the real ask comes later. The engine currently only catches
*later-stage* language (money, urgency, off-platform). The opener
corpus + move-sequence matching is the fix.

### Eye/pupil analysis — blocked
Built a real pupil-shape analyzer measuring circularity, ellipse
aspect ratio, solidity, and convexity defects ("spikes"). Validated
perfectly on synthetic ground truth:

| Test shape | circularity | aspect | spikes |
|---|---|---|---|
| Clean circle | 0.89 | 1.00 | 0 |
| Circle + catchlight | 0.89 | 1.00 | 0 |
| Circle + BIG catchlight | 0.89 | 1.00 | 0 |
| Spiky star | 0.25 | — | 8 |
| Oval pupil | 0.73 | 2.06 | 0 |

Two real bugs found and fixed along the way: the spike counter was
silently returning 0 for every image (wrong index order — caught
because it contradicted the solidity numbers), and blob selection was
sometimes grabbing the eyelash/eyelid shadow band instead of the pupil.

**But it does not work on real photos yet.** Real eyes score
circularity 0.12–0.53 with 2–5 spikes across the board, versus 0.89 for
a synthetic circle. Something in real-photo segmentation still
fragments the pupil. No threshold built on this means anything until
real round pupils actually measure round.

**CRITICAL SAFETY NOTE for whoever continues this** — irregular pupil
shape *is* a documented AI-generation tell, but real human eyes are
also irregular for real medical reasons (injury, coloboma,
corectopia, surgical iridectomy). Greg supplied a real photo of a real
person with an injury-deformed, spiky-looking pupil specifically as a
hard-negative test. Across six real eyes, that injured pupil was **not
an outlier** — on some metrics it scored *more* circular than the same
person's normal eye. Any threshold here must be trained on data that
includes real deformed eyes on the "real" side, or the tool will flag
people for having eye injuries. This is documented in
`pupil_shape.py` and must not be dropped.

### Environment blocker
Robust iris segmentation and modern face-landmark detection both need
pretrained model files downloaded from the internet. The current
sandbox has no network access, which is also why eye detection keeps
failing on glasses and off-angle heads. Classical Daugman's
integro-differential operator could be written from scratch here as an
intermediate step. The real fix is Google Colab or a local machine
with internet.

### Skin uniformity — largely a dead end
Turned out to be confounded by camera distance, not authenticity. The
same real person scored 12.4 (close selfie) vs 222 and 2,690 (distant
shots). The classifier independently learned to weight it at ~0.001,
i.e. essentially ignore it. Correct outcome, but don't expect signal
here.

### Gravity check threshold edge case
The "Bella" SMS photo produced the largest delta we've seen — face
3.9°, background 28.6°, **delta 24.7°**, over 3× the threshold — but
did **not** flag CRITICAL, because the rule requires the face be under
3.0° to count as "level" and it measured 3.93°. It missed by less than
a degree. A 24.7° mismatch is enormous regardless of a sub-degree
difference in face angle. The hard threshold needs rethinking.

### Open items not yet started
- **Category taxonomy rethink.** Greg's point stands: "gift-card scam"
  is a *monetization method*, not a farm type. Type B/C/C-W describe
  *who runs it and how many profiles*; gift cards / crypto / meet-up
  lures describe *what they want*. Those are two independent axes and
  both flags should be able to fire together.
- **Type E** — inherited as an undefined placeholder, still has no
  criteria.
- **Call/audio protection (Phase 2)** — untouched, no offline
  speech-to-text available.
- **Two full chat transcripts** (opener → gift-card punchline) were
  mentioned but not yet processed. These are high value: they'd show
  whether the *middle* of the script is as templated as the opening.

---

## 3. Honest framing note

The rule engines, perceptual hashing, privacy layer, and the
opener/profile-metadata work are solid ground to build on. The
classifier is a working *method* with 28 labeled examples — good for
demonstrating the approach, **not** enough to trust the current
weights. The image-forensics track is real but mostly blocked on either
a networked environment or substantially more labeled photos.

One scope observation worth passing along: this project now has roughly
eight parallel threads open. The metadata + text track is the cheapest,
least blocked, and has already shown real signal. The pixel-forensics
track is the most expensive and most blocked. Prioritizing accordingly
is probably the highest-leverage decision available right now.
