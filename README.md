# Sentinel — Project A1

Explainable forensics for the two things a scam actually arrives as: **a photograph
that isn't of anyone**, and **a conversation running a script**.

Two analysis paths share one design. Modules emit *measurements*, never verdicts.
A single calibrated fusion step turns measurements into a probability. An
explainer renders that probability as evidence a human can check, argue with, and
appeal.

```
                      ┌──────────────┐
   image ──▶ ocular ──┤              │
             optics ──┤   fusion     │──▶ verdict ──▶ Dakota explainer
           spectral ──┤ (calibrated  │      + coverage    (evidence both ways,
         provenance ──┤  log-odds)   │                     caveats, what wasn't
                      └──────────────┘                     measurable)

   chat  ──▶ playbooks ─ escalation ─ stylometry ──▶ risk band + quoted phrases
                                                     + extracted wallets/handles
                                                     + what to do next
```

---

## The headline result

Three corpora, three views, and one number that survives all of them.

The labelled folders in `1strealtest` are **badly confounded**. The authentic set
is 256×256 aligned face thumbnails; the synthetic set is 640–1280px full-scene
selfies. Resolution, JPEG history and framing all correlate perfectly with the
label, so almost any statistic separates them — and almost none of that
separation is about authenticity.

So the evaluation builds two views of the same data, then adds a **second,
independent corpus from a different generator family** (`Emergent`, StyleGAN-type
faces vs FFHQ originals, both natively 256×256 — internally matched, no
resolution confound):

| what was measured | how | AUC | accuracy |
|---|---|---|---|
| `1strealtest` raw | the files as given | 0.991 | 96.4% |
| `1strealtest` controlled | face-cropped to 256×256, one fixed JPEG quality | 0.775 | 81.6% |
| **transfer** → unseen generator | fit on `1strealtest`, scored on `Emergent`, n=25 | **0.628** | 68% |
| **pooled, both corpora** | LOO-CV over 74 images from both families | **0.625** | — |
| pooled, corpus-balanced | 24/24, so corpus cannot proxy for label | **0.618** | 66.7% |

**0.62 is the honest number.** Not 0.99, and not 0.775 either.

The 0.991 is what you get by measuring the folders. The 0.775 is what you get
after controlling capture pipeline but still measuring *one* generator — roughly
0.15 AUC of it turns out to be corpus-specific. The three independent estimates
that involve an unseen generator (transfer, pooled, balanced-pooled) all land
between 0.618 and 0.628, and the balanced run confirms the agreement is not an
artefact of corpus membership correlating with the label.

With n=25 in the transfer set the standard error is roughly ±0.11, so the honest
claim is "meaningfully better than chance, a long way short of reliable."

### What survives contact with a second generator

Pooling the two corpora re-fits every signal, and the weight rule (below) then
zeroes anything that fails to separate both. Four signals reverse direction
between corpora and are automatically demoted to weight 0 — a signal that points
at *synthetic* in one corpus and at *authentic* in the other is worse than a weak
one, because it will confidently mislabel whichever population it was not fitted
on.

| signal | AUC `1strealtest` | AUC `Emergent` | shipped weight | verdict |
|---|---|---|---|---|
| `residual_energy` | 0.745 | 0.724 | **0.33** | stable across both — the workhorse |
| `residual_kurtosis` | 0.444 | 0.205 | **0.18** | stable, and inverted (see below) |
| `corneal_env_dissimilarity` | 0.695 | 0.580 | **0.14** | real, modest |
| `ocular_offset_disparity` | 0.525 | 0.716 | 0.04 | weak but consistent in sign |
| `ca_radial_alignment` | 0.736 | 0.410 | **0.00** | SIGN FLIP — was corpus, not lens physics |
| `upsampling_peak_prominence` | 0.405 | 0.615 | **0.00** | SIGN FLIP |
| `ocular_area_ratio` | 0.450 | 0.648 | **0.00** | SIGN FLIP |
| `ca_radial_slope` | 0.606 | 0.487 | **0.00** | SIGN FLIP |
| `jpeg_blockiness` | 0.437 | 0.417 | 0.00 | pure compression history, both corpora |

`ca_radial_alignment` is the cautionary tale. In the single-corpus controlled
view it looked like the second-best signal in the system and like genuine lens
physics that the confound had been masking. Against a different generator it
points the other way. One corpus could not have told us that.

### One finding worth stating plainly

**The "AI eyes disagree with each other" heuristic does not hold up.**
`ocular_angle_divergence` scored AUC 0.380 on the controlled `1strealtest` view —
apparently a strong *inverted* signal, with authentic photographs showing **more**
catch-light disagreement than generated ones. It was tempting to write that up as
a real discovery about generator over-regularisation. On the second corpus it
scores 0.455, i.e. nothing. Pooled, it carries weight 0.01.

The residual statistics tell a better-supported version of the same story:
`residual_kurtosis` is stably *inverted* across both corpora (0.444 / 0.205) —
authentic photographs have heavier-tailed high-pass residuals than generated
ones, which is what you would expect from real sensor noise versus a learned
prior. The system reads every signal in its measured direction and discloses
inversions in the report rather than asserting the textbook story.

Numbers reproduce with:

```bash
python eval/build_dataset.py --authentic <real_dir> --synthetic <fake_dir> --out data/ds
python eval/run_eval.py --view data/ds/raw        --name raw
python eval/run_eval.py --view data/ds/controlled --name controlled

# the two that actually matter: transfer to an unseen generator, and the pooled fit
python eval/cross_corpus.py --train data/ds/controlled --test data/ds_emergent/controlled \
       --train-name 1strealtest --test-name emergent
python eval/run_eval.py --view data/ds_pooled/controlled --name pooled_2corpus \
       --write-calibration sentinel/calibration.json
```

### What this means in the product

A system with AUC 0.62 should not be handing out confident verdicts, and after
the re-fit it doesn't. Scored against the 25 `Emergent` images, the shipped
calibration returns `INCONCLUSIVE` on 18 of them, `LEANS_SYNTHETIC` on 3 (all
correct), `LEANS_AUTHENTIC` on 3 (2 correct), and **`LIKELY_SYNTHETIC` or
`LIKELY_AUTHENTIC` on none**. The probabilities stay bunched near 0.5 because
that is where the evidence actually is.

That is the intended behaviour. The banding thresholds were not tuned to produce
it — they are unchanged from the single-corpus build. Widening the evidence base
made the system quieter on its own, which is the outcome you want from a
calibration that is telling the truth about its own uncertainty.

Full per-signal tables live in `eval/report_raw.json`, `eval/report_controlled.json`,
`eval/report_transfer.json`, `eval/report_pooled.json` and
`eval/report_pooled_balanced.json`.

---

## `Social media pics`: the gate that had to exist

The third corpus in this project is not faces at all. `Social media pics` is 17
phone screenshots — Telegram, TikTok, Facebook, Threads, Messages, X, Zangi —
and running them through the image pipeline exposed the worst failure mode the
system had.

**Before the fix**, the pipeline scored every one of them without complaint:
`LEANS_SYNTHETIC` on three, `LEANS_AUTHENTIC` on four, coverage 1.00, no
indication that anything was wrong.

Those numbers were meaningless. Every signal in the pipeline is a statement
about a *capture pipeline* — sensor noise residuals, lens dispersion, JPEG
history. A screenshot has none of that: the phone's compositor drew it, so the
residual statistics describe a display buffer. The calibration, fitted on
photographs, had no idea what it was looking at, and said so with a probability
to three decimal places.

`sentinel/image/screen.py` is now a **gate, not a signal**. It contributes no
evidence to the fused score; when it fires the verdict is refused outright as
`INCONCLUSIVE_NOT_A_PHOTOGRAPH`, with `p_synthetic: null` rather than a number.
Three scale-free structural cues, none of which rely on EXIF (trivially
stripped):

| cue | screenshots (n=17) | photographs (n=80) |
|---|---|---|
| `uniform_run` — longest near-constant horizontal run, as a fraction of width | median **0.999** | max **0.402** |
| `flat_region_fraction` — 8×8 tiles with essentially zero variance | median **0.545** | max **0.299** |
| `palette_ratio` — distinct colours per pixel, nearest-neighbour resampled | median **0.188** | min **0.216** |

Photographs almost never contain a truly flat 8×8 patch, because sensors have
noise; interfaces are full of them. The rule is deliberately biased against
firing, because wrongly refusing a real photograph is worse than analysing a
screenshot:

> **recall 16/17 (94%), false positives 0/80.**

The single miss is a screenshot whose content is mostly one large photograph —
the expected boundary case, and the one where running the photo pipeline does
least harm. Those thresholds are fitted to this data and validated against phone
screenshots of chat and social apps; they are not validated against scans,
photographs-of-screens, or heavy graphic design.

The user-facing message matters as much as the detection. A refusal says *"this
is a screenshot, send me the original file"* — not *"this is fake"*. Someone who
screenshots a dating profile and asks "is this AI?" deserves an honest "I can't
tell from this" rather than a fabricated probability.

---

## `Chats` and `Chat string`: what a real scam did to the rule set

The last two corpora are conversations, and between them they supply what the
text pipeline never had — a labelled positive **and** a hard negative.

`Chat string` contains a complete gift-card romance scam, transcribed. It runs
the full funnel: cold open, location probe, isolation probe, rapid
sexualisation, an errand, a transport pretext, the card, and then four separate
requests to *photograph* it. `Chats` is 18 screenshots of an ordinary domestic
argument — emotionally charged, mentions buying a birthday present, includes an
accusation about not answering the phone. Every one of those neighbours a scam
rule.

**Before the fix**, the rule set scored them like this:

| conversation | score | band | rules fired |
|---|---|---|---|
| completed gift-card scam | 0.29 | ELEVATED | **1 of 17** (`Apple Card`) |
| domestic argument | 0.00 | MINIMAL | 0 |

Specificity was fine. Sensitivity was not: an unambiguous, completed fraud —
one where the victim had already bought the card — scored three bands below the
top on the strength of a single keyword.

The reason is that the 17 rules were written for **investment** fraud: pig
butchering, task scams, account takeover, recovery fraud. Gift-card romance
fraud runs a different funnel, and the most diagnostic moment in it was missing
entirely. A gift card in the victim's hand costs the fraudster nothing. **The
money moves when they photograph the numbers.** "Can I see a picture of it" —
repeated four times in the transcript — was invisible to the system.

Six rules were added, covering the funnel rather than the keyword:

| rule | stage | what it catches |
|---|---|---|
| `groom_locate` | 1 | early probe for a zip code or precise location |
| `groom_isolation` | 2 | "do you live alone" |
| `pay_errand` | 3 | "can you get me something at the store" |
| `pay_pretext` | 3 | unverifiable stranded / no-gas / customs-fee reason |
| `pay_proof` | 4 | **the card numbers, code, or a photo of the back** |
| `iso_guilt` | 3 | "you are ignoring me" when the target stalls |

**After**, the same two conversations:

| conversation | score | band | playbooks |
|---|---|---|---|
| completed gift-card scam | **0.68** | **HIGH** | payment_pressure, grooming, coercion |
| domestic argument | 0.00 | MINIMAL | none |

and the report now quotes the funnel back in the sender's own words, in order:
`What your zip code` → `did you live alone` → `Can you get me` → `no gas in my
car` → `Apple Card` → `Can I see a picture of it` → `You are ignoring me` →
`What of the card`.

### Two defects the new tests caught immediately

Writing the specificity tests found two bugs in the same afternoon they were
introduced, which is the argument for writing them.

**A false positive I created.** The first `pay_proof` pattern made the object
optional, so a bare *"did you see the picture I sent"* — an ordinary sentence —
scored 0.58 HIGH on its own. The rule now requires the referent (`of it`, `of
the card`, `the numbers on the back`). That costs one hit on the reference
transcript and removes the false positive; a context-free rule cannot have both,
and precision wins because the surrounding funnel still fires.

**A report that contradicted itself.** A playbook needs 0.25 to be named, but a
single low-severity grooming hit saturates to 0.20. The response was listing the
matched phrase under `payment_pressure` while `identified_playbooks` said
`none_matched`. Weak matches are now reported as `weak_matches` rather than
silently dropped, and a test asserts the two fields partition.

### The escalation metric was measuring the wrong thing

`escalation_velocity` scored `1 − (first_ask − 1) / 10`, which reaches zero at
message 11 and stays there. That is a cliff, not a decay: an ask at message 21
scored identically to an ask at message 500. Since real gift-card and romance
approaches routinely spend twenty messages building rapport before the ask, the
component contributed **exactly nothing on the cases it existed to catch** — it
scored 0.0 on the transcript above. It is now a hyperbolic decay
(`k / (k + first − 1)`), which keeps the strong preference for early asks
without ever claiming a mid-conversation ask is uninformative.

### The honest caveat

These rules were derived from **one** real scam transcript plus published
gift-card fraud tradecraft, and validated against **one** real benign
conversation. That is enough to prove the funnel was missing and to fix it; it
is nowhere near enough to quote a false-positive rate. The text scorer remains a
**rule-based rubric, not a statistically calibrated model**, and it says so in
its own output. `eval/run_text_eval.py` exists to fit a real calibration through
the same fusion machinery as the image path, and it needs a labelled transcript
corpus that does not yet exist.

---

## How the scoring works

Every threshold lives in `sentinel/calibration.json`, not in code. A calibration
entry describes how a signal is distributed under each hypothesis, so scoring a
signal is a textbook log-likelihood ratio:

```
llr = log N(x | μ_synthetic, σ_synthetic) − log N(x | μ_authentic, σ_authentic)
```

Three rules keep that honest:

- **Weight is earned, not assigned.** A signal's weight comes from the
  discriminative power it actually demonstrated (`|AUC − 0.5|`). A signal that
  didn't separate the calibration set contributes exactly nothing, however
  scientifically appealing it sounds. Nine of thirteen signals currently score
  zero, and the report names them.
- **No single signal can dominate.** Each is clipped to ±2.5 nats.
- **Unmeasurable signals abstain.** No face means the ocular signals drop out
  and `coverage` falls; below 35% coverage the verdict is forced to
  `INCONCLUSIVE_INSUFFICIENT_SIGNAL`. A confident wrong answer about someone's
  photograph is worse than no answer.

`GET /calibration` exposes the whole thing. Secret thresholds are unreviewable
thresholds.

---

## Quick start

```bash
pip install -r requirements-dev.txt
python -m pytest -q                                  # 46 tests

python -m sentinel image path/to/photo.jpg           # human-readable
python -m sentinel image path/to/photo.jpg --json    # machine-readable
python -m sentinel chat  samples/pig_butchering.json
python -m sentinel chat  samples/ordinary_chat.json  # the quiet control

uvicorn sentinel.api:app --port 8000
```

### What a report looks like

```
  verdict     LEANS_SYNTHETIC
  p(synthetic) 0.669    evidence coverage 100%

  Points to synthetic:
    - High-frequency residual energy (slight)
        The high-pass residual carries 10.7 grey levels RMS of fine-grain
        energy. Camera sensors add broadband photon and read noise to every
        exposure; generated pixels tend to be smooth underneath their detail.
        On the reference set this reading is more typical of synthetic images
        (AUC 0.75); it carries slight weight here.

  Points to authentic:
    - Catch-light angular divergence between eyes (slight)
        The catch light sits 93.1 degrees apart in angle between the two eyes.
        ... Note: this runs opposite to the usual expectation. On the reference
        set, authentic photographs scored *higher* on this measure than
        generated ones, so it is read in that learned direction.

  Not measured:
    - JPEG 8px grid strength: not discriminative on calibration set (AUC 0.44)
```

Evidence in **both** directions, every claim carrying its measurement and its
measured worth, and an explicit list of what could not be checked. A report that
only lists incriminating findings is a prosecution brief, not a forensic one.

---

## The conversation path

Rule-based and **openly uncalibrated** — no labelled corpus of scam transcripts
ships with this project, so there is no honest way to fit likelihood ratios the
way the image path does. The report says so in `scoring.method` rather than
dressing a rubric up as a measurement.

17 rules across 8 playbooks: pig butchering (sha zhu pan), task/negative-balance
scams, 2FA and remote-access takeover, recovery fraud, irreversible payment
rails, coercion, off-platform funnelling, identity evasion.

What makes it more than a keyword list:

- **Stage progression beats vocabulary.** A playbook whose hits span multiple
  funnel stages is escalated. Touching stage 1 *and* stage 3 is the signature of
  a real operation; quoting one phrase is not.
- **Saturating scores.** Ten repeats of one phrase cannot manufacture a critical
  verdict. There's a test for it.
- **Escalation velocity is scored separately.** An ask to move off-platform at
  message 2 is a different animal from the same ask at message 20.
- **Every finding quotes the span it matched**, so a user can check it against
  their own chat.
- **Wallet addresses, `t.me`/`wa.me` links and URLs are extracted verbatim** for
  reporting to a bank or fraud service.
- **It ends with what to do**, keyed to what was found. A risk score alone helps
  nobody; someone deep in a romance scam has usually already been told it's a
  scam. What changes the outcome is a concrete next step.

```
  risk        CRITICAL  (0.88)
  playbooks   pig_butchering, coercion, off_platform
  method      rule-based rubric (not statistically calibrated)

  First off-platform/payment/credential request arrived at message 2 of 5.
  That is unusually fast.

    - [msg 5] Deposit / top-up to unlock funds: 'small test deposit'
    - [msg 6] Instruction to keep it secret from family or bank: 'keep this between us'
  Identifiers seen:
    - tron_address: TJRabPrwbZy45sbavfcjinPJC18kjpRTv8
```

---

## Bugs fixed from the prototype modules

The Drive prototypes (`OpticalLoupeGlintExtractor`, `OcularAlignmentComparator`,
`AutomatedGlintLoupeExtractor`, `ModernScamClassifier`, the advanced-forensics
engines) were the starting point. Carried forward, with these corrected:

1. **They don't run on a current OpenCV.** Every module calls
   `cv2.data.haarcascades` at construction. OpenCV 5 removed the bundled cascade
   XMLs, so each one raises before analysing anything. Detection is now isolated
   in `sentinel/detect.py` behind a fallback ladder (YuNet → Haar → explicit
   abstain), the dependency is pinned, and CI asserts a backend exists.
2. **Every measurement was in raw pixels**, so the same face scored differently
   at 1024px and at 256px. Everything is now normalised by eye width or image
   diagonal — which matters enormously, because resolution *is* the confound.
3. **The chromatic-aberration search could only ever return −1, 0 or +1 pixels**
   (15×15 template in a 17×17 window) for an effect that is well under a pixel.
   The window is wider and the correlation peak is refined sub-pixel.
4. **The FFT spike count was really a resolution counter** — thresholding the raw
   spectrum at `mean + 3.8σ` mostly counts natural low-frequency structure and
   scales with image size. The spectrum is now flattened by its own radial
   average first, so a value means "anomalous *for its own frequency band*".
5. **Thresholds were magic numbers** (`anomaly < 0.42`, `circularity < 0.30`,
   `Sq >= 1.5`) with weights summing past 1.0. All replaced by fitted
   distributions, and any signal that fails to earn its keep is zeroed.
6. **The explainer could contradict its own numbers.** It picked between a
   "looks fake" and a "looks real" sentence by the sign of the score; when
   calibration learned the inverted direction, it printed *"the catch light
   falls at nearly the same angle in both eyes (93.1 degrees apart)"*. Phrasings
   are now neutral statements of fact and the direction comes from calibration.
   `tests/test_explain.py` locks this shut.
7. **One crash lost every module's output.** Failures are isolated per module.

Deliberately *not* carried over: the acoustic `GlottalPulsePhaseEngine`. Its
physics is sound, but with no labelled audio here it could only ship as
uncalibrated magic numbers, which is the exact failure mode this rewrite exists
to remove. The signal interface takes an audio module unchanged whenever data
exists to calibrate it against.

---

## Layout

```
sentinel/
  types.py         Signal / ModuleReport / Evidence / Verdict
  fusion.py        calibrated log-odds fusion, abstention, coverage
  explain.py       "Dakota" narrative layer
  detect.py        face/eye detection ladder
  calibration.json fitted on the pooled two-corpus view — the only thresholds anywhere
  image/           ocular · optics · spectral · provenance · pipeline
  text/            playbooks · classifier · stylometry · pipeline
  api.py           FastAPI (+ /forensics/ocular-comparator compatibility shim)
  cli.py           python -m sentinel
eval/              build_dataset.py · run_eval.py · reports
tests/             46 tests
samples/           demo transcripts
```

The existing Next.js console (`TrustSafetyConsole.tsx`) and Jetpack Compose
client keep working: `/forensics/ocular-comparator` still returns
`ocular_consistency_forensics` and a `dakota_report`, now backed by calibrated
numbers and carrying `supporting_authentic_findings` and `caveats` alongside the
discrepancy list.

---

## Limits — read before relying on this

- **Calibrated on 49 images (37 authentic / 12 synthetic).** That is enough to
  demonstrate the method and to rank signals. It is *not* enough to deploy
  against. Treat the shipped `calibration.json` as a worked example and refit on
  your own corpus.
- **One generator family, one authentic source.** Generalisation to other models
  is unmeasured, and the honest expectation is that it degrades.
- **Physics-based AI-image detection is genuinely hard**, and gets harder every
  model release. 0.775 AUC is a real signal, not a solved problem.
- **Never use this alone to accuse a person.** The `INCONCLUSIVE` band and the
  coverage figure exist because refusing to call it is often the correct output.
- The conversation rubric is English-only and pattern-based; it will miss
  translated and paraphrased scripts.

Local evaluation images are gitignored. Photographs of real people don't belong
in a repository.
