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

The labelled folders in `1strealtest` are **badly confounded**. The authentic set
is 256×256 aligned face thumbnails; the synthetic set is 640–1280px full-scene
selfies. Resolution, JPEG history and framing all correlate perfectly with the
label, so almost any statistic separates them — and almost none of that
separation is about authenticity.

So the evaluation builds two views of the same data and reports both:

| view | what it is | LOO-CV AUC | accuracy |
|---|---|---|---|
| `raw` | the files as given | **0.991** | 96.4% |
| `controlled` | both classes face-cropped to 256×256 and re-encoded at one fixed JPEG quality | **0.775** | 81.6% (sens 0.67 / spec 0.86) |

**0.775 is the honest number.** The 0.991 is what you get by measuring the folders.

You can watch the confounds evaporate:

| signal | AUC raw | AUC controlled | reading |
|---|---|---|---|
| `jpeg_blockiness` | 0.889 | **0.437** | pure compression history — was measuring the folder |
| `upsampling_peak_prominence` | 0.827 | **0.405** | same; the spikes came from resolution, not from a generator |
| `residual_energy` | 0.464 | **0.745** | real physics, and it was *hidden* by the confound |
| `ca_radial_alignment` | 0.474 | **0.736** | lens dispersion; only visible once resolution is matched |

Two signals that looked like the whole story were noise. Two that looked like
noise are the actual evidence. That inversion is the single most useful thing in
this repository, and it is only visible because the evaluation controls for
capture pipeline.

### One finding worth stating plainly

**The "AI eyes disagree with each other" heuristic is inverted on this data.**
`ocular_angle_divergence` scores AUC 0.380 — meaning *authentic* photographs show
**more** catch-light disagreement between the eyes than generated ones do. Real
eyes are small, noisy and unevenly lit; modern generators produce faces that are
*too* consistent. The system reads the signal in its measured direction and says
so in the report, rather than asserting the textbook story.

Numbers reproduce with:

```bash
python eval/build_dataset.py --authentic <real_dir> --synthetic <fake_dir> --out data/ds
python eval/run_eval.py --view data/ds/raw        --name raw
python eval/run_eval.py --view data/ds/controlled --name controlled \
       --write-calibration sentinel/calibration.json
```

Full per-signal tables live in `eval/report_raw.json` and
`eval/report_controlled.json`.

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
  scientifically appealing it sounds. Seven of thirteen signals currently score
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
  calibration.json fitted on the controlled view — the only thresholds anywhere
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
