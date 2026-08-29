"""Plain-English rendering of a verdict -- the 'Dakota' layer.

The point of this module is defensibility. "Our model says 88% fake" cannot be
appealed, taught from, or cross-examined. "The catch light in the two eyes sits
84 degrees apart, where one light source should put them within about 20" can
be.

Structure, and why it is this shape
-----------------------------------
An earlier draft wrote two alternative sentences per signal -- one for "this
looks fake", one for "this looks real" -- and picked between them by the sign
of the evidence. That is a trap. When calibration learns a direction opposite
to the intuitive one (and on this project's data, catch-light divergence does
exactly that: authentic photographs are *less* consistent than generated ones,
because real eyes are small, noisy and unevenly lit), the "looks real" sentence
ends up asserting "the angles nearly match" about a 93-degree gap. The prose
silently contradicts the number it is printing.

So each signal now contributes exactly two things, both always true:

``fact``       a neutral statement of what was measured
``mechanism``  why that quantity is physically informative

The direction of the inference is never baked into the prose. It is stated
separately, sourced from the calibration, and it names the measured AUC so a
reader can see how much that inference is worth. A signal whose behaviour
inverts the textbook expectation is called out explicitly rather than papered
over -- that is a finding, not an embarrassment.
"""

from __future__ import annotations

from typing import Any, Callable

from .types import Evidence, ModuleReport, Verdict

#: A phrasing turns (value, context) into (neutral fact, physical mechanism).
Phrasing = Callable[[float, dict[str, Any]], tuple[str, str]]


def _fmt(x: float, places: int = 2) -> str:
    return f"{x:,.{places}f}"


def _ocular_angle(v: float, ctx: dict[str, Any]) -> tuple[str, str]:
    return (
        f"The catch light sits {_fmt(v, 1)} degrees apart in angle between the "
        f"two eyes.",
        "Two corneas are convex mirrors a few centimetres apart, so under one "
        "dominant light their highlights should land at similar angles.",
    )


def _ocular_offset(v: float, ctx: dict[str, Any]) -> tuple[str, str]:
    return (
        f"The highlight-to-pupil distance differs between the eyes by "
        f"{_fmt(v * 100, 1)}% of eye width.",
        "That distance is set by where the light sits relative to the head, "
        "which is the same for both eyes on a real face.",
    )


def _ocular_circ(v: float, ctx: dict[str, Any]) -> tuple[str, str]:
    return (
        f"The two highlights differ in roundness by {_fmt(v, 2)} on a 0-1 scale.",
        "One light source has one shape, and a cornea largely preserves it.",
    )


def _ocular_area(v: float, ctx: dict[str, Any]) -> tuple[str, str]:
    return (
        f"One eye's highlight is {_fmt(v, 1)}x the area of the other's.",
        "Two adjacent corneas see the same source at nearly the same distance, "
        "so highlight size should be close.",
    )


def _corneal_env(v: float, ctx: dict[str, Any]) -> tuple[str, str]:
    return (
        f"What the two corneas reflect differs by {_fmt(v, 2)} on a 0-1 "
        f"dissimilarity scale.",
        "Both eyes face the same room, so the surroundings they mirror should "
        "broadly agree.",
    )


def _ca_alignment(v: float, ctx: dict[str, Any]) -> tuple[str, str]:
    n = ctx.get("samples", "?")
    return (
        f"Colour fringing on edges aligns with the radial direction at "
        f"{_fmt(v, 2)} (|cos|, 1.0 = perfectly radial), over {n} edges.",
        "Glass disperses wavelengths radially about the optical axis, so real "
        "lens fringing points away from the frame centre; learned texture does "
        "not have to.",
    )


def _ca_slope(v: float, ctx: dict[str, Any]) -> tuple[str, str]:
    return (
        f"Fringing magnitude changes with distance from the centre at a slope "
        f"of {_fmt(v, 3)} pixels per unit radius.",
        "Lateral chromatic aberration strengthens away from the optical axis, "
        "so a real lens gives a clearly positive slope.",
    )


def _upsampling(v: float, ctx: dict[str, Any]) -> tuple[str, str]:
    return (
        f"The strongest periodic frequency stands {_fmt(v, 1)}x above the "
        f"typical frequency at the same radius.",
        "Transposed-convolution and pixel-shuffle upsamplers leave a regular "
        "checkerboard, which appears as isolated spikes in the spectrum.",
    )


def _blockiness(v: float, ctx: dict[str, Any]) -> tuple[str, str]:
    return (
        f"Gradient energy on the 8-pixel JPEG grid is {_fmt(v, 2)}x that of "
        f"off-grid positions.",
        "This measures compression history, not authenticity: screenshotting "
        "and resaving a genuine photo raises it too, so it is weak evidence "
        "and a known confounder.",
    )


def _resid_energy(v: float, ctx: dict[str, Any]) -> tuple[str, str]:
    return (
        f"The high-pass residual carries {_fmt(v, 1)} grey levels RMS of "
        f"fine-grain energy.",
        "Camera sensors add broadband photon and read noise to every exposure; "
        "generated pixels tend to be smooth underneath their detail.",
    )


def _resid_kurt(v: float, ctx: dict[str, Any]) -> tuple[str, str]:
    return (
        f"The residual's excess kurtosis is {_fmt(v, 1)}.",
        "Sensor grain is broad and Gaussian-ish; detail that is all edges and "
        "no grain is spiky instead.",
    )


def _exif(v: float, ctx: dict[str, Any]) -> tuple[str, str]:
    present = ctx.get("fields_present") or []
    if present:
        return (
            f"The file carries {int(round(v * 8))} of 8 camera capture fields "
            f"({', '.join(present[:4])}).",
            "Cameras write make, model, exposure and lens data; generators "
            "normally do not.",
        )
    return (
        "The file carries no camera capture metadata at all.",
        "Absence is weak evidence: nearly every social platform strips metadata "
        "on upload, so authentic photos routinely arrive bare.",
    )


def _generator_tag(v: float, ctx: dict[str, Any]) -> tuple[str, str]:
    if v >= 0.5:
        return (
            "The file's own metadata names a generative image tool.",
            "Self-declared provenance is strong when present, though it is "
            "trivially removable, so its absence proves nothing.",
        )
    return (
        "No generative-tool tag appears in the file's metadata.",
        "Tags are trivially removable, so their absence proves nothing.",
    )


PHRASINGS: dict[str, Phrasing] = {
    "ocular_angle_divergence": _ocular_angle,
    "ocular_offset_disparity": _ocular_offset,
    "ocular_circularity_delta": _ocular_circ,
    "ocular_area_ratio": _ocular_area,
    "corneal_env_dissimilarity": _corneal_env,
    "ca_radial_alignment": _ca_alignment,
    "ca_radial_slope": _ca_slope,
    "upsampling_peak_prominence": _upsampling,
    "jpeg_blockiness": _blockiness,
    "residual_energy": _resid_energy,
    "residual_kurtosis": _resid_kurt,
    "exif_capture_completeness": _exif,
    "generator_tag_present": _generator_tag,
}

#: Signals whose textbook expectation is "higher means more likely synthetic".
#: Used only to notice when calibration disagrees, never to score anything.
INTUITIVE_SYNTHETIC_HIGH = {
    "ocular_angle_divergence",
    "ocular_offset_disparity",
    "ocular_circularity_delta",
    "ocular_area_ratio",
    "corneal_env_dissimilarity",
    "upsampling_peak_prominence",
    "generator_tag_present",
}


def _strength(contribution: float) -> str:
    a = abs(contribution)
    if a >= 1.2:
        return "strong"
    if a >= 0.5:
        return "moderate"
    return "slight"


def _inference(ev: Evidence) -> str:
    leans = "synthetic" if ev.contribution > 0 else "authentic"
    auc = f"AUC {ev.calibrated_auc:.2f}" if ev.calibrated_auc is not None else "uncalibrated"
    return (
        f"On the reference set this reading is more typical of {leans} images "
        f"({auc}); it carries {_strength(ev.contribution)} weight here."
    )


def _counterintuitive(ev: Evidence) -> str | None:
    """Flag a signal whose measured direction inverts the textbook story."""
    if ev.key not in INTUITIVE_SYNTHETIC_HIGH or ev.calibrated_auc is None:
        return None
    if ev.calibrated_auc >= 0.45:
        return None
    return (
        "Note: this runs opposite to the usual expectation. On the reference "
        "set, authentic photographs scored *higher* on this measure than "
        "generated ones, so it is read in that learned direction rather than "
        "the intuitive one."
    )


def explain(verdict: Verdict, reports: list[ModuleReport]) -> dict[str, Any]:
    used = [e for e in verdict.evidence if e.used and abs(e.contribution) > 1e-6]
    against = [e for e in used if e.contribution > 0]
    supporting = [e for e in used if e.contribution < 0]

    def render(items: list[Evidence]) -> list[dict[str, Any]]:
        out = []
        for ev in items:
            ctx = _ctx_for(ev, reports)
            phrasing = PHRASINGS.get(ev.key)
            if phrasing and ev.value is not None:
                fact, mechanism = phrasing(ev.value, ctx)
            else:
                fact = f"{ev.label}: {_fmt(ev.value or 0.0)} {ev.unit}".strip()
                mechanism = ""
            parts = [fact, mechanism, _inference(ev)]
            note = _counterintuitive(ev)
            if note:
                parts.append(note)
            out.append(
                {
                    "signal": ev.key,
                    "headline": f"{ev.label} ({_strength(ev.contribution)})",
                    "measured": ev.value,
                    "unit": ev.unit,
                    "fact": fact,
                    "mechanism": mechanism,
                    "inference": _inference(ev),
                    "finding": " ".join(p for p in parts if p),
                    "calibrated_auc": ev.calibrated_auc,
                    "weight_of_evidence_nats": round(ev.contribution, 3),
                }
            )
        return out

    not_measured = [
        {"signal": e.key, "label": e.label, "reason": e.skip_reason}
        for e in verdict.evidence
        if not e.used
    ]

    caveats = [
        "This is a probabilistic assessment from physical and statistical cues, "
        "not proof. Treat it as one input to a human decision.",
    ]
    if verdict.coverage < 0.6:
        caveats.append(
            f"Only {verdict.coverage:.0%} of the available evidence could be "
            f"measured on this file, so confidence is limited."
        )
    for rep in reports:
        caveats.extend(rep.notes)
        if rep.error:
            caveats.append(f"{rep.module}: {rep.error}")

    return {
        "verdict": verdict.label,
        "probability_synthetic": round(verdict.p_synthetic, 4),
        "coverage": round(verdict.coverage, 4),
        "summary": _summary(verdict, len(against), len(supporting)),
        "evidence_for_synthetic": render(against),
        "evidence_for_authentic": render(supporting),
        "not_measured": not_measured,
        "caveats": caveats,
    }


def _ctx_for(ev: Evidence, reports: list[ModuleReport]) -> dict[str, Any]:
    for rep in reports:
        sig = rep.signal(ev.key)
        if sig is not None:
            return sig.context
    return {}


def _summary(verdict: Verdict, n_against: int, n_for: int) -> str:
    p = verdict.p_synthetic
    if verdict.label.startswith("INCONCLUSIVE"):
        return (
            f"Inconclusive. {n_against} reading(s) lean synthetic and {n_for} "
            f"lean authentic, with {verdict.coverage:.0%} of the evidence "
            f"measurable -- not enough to support a call either way."
        )
    if p >= 0.62:
        return (
            f"Leans synthetic ({p:.0%}). {n_against} reading(s) are more typical "
            f"of generated images; {n_for} point the other way."
        )
    return (
        f"Leans authentic ({1 - p:.0%} likelihood of a genuine capture). "
        f"{n_for} reading(s) match real capture optics; {n_against} do not."
    )
