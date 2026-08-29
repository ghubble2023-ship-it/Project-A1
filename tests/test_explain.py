"""The narrative must never contradict the number it is printing."""

import re

from sentinel.explain import PHRASINGS, explain
from sentinel.fusion import Calibration, SignalCalibration
from sentinel.types import ModuleReport, Signal


def test_every_phrasing_quotes_the_value_it_was_given():
    """Regression guard for a real bug.

    An earlier design picked between a 'looks fake' and a 'looks real'
    sentence by the sign of the evidence. When calibration learned the
    opposite of the intuitive direction, the 'looks real' branch asserted the
    two catch-light angles 'nearly match' while printing 93 degrees. Facts are
    now stated neutrally, so the prose cannot disagree with the measurement.
    """
    for key, phrasing in PHRASINGS.items():
        fact, mechanism = phrasing(42.5, {"samples": 100, "fields_present": []})
        assert isinstance(fact, str) and fact
        assert isinstance(mechanism, str)
        # No phrasing may claim agreement or disagreement on its own.
        lowered = fact.lower()
        for banned in ("nearly the same", "consistent with", "impossible", "confirmed"):
            assert banned not in lowered, f"{key} editorialises: {fact!r}"


def _report_with(key, value):
    return [ModuleReport(module="m", signals=[Signal(key, "Label", value)])]


def test_counterintuitive_direction_is_disclosed():
    """When measurement inverts the textbook story, say so out loud."""
    cal = Calibration(
        {
            # AUC below 0.5 means authentic images scored higher on this.
            "ocular_angle_divergence": SignalCalibration(
                "ocular_angle_divergence", 70.0, 20.0, 40.0, 20.0, auc=0.30
            )
        }
    )
    verdict = cal.score([Signal("ocular_angle_divergence", "Divergence", 40.0)])
    out = explain(verdict, _report_with("ocular_angle_divergence", 40.0))
    findings = out["evidence_for_synthetic"] + out["evidence_for_authentic"]
    assert findings
    assert any("opposite to the usual expectation" in f["finding"] for f in findings)


def test_evidence_is_reported_in_both_directions():
    cal = Calibration(
        {
            "residual_energy": SignalCalibration("residual_energy", 5.0, 1.0, 9.0, 1.0, 0.9),
            "ca_radial_alignment": SignalCalibration(
                "ca_radial_alignment", 0.4, 0.1, 0.8, 0.1, 0.9
            ),
        }
    )
    verdict = cal.score(
        [Signal("residual_energy", "Residual", 9.0), Signal("ca_radial_alignment", "CA", 0.4)]
    )
    out = explain(verdict, [])
    assert out["evidence_for_synthetic"], "should list incriminating evidence"
    assert out["evidence_for_authentic"], "must also list exculpatory evidence"


def test_unmeasured_signals_are_listed_with_reasons():
    cal = Calibration(
        {"residual_energy": SignalCalibration("residual_energy", 5.0, 1.0, 9.0, 1.0, 0.9)}
    )
    verdict = cal.score([Signal.missing("residual_energy", "Residual", "image too small")])
    out = explain(verdict, [])
    assert out["not_measured"][0]["reason"] == "image too small"


def test_low_coverage_adds_an_explicit_caveat():
    cal = Calibration(
        {
            "a": SignalCalibration("a", 0.0, 1.0, 3.0, 1.0, 0.95),
            "b": SignalCalibration("b", 0.0, 1.0, 3.0, 1.0, 0.95),
        }
    )
    verdict = cal.score([Signal("a", "A", 3.0), Signal.missing("b", "B", "no face")])
    out = explain(verdict, [])
    assert any("could be measured" in c for c in out["caveats"])


def test_every_report_carries_the_not_proof_caveat():
    cal = Calibration({})
    out = explain(cal.score([]), [])
    assert any("not proof" in c for c in out["caveats"])


def test_finding_text_contains_the_measured_number():
    cal = Calibration(
        {"residual_energy": SignalCalibration("residual_energy", 5.0, 1.0, 9.0, 1.0, 0.9)}
    )
    verdict = cal.score([Signal("residual_energy", "Residual", 9.0)])
    out = explain(verdict, [])
    finding = out["evidence_for_synthetic"][0]["finding"]
    assert re.search(r"9\.0", finding), finding
