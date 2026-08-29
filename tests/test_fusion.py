"""The fusion layer is where every threshold lives, so it gets the most tests."""

import math

import pytest

from sentinel.fusion import LLR_CLIP, Calibration, SignalCalibration, label_for
from sentinel.types import Signal


def cal(auc=0.95, mu_a=0.0, mu_s=3.0, sd=1.0):
    return SignalCalibration("s", mu_a, sd, mu_s, sd, auc)


def test_uninformative_signal_earns_no_weight():
    """A signal that did not separate the calibration set must not vote."""
    weak = SignalCalibration("weak", 0.0, 1.0, 0.1, 1.0, auc=0.52)
    assert weak.weight == 0.0

    c = Calibration({"weak": weak})
    v = c.score([Signal("weak", "Weak", 99.0)])
    assert v.total_llr == 0.0
    assert v.evidence[0].used is False
    assert "not discriminative" in v.evidence[0].skip_reason


def test_strong_signal_earns_weight_and_moves_score():
    c = Calibration({"s": cal()})
    high = c.score([Signal("s", "S", 3.0)])
    low = c.score([Signal("s", "S", 0.0)])
    assert high.p_synthetic > 0.5 > low.p_synthetic


def test_single_signal_cannot_dominate():
    """Clipping stops one wild measurement from railroading the verdict."""
    c = Calibration({"s": cal()})
    v = c.score([Signal("s", "S", 1e6)])
    assert abs(v.total_llr) <= LLR_CLIP + 1e-9


def test_unmeasured_signal_abstains_rather_than_guessing():
    c = Calibration({"s": cal()})
    v = c.score([Signal.missing("s", "S", "no face detected")])
    assert v.abstained == ["s"]
    assert v.total_llr == 0.0
    assert v.coverage == 0.0
    assert v.evidence[0].skip_reason == "no face detected"


def test_low_coverage_forces_inconclusive():
    """Never issue a confident verdict off a sliver of the evidence."""
    c = Calibration({"a": cal(), "b": cal(), "c": cal()})
    v = c.score(
        [
            Signal("a", "A", 3.0),
            Signal.missing("b", "B", "unmeasurable"),
            Signal.missing("c", "C", "unmeasurable"),
        ]
    )
    assert v.coverage < 0.35
    assert v.label == "INCONCLUSIVE_INSUFFICIENT_SIGNAL"


def test_uncalibrated_signal_is_reported_but_never_scored():
    c = Calibration({})
    v = c.score([Signal("mystery", "Mystery", 5.0)])
    assert v.total_llr == 0.0
    assert v.evidence[0].skip_reason == "no calibration for this signal"


def test_label_bands():
    assert label_for(0.95, 1.0) == "LIKELY_SYNTHETIC"
    assert label_for(0.70, 1.0) == "LEANS_SYNTHETIC"
    assert label_for(0.50, 1.0) == "INCONCLUSIVE"
    assert label_for(0.30, 1.0) == "LEANS_AUTHENTIC"
    assert label_for(0.05, 1.0) == "LIKELY_AUTHENTIC"
    assert label_for(0.99, 0.1).startswith("INCONCLUSIVE")


def test_calibration_round_trip(tmp_path):
    c = Calibration({"s": cal()}, prior_log_odds=0.25, meta={"view": "test"})
    p = tmp_path / "cal.json"
    c.dump(str(p))
    back = Calibration.load(str(p))
    assert back.prior_log_odds == 0.25
    assert back.meta["view"] == "test"
    assert back.entries["s"].auc == 0.95
    assert math.isclose(back.entries["s"].llr(3.0), c.entries["s"].llr(3.0))


def test_missing_calibration_file_is_not_fatal(tmp_path):
    c = Calibration.load(str(tmp_path / "absent.json"))
    assert c.entries == {}
