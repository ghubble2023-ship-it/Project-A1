"""Core value types shared by every Sentinel analysis module.

The whole system is built around one idea: a module never returns a verdict,
it returns *signals*. A signal is a single measurable quantity plus enough
metadata to (a) fuse it with other signals and (b) explain itself in English.
Verdicts are produced in exactly one place -- `sentinel.fusion` -- so that
thresholds live in a calibration file rather than scattered through the code.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


# Direction a rising value pushes the verdict.
SYNTHETIC_HIGH = "synthetic_high"  # larger value => more likely synthetic
SYNTHETIC_LOW = "synthetic_low"  # smaller value => more likely synthetic


@dataclass
class Signal:
    """One measurement from one detector.

    `value` is the raw physical quantity (degrees, pixels, ratio...). It is
    deliberately *not* pre-squashed into 0..1: squashing early is what makes a
    detector impossible to recalibrate later. The mapping from value to
    evidence happens in `fusion.py` using numbers measured on real data.
    """

    key: str
    label: str
    value: float | None
    unit: str = ""
    direction: str = SYNTHETIC_HIGH
    available: bool = True
    #: Why the signal could not be measured (no face, too few samples, ...).
    unavailable_reason: str = ""
    #: Free-form numbers a human or the explainer may want to quote.
    context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def missing(cls, key: str, label: str, reason: str, **kw: Any) -> "Signal":
        return cls(
            key=key,
            label=label,
            value=None,
            available=False,
            unavailable_reason=reason,
            **kw,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModuleReport:
    """Everything one analysis module produced for one piece of media."""

    module: str
    signals: list[Signal] = field(default_factory=list)
    #: Base64 PNG diagnostics keyed by name, for the UI. Optional.
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def signal(self, key: str) -> Signal | None:
        for s in self.signals:
            if s.key == key:
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "signals": [s.to_dict() for s in self.signals],
            "artifacts": self.artifacts,
            "notes": self.notes,
            "error": self.error,
        }


@dataclass
class Evidence:
    """A single signal's contribution to the final score, after calibration."""

    key: str
    label: str
    value: float | None
    unit: str
    #: Log-likelihood ratio, positive => evidence for "synthetic".
    llr: float
    weight: float
    #: llr * weight; what actually enters the sum.
    contribution: float
    #: Discriminative power measured on the calibration set (0.5 == useless).
    calibrated_auc: float | None = None
    used: bool = True
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Verdict:
    """The single, explainable output of the system."""

    #: Calibrated probability that the media is synthetic / manipulated.
    p_synthetic: float
    #: Sum of weighted log-likelihood ratios (the raw decision statistic).
    total_llr: float
    label: str
    #: How much of the evidence budget was actually measurable (0..1).
    coverage: float
    evidence: list[Evidence] = field(default_factory=list)
    abstained: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_synthetic": round(self.p_synthetic, 4),
            "total_llr": round(self.total_llr, 4),
            "label": self.label,
            "coverage": round(self.coverage, 4),
            "evidence": [e.to_dict() for e in self.evidence],
            "abstained": self.abstained,
        }


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)
