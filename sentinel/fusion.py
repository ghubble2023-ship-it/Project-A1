"""Turns a bag of signals into one calibrated, explainable verdict.

Design notes
------------
Every threshold in this system lives in a calibration file, not in code. A
calibration entry describes how a signal is distributed under each hypothesis
(authentic vs synthetic) as a Gaussian, plus the AUC that distribution pair
achieved on the calibration set.

Scoring a signal is then a textbook log-likelihood ratio::

    llr = log N(x | mu_syn, sd_syn) - log N(x | mu_auth, sd_auth)

Two guard rails keep this honest:

* **Clipping.** A single signal can contribute at most ``LLR_CLIP`` nats, so
  one blown-up measurement cannot railroad the verdict.
* **Earned weight.** A signal's weight is derived from the discriminative
  power it actually demonstrated (``|AUC - 0.5|``). A signal that did not
  separate the calibration set contributes nothing, no matter how
  scientifically appealing it sounds.

Signals that could not be measured *abstain*: they are dropped from the sum
and reported in ``Verdict.abstained``, and the fraction of weight that was
measurable is reported as ``coverage``. A verdict with low coverage is a
verdict you should not lean on, and it says so rather than quietly guessing.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable

from .types import Evidence, ModuleReport, Signal, Verdict, sigmoid

#: Maximum absolute evidence, in nats, any one signal may contribute.
LLR_CLIP = 2.5

#: A signal must beat this |AUC - 0.5| on the calibration set to earn weight.
MIN_INFORMATIVE_MARGIN = 0.08

#: Floor on a calibrated standard deviation, to avoid divide-by-almost-zero.
MIN_SD = 1e-6

DEFAULT_CALIBRATION_PATH = os.path.join(
    os.path.dirname(__file__), "calibration.json"
)


@dataclass
class SignalCalibration:
    key: str
    mu_authentic: float
    sd_authentic: float
    mu_synthetic: float
    sd_synthetic: float
    auc: float
    n_authentic: int = 0
    n_synthetic: int = 0

    @property
    def margin(self) -> float:
        return abs(self.auc - 0.5)

    @property
    def weight(self) -> float:
        """Weight earned by measured separation, 0 for uninformative signals.

        Scaled so a perfect separator (AUC 1.0) earns weight 1.0 and anything
        at or below the informative margin earns 0.
        """
        if self.margin < MIN_INFORMATIVE_MARGIN:
            return 0.0
        span = 0.5 - MIN_INFORMATIVE_MARGIN
        return min(1.0, (self.margin - MIN_INFORMATIVE_MARGIN) / span)

    def llr(self, x: float) -> float:
        sa = max(self.sd_authentic, MIN_SD)
        ss = max(self.sd_synthetic, MIN_SD)
        za = (x - self.mu_authentic) / sa
        zs = (x - self.mu_synthetic) / ss
        ll_auth = -0.5 * za * za - math.log(sa)
        ll_syn = -0.5 * zs * zs - math.log(ss)
        return max(-LLR_CLIP, min(LLR_CLIP, ll_syn - ll_auth))


class Calibration:
    """A loaded set of per-signal calibrations plus a prior."""

    def __init__(
        self,
        entries: dict[str, SignalCalibration] | None = None,
        prior_log_odds: float = 0.0,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.entries = entries or {}
        self.prior_log_odds = prior_log_odds
        self.meta = meta or {}

    # -- persistence -----------------------------------------------------
    @classmethod
    def load(cls, path: str | None = None) -> "Calibration":
        path = path or DEFAULT_CALIBRATION_PATH
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        entries = {
            k: SignalCalibration(key=k, **v) for k, v in raw.get("signals", {}).items()
        }
        return cls(
            entries=entries,
            prior_log_odds=float(raw.get("prior_log_odds", 0.0)),
            meta=raw.get("meta", {}),
        )

    def dump(self, path: str) -> None:
        payload = {
            "meta": self.meta,
            "prior_log_odds": self.prior_log_odds,
            "signals": {
                k: {
                    "mu_authentic": c.mu_authentic,
                    "sd_authentic": c.sd_authentic,
                    "mu_synthetic": c.mu_synthetic,
                    "sd_synthetic": c.sd_synthetic,
                    "auc": c.auc,
                    "n_authentic": c.n_authentic,
                    "n_synthetic": c.n_synthetic,
                }
                for k, c in self.entries.items()
            },
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")

    # -- scoring ---------------------------------------------------------
    def score(self, signals: Iterable[Signal]) -> Verdict:
        evidence: list[Evidence] = []
        abstained: list[str] = []
        total = self.prior_log_odds
        weight_available = 0.0
        weight_total = 0.0

        for sig in signals:
            cal = self.entries.get(sig.key)
            weight = cal.weight if cal else 0.0
            # Uncalibrated signals are carried through for display but never
            # allowed to move the score -- we have no evidence they should.
            weight_total += weight

            if not sig.available or sig.value is None:
                abstained.append(sig.key)
                evidence.append(
                    Evidence(
                        key=sig.key,
                        label=sig.label,
                        value=None,
                        unit=sig.unit,
                        llr=0.0,
                        weight=weight,
                        contribution=0.0,
                        calibrated_auc=cal.auc if cal else None,
                        used=False,
                        skip_reason=sig.unavailable_reason or "not measured",
                    )
                )
                continue

            if cal is None:
                evidence.append(
                    Evidence(
                        key=sig.key,
                        label=sig.label,
                        value=sig.value,
                        unit=sig.unit,
                        llr=0.0,
                        weight=0.0,
                        contribution=0.0,
                        calibrated_auc=None,
                        used=False,
                        skip_reason="no calibration for this signal",
                    )
                )
                continue

            if weight == 0.0:
                evidence.append(
                    Evidence(
                        key=sig.key,
                        label=sig.label,
                        value=sig.value,
                        unit=sig.unit,
                        llr=cal.llr(sig.value),
                        weight=0.0,
                        contribution=0.0,
                        calibrated_auc=cal.auc,
                        used=False,
                        skip_reason=(
                            f"not discriminative on calibration set "
                            f"(AUC {cal.auc:.2f})"
                        ),
                    )
                )
                continue

            llr = cal.llr(sig.value)
            contribution = llr * weight
            total += contribution
            weight_available += weight
            evidence.append(
                Evidence(
                    key=sig.key,
                    label=sig.label,
                    value=sig.value,
                    unit=sig.unit,
                    llr=llr,
                    weight=weight,
                    contribution=contribution,
                    calibrated_auc=cal.auc,
                    used=True,
                )
            )

        coverage = (weight_available / weight_total) if weight_total > 0 else 0.0
        p = sigmoid(total)
        evidence.sort(key=lambda e: abs(e.contribution), reverse=True)
        return Verdict(
            p_synthetic=p,
            total_llr=total,
            label=label_for(p, coverage),
            coverage=coverage,
            evidence=evidence,
            abstained=abstained,
        )


def label_for(p: float, coverage: float) -> str:
    """Human-facing band. Low coverage always downgrades to INCONCLUSIVE.

    Refusing to call a verdict on thin evidence is a feature: a confident
    wrong answer about someone's photograph is worse than no answer.
    """
    if coverage < 0.35:
        return "INCONCLUSIVE_INSUFFICIENT_SIGNAL"
    if p >= 0.85:
        return "LIKELY_SYNTHETIC"
    if p >= 0.62:
        return "LEANS_SYNTHETIC"
    if p <= 0.15:
        return "LIKELY_AUTHENTIC"
    if p <= 0.38:
        return "LEANS_AUTHENTIC"
    return "INCONCLUSIVE"


def collect_signals(reports: Iterable[ModuleReport]) -> list[Signal]:
    out: list[Signal] = []
    for rep in reports:
        out.extend(rep.signals)
    return out
