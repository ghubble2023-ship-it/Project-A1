"""Sentinel -- explainable media and conversation forensics for Project A1.

Two analysis paths share one design: modules emit *measurements*, a single
calibrated fusion step turns measurements into a probability, and an explainer
renders that probability as evidence a human can check.

    from sentinel import analyse_image, analyse_conversation
"""

from .fusion import Calibration
from .image.pipeline import analyse_image
from .text.pipeline import analyse_conversation
from .types import ModuleReport, Signal, Verdict

__all__ = [
    "analyse_image",
    "analyse_conversation",
    "Calibration",
    "Signal",
    "ModuleReport",
    "Verdict",
]
__version__ = "0.1.0"
