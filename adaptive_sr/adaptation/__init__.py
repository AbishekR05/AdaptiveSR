"""
adaptive_sr.adaptation
======================
Step 7 — FPS Adaptation & Feasibility Package.
"""

from adaptive_sr.adaptation.fps_adapter import (
    FPSAdapter,
    FPSAdaptationSignal,
    classify_realtime_ratio,
)
from adaptive_sr.adaptation.bitrate_adapter import (
    BitrateAdapter,
    BitrateAdaptationSignal,
)
from adaptive_sr.adaptation.edge_evaluator import (
    EdgeResourceEvaluator,
    EdgeResourceSignal,
    EdgeResourceState,
)

__all__ = [
    "FPSAdapter",
    "FPSAdaptationSignal",
    "classify_realtime_ratio",
    "BitrateAdapter",
    "BitrateAdaptationSignal",
    "EdgeResourceEvaluator",
    "EdgeResourceSignal",
    "EdgeResourceState",
]
