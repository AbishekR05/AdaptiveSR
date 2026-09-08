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

__all__ = ["FPSAdapter", "FPSAdaptationSignal", "classify_realtime_ratio"]
