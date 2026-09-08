"""
adaptive_sr.adaptation.fps_adapter
==================================
Step 7 — FPS Adaptation & Feasibility Layer.

Compares video playback FPS requirements against measured SR processing latency
(from Step 5 benchmark data or Step 6 Edge runtime telemetry) and produces a
machine-readable adaptation signal (FPSAdaptationSignal).
"""

import math
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


def classify_realtime_ratio(real_time_ratio: float) -> str:
    """
    Classifies a real-time feasibility ratio into deterministic computational proximity tiers:
    - "realtime": real_time_ratio >= 1.0 (SR latency <= frame budget; computational headroom >= 0%)
    - "near_realtime": 0.75 <= real_time_ratio < 1.0 (SR latency is within 25% of frame budget)
    - "below_realtime": 0.50 <= real_time_ratio < 0.75 (SR latency takes 1.33x to 2.0x frame budget)
    - "severely_below_realtime": real_time_ratio < 0.50 (SR latency takes > 2.0x frame budget)
    - "invalid": non-positive or undefined ratio

    Note: These tiers classify computational proximity only. Adaptation policies in later steps
    determine specific runtime actions (e.g. model switching, scaling, or frame skipping).
    """
    if real_time_ratio is None or math.isnan(real_time_ratio) or real_time_ratio <= 0.0:
        return "invalid"
    if real_time_ratio >= 1.0:
        return "realtime"
    elif real_time_ratio >= 0.75:
        return "near_realtime"
    elif real_time_ratio >= 0.50:
        return "below_realtime"
    else:
        return "severely_below_realtime"


@dataclass
class FPSAdaptationSignal:
    source_fps: float
    frame_budget_ms: float
    measured_latency_ms: float
    estimated_processing_fps: float
    real_time_ratio: float
    realtime_feasible: bool
    adaptation_tier: str
    model_id: str
    scale: int
    device: str
    base_representation_id: str
    measurement_provenance: str
    decision_eligible: bool
    end_to_end_streaming_feasible: Optional[bool]
    end_to_end_status: str
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FPSAdapter:
    """
    FPS Adaptation Layer.
    Evaluates real-time SR feasibility and adaptation signals without altering underlying SR execution pipelines.
    """

    @staticmethod
    def evaluate(
        source_fps: float,
        measured_latency_ms: float,
        model_id: str = "unknown",
        scale: int = 2,
        device: str = "cpu",
        base_representation_id: str = "360p",
        measurement_provenance: str = "direct_measurement",
        decision_eligible: bool = True,
        network_rtt_ms: Optional[float] = None,
        end_to_end_latency_ms: Optional[float] = None
    ) -> FPSAdaptationSignal:
        """
        Evaluates FPS feasibility for a given source FPS and measured SR processing latency.
        """
        warnings: List[str] = []

        if source_fps is None or source_fps <= 0.0:
            raise ValueError(f"Invalid source_fps: {source_fps}. Must be a positive number.")

        if measured_latency_ms is None or measured_latency_ms <= 0.0:
            raise ValueError(f"Invalid measured_latency_ms: {measured_latency_ms}. Must be a positive number.")

        frame_budget_ms = 1000.0 / float(source_fps)
        estimated_processing_fps = 1000.0 / float(measured_latency_ms)
        real_time_ratio = frame_budget_ms / float(measured_latency_ms)
        realtime_feasible = float(measured_latency_ms) <= frame_budget_ms
        adaptation_tier = classify_realtime_ratio(real_time_ratio)

        # End-to-end streaming feasibility evaluation
        # Require complete pipeline evidence (client decode/render + network + edge processing).
        # Cloud RTT or SR latency alone is incomplete; return null and "not_evaluated" unless full e2e latency is provided.
        if end_to_end_latency_ms is not None and end_to_end_latency_ms > 0.0:
            end_to_end_streaming_feasible: Optional[bool] = (end_to_end_latency_ms <= frame_budget_ms)
            end_to_end_status = "evaluated"
        else:
            end_to_end_streaming_feasible = None
            end_to_end_status = "not_evaluated"
            warnings.append("End-to-end streaming feasibility not evaluated (complete pipeline evidence unavailable in Step 7).")

        if network_rtt_ms is not None:
            warnings.append(f"Network RTT recorded ({network_rtt_ms:.2f} ms); Cloud RTT does not substitute for complete end-to-end pipeline evidence.")

        if not decision_eligible:
            warnings.append("Configuration is not decision-eligible under Step 5 variance/session requirements.")

        return FPSAdaptationSignal(
            source_fps=float(source_fps),
            frame_budget_ms=float(round(frame_budget_ms, 4)),
            measured_latency_ms=float(round(measured_latency_ms, 4)),
            estimated_processing_fps=float(round(estimated_processing_fps, 4)),
            real_time_ratio=float(round(real_time_ratio, 4)),
            realtime_feasible=bool(realtime_feasible),
            adaptation_tier=adaptation_tier,
            model_id=str(model_id),
            scale=int(scale),
            device=str(device),
            base_representation_id=str(base_representation_id),
            measurement_provenance=str(measurement_provenance),
            decision_eligible=bool(decision_eligible),
            end_to_end_streaming_feasible=end_to_end_streaming_feasible,
            end_to_end_status=end_to_end_status,
            warnings=warnings
        )

    @classmethod
    def evaluate_from_step6_telemetry(
        cls,
        telemetry: Dict[str, Any],
        source_fps: float = 30.0
    ) -> FPSAdaptationSignal:
        """
        Evaluates FPS adaptation signal directly from a Step 6 Edge telemetry record or response header dict.
        """
        norm = {str(k).lower(): v for k, v in telemetry.items()}

        sr_time = telemetry.get("sr_processing_time") or norm.get("x-sr-processing-time")
        if sr_time is not None:
            try:
                sr_time = float(sr_time)
            except (ValueError, TypeError):
                sr_time = None

        if (sr_time is None or sr_time <= 0.0):
            edge_time = telemetry.get("edge_processing_time") or norm.get("x-edge-processing-time")
            if edge_time is not None:
                try:
                    sr_time = float(edge_time)
                except (ValueError, TypeError):
                    sr_time = None

        if sr_time is None or sr_time <= 0.0:
            raise ValueError("Telemetry record lacks valid positive SR processing time.")

        # Convert to ms if given in seconds (e.g. 0.042s -> 42.0ms)
        latency_ms = sr_time * 1000.0 if sr_time < 10.0 else sr_time

        model_id = telemetry.get("sr_model_id") or norm.get("x-sr-model") or "tinysr"
        scale = telemetry.get("sr_scale") or norm.get("x-sr-scale") or 2
        device = telemetry.get("sr_device") or norm.get("x-sr-device") or "cpu"
        base_rep = telemetry.get("representation_id") or norm.get("base_representation_id") or "360p"
        rtt_val = telemetry.get("rtt") or norm.get("x-edge-cloud-rtt")
        if rtt_val is not None and str(rtt_val).upper() != "N/A":
            try:
                rtt_float = float(rtt_val)
                rtt_ms = rtt_float * 1000.0 if rtt_float < 10.0 else rtt_float
            except (ValueError, TypeError):
                rtt_ms = None
        else:
            rtt_ms = None

        return cls.evaluate(
            source_fps=source_fps,
            measured_latency_ms=latency_ms,
            model_id=str(model_id),
            scale=int(scale),
            device=str(device),
            base_representation_id=str(base_rep),
            measurement_provenance="step6_edge_telemetry",
            decision_eligible=True,
            network_rtt_ms=rtt_ms
        )

    @classmethod
    def evaluate_from_step5_record(
        cls,
        record: Dict[str, Any],
        source_fps: float = 30.0
    ) -> FPSAdaptationSignal:
        """
        Evaluates FPS adaptation signal from a Step 5 benchmark record or fps_feasibility record.
        """
        config = record.get("config", {})
        model_id = config.get("model_id") or record.get("model_id") or "unknown"
        scale = config.get("scale") or record.get("scale") or 2
        device = config.get("device") or record.get("device") or "cpu"
        base_rep = config.get("input_id") or record.get("benchmark_video_id") or "360p"
        latency_ms = record.get("latency_ms")
        if latency_ms is None and "trial_latencies" in record:
            import numpy as np
            lats = record["trial_latencies"]
            if lats:
                latency_ms = float(np.median(lats)) * 1000.0

        if latency_ms is None or latency_ms <= 0.0:
            raise ValueError(f"Step 5 record lacks valid latency: {record}")

        decision_eligible = record.get("decision_eligible", True)
        src_fps = record.get("source_fps") or source_fps

        return cls.evaluate(
            source_fps=src_fps,
            measured_latency_ms=latency_ms,
            model_id=str(model_id),
            scale=int(scale),
            device=str(device),
            base_representation_id=str(base_rep),
            measurement_provenance="step5_benchmark",
            decision_eligible=bool(decision_eligible)
        )
