"""
tests/test_fuzzy_decision.py
=============================
Unit and integration test suite for Step 10 — Fuzzy Adaptive Decision Engine.
"""

import pytest
from typing import List

from adaptive_sr.adaptation.fps_adapter import FPSAdaptationSignal, FPSAdapter
from adaptive_sr.adaptation.bitrate_adapter import BitrateAdaptationSignal, BitrateAdapter
from adaptive_sr.adaptation.edge_evaluator import EdgeResourceSignal, EdgeResourceEvaluator, EdgeResourceState
from adaptive_sr.adaptation.fuzzy_engine import FuzzyAdaptiveDecisionEngine, FuzzyDecisionSignal


def create_sample_fps_signal(
    real_time_ratio: float = 1.2,
    base_representation_id: str = "360p",
    model_id: str = "tinysr",
    scale: int = 2,
    device: str = "cpu",
    decision_eligible: bool = True,
) -> FPSAdaptationSignal:
    return FPSAdaptationSignal(
        source_fps=30.0,
        frame_budget_ms=33.33,
        measured_latency_ms=25.0,
        estimated_processing_fps=40.0,
        real_time_ratio=real_time_ratio,
        realtime_feasible=real_time_ratio >= 1.0,
        adaptation_tier="realtime" if real_time_ratio >= 1.0 else "below_realtime",
        model_id=model_id,
        scale=scale,
        device=device,
        base_representation_id=base_representation_id,
        measurement_provenance="test_provenance",
        decision_eligible=decision_eligible,
        end_to_end_streaming_feasible=None,
        end_to_end_status="not_evaluated",
        warnings=[],
    )


def create_sample_bitrate_signal(
    saving_percent: float = 50.0,
    base_representation_id: str = "360p",
    model_id: str = "tinysr",
    scale: int = 2,
    device: str = "cpu",
    vmaf: float = 85.0,
    psnr: float = 35.0,
    ssim: float = 0.92,
    quality_evaluable: bool = True,
    decision_eligible: bool = True,
) -> BitrateAdaptationSignal:
    return BitrateAdaptationSignal(
        reference_representation_id="720p",
        candidate_representation_id=base_representation_id,
        reference_bitrate_bps=5000000.0,
        candidate_bitrate_bps=2500000.0,
        bitrate_saving_percent=saving_percent,
        base_resolution="640x360",
        target_resolution="1280x720",
        model_id=model_id,
        scale=scale,
        device=device,
        psnr_db=psnr if quality_evaluable else None,
        ssim=ssim if quality_evaluable else None,
        vmaf=vmaf if quality_evaluable else None,
        quality_evaluable=quality_evaluable,
        quality_provenance="test_quality",
        measurement_provenance="test_provenance",
        decision_eligible=decision_eligible,
        quality_equivalent_to_native=None,
        quality_equivalence_status="not_evaluated",
        warnings=[],
    )


def create_sample_resource_signal(
    edge_id: str = "edge_01",
    resource_feasible: bool = True,
    cpu_util: float = 25.0,
    gpu_available: bool = True,
    rtt_ms: float = 15.0,
    base_representation_id: str = "360p",
    model_id: str = "tinysr",
    scale: int = 2,
    device: str = "cpu",
) -> EdgeResourceSignal:
    return EdgeResourceSignal(
        edge_id=edge_id,
        cluster_id="cluster_01",
        model_id=model_id,
        scale=scale,
        device=device,
        base_representation_id=base_representation_id,
        target_resolution="1280x720",
        resource_feasible=resource_feasible,
        network_feasible=True,
        feasibility_status="feasible" if resource_feasible else "infeasible",
        hardware_capability={
            "cpu_cores_total": 8,
            "gpu_available": gpu_available,
            "supported_devices": ["cpu", "cuda"],
            "supported_models": ["tinysr", "real_esrgan"],
        },
        resource_availability={
            "cpu_utilization_percent": cpu_util,
            "memory_utilization_percent": 40.0,
            "gpu_utilization_percent": 10.0 if gpu_available else None,
            "gpu_memory_free_bytes": 4000000000 if gpu_available else None,
        },
        network_telemetry={
            "cloud_edge_rtt_ms": rtt_ms,
            "measured_bandwidth_mbps": 100.0,
        },
        sr_telemetry={"sr_processing_time_ms": 25.0},
        measurement_provenance="test_provenance",
        warnings=[],
    )


def test_single_feasible_candidate():
    engine = FuzzyAdaptiveDecisionEngine(min_suitability_threshold=35.0)
    fps = [create_sample_fps_signal(real_time_ratio=1.3)]
    bitrate = [create_sample_bitrate_signal(saving_percent=55.0)]
    resource = [create_sample_resource_signal(edge_id="edge_01", cpu_util=20.0)]

    result = engine.evaluate_candidates(fps, bitrate, resource)
    assert result.decision == "selected"
    assert result.selected_edge_id == "edge_01"
    assert result.selected_representation_id == "360p"
    assert result.fuzzy_suitability is not None
    assert result.fuzzy_suitability > 60.0
    assert result.baseline_comparison_ready is True


def test_multiple_feasible_candidates_highest_suitability_selected():
    engine = FuzzyAdaptiveDecisionEngine(min_suitability_threshold=35.0)
    
    # Candidate 1 (edge_01): Heavy CPU load (80%), lower RT ratio (0.85)
    fps_1 = create_sample_fps_signal(real_time_ratio=0.85, base_representation_id="360p")
    bitrate_1 = create_sample_bitrate_signal(saving_percent=30.0, base_representation_id="360p")
    resource_1 = create_sample_resource_signal(edge_id="edge_01", cpu_util=80.0, base_representation_id="360p")

    # Candidate 2 (edge_02): Idle CPU (15%), high RT ratio (1.4), high BW saving (65%)
    fps_2 = create_sample_fps_signal(real_time_ratio=1.4, base_representation_id="480p")
    bitrate_2 = create_sample_bitrate_signal(saving_percent=65.0, base_representation_id="480p")
    resource_2 = create_sample_resource_signal(edge_id="edge_02", cpu_util=15.0, base_representation_id="480p")

    result = engine.evaluate_candidates([fps_1, fps_2], [bitrate_1, bitrate_2], [resource_1, resource_2])
    assert result.decision == "selected"
    assert result.selected_edge_id == "edge_02"
    assert result.selected_representation_id == "480p"
    assert len(result.candidate_evaluations) == 2


def test_hard_infeasible_candidate_rejected():
    engine = FuzzyAdaptiveDecisionEngine()
    
    # Infeasible resource signal (e.g. CPU overloaded >= 95% in Step 9)
    resource_infeasible = create_sample_resource_signal(edge_id="edge_bad", resource_feasible=False)
    fps = create_sample_fps_signal()
    bitrate = create_sample_bitrate_signal()

    result = engine.evaluate_candidates([fps], [bitrate], [resource_infeasible])
    assert result.decision == "no_feasible_candidates"
    assert len(result.rejected_candidates) == 1
    assert "step9_resource_infeasible" in result.rejected_candidates[0]["rejection_reasons"][0]


def test_cuda_requested_gpu_unavailable_hard_rejected():
    engine = FuzzyAdaptiveDecisionEngine()
    
    resource = create_sample_resource_signal(edge_id="edge_cpu_only", device="cuda", gpu_available=False)
    fps = create_sample_fps_signal(device="cuda")
    bitrate = create_sample_bitrate_signal(device="cuda")

    result = engine.evaluate_candidates([fps], [bitrate], [resource])
    assert result.decision == "no_feasible_candidates"
    assert len(result.rejected_candidates) == 1
    assert "cuda_requested_but_gpu_unavailable" in result.rejected_candidates[0]["rejection_reasons"]


def test_tie_breaking_determinism():
    engine = FuzzyAdaptiveDecisionEngine()

    # Create two identical candidate evaluations with same suitability
    fps_a = create_sample_fps_signal(real_time_ratio=1.2, base_representation_id="360p")
    bitrate_a = create_sample_bitrate_signal(saving_percent=50.0, base_representation_id="360p")
    resource_a = create_sample_resource_signal(edge_id="edge_b", base_representation_id="360p")

    fps_b = create_sample_fps_signal(real_time_ratio=1.2, base_representation_id="360p")
    bitrate_b = create_sample_bitrate_signal(saving_percent=50.0, base_representation_id="360p")
    resource_b = create_sample_resource_signal(edge_id="edge_a", base_representation_id="360p")

    result1 = engine.evaluate_candidates([fps_a, fps_b], [bitrate_a, bitrate_b], [resource_a, resource_b])
    result2 = engine.evaluate_candidates([fps_b, fps_a], [bitrate_b, bitrate_a], [resource_b, resource_a])

    # Should break ties deterministically by edge_id ("edge_a" < "edge_b")
    assert result1.selected_edge_id == "edge_a"
    assert result2.selected_edge_id == "edge_a"


def test_minimum_suitability_threshold():
    # Set high threshold so no candidate passes
    engine = FuzzyAdaptiveDecisionEngine(min_suitability_threshold=95.0)

    fps = [create_sample_fps_signal(real_time_ratio=0.75)]
    bitrate = [create_sample_bitrate_signal(saving_percent=10.0)]
    resource = [create_sample_resource_signal(cpu_util=70.0)]

    result = engine.evaluate_candidates(fps, bitrate, resource)
    assert result.decision == "no_suitable_candidate"
    assert result.selected_candidate is None


def test_missing_quality_metrics_handling():
    engine = FuzzyAdaptiveDecisionEngine()

    fps = [create_sample_fps_signal()]
    # Bitrate signal with quality_evaluable = False
    bitrate = [create_sample_bitrate_signal(quality_evaluable=False)]
    resource = [create_sample_resource_signal()]

    result = engine.evaluate_candidates(fps, bitrate, resource)
    assert result.decision == "selected"
    assert any("missing_quality_metrics_using_neutral_fuzzy_membership" in w for w in result.warnings)


def test_missing_network_telemetry_handling():
    engine = FuzzyAdaptiveDecisionEngine()

    fps = [create_sample_fps_signal()]
    bitrate = [create_sample_bitrate_signal()]
    resource = [create_sample_resource_signal()]
    # Clear network telemetry
    resource[0].network_telemetry = {}

    result = engine.evaluate_candidates(fps, bitrate, resource)
    assert result.decision == "selected"
    assert any("missing_network_telemetry_using_neutral_network_condition" in w for w in result.warnings)


def test_missing_resource_telemetry_handling():
    engine = FuzzyAdaptiveDecisionEngine()

    fps = [create_sample_fps_signal()]
    bitrate = [create_sample_bitrate_signal()]
    resource = [create_sample_resource_signal()]
    # Clear CPU utilization telemetry
    resource[0].resource_availability = {"cpu_utilization_percent": None}

    result = engine.evaluate_candidates(fps, bitrate, resource)
    assert result.decision == "selected"
    assert any("missing_cpu_telemetry_using_neutral_resource_condition" in w for w in result.warnings)


def test_realtime_feasible_vs_decision_eligible_distinction():
    """Verify realtime_feasible=True can occur when decision_eligible=False, and engine handles signals appropriately."""
    fps = create_sample_fps_signal(real_time_ratio=1.2, decision_eligible=False)
    bitrate = create_sample_bitrate_signal(decision_eligible=True)
    resource = create_sample_resource_signal()

    assert fps.realtime_feasible is True
    assert fps.decision_eligible is False

    engine = FuzzyAdaptiveDecisionEngine()
    result = engine.evaluate_candidates([fps], [bitrate], [resource])
    assert result.decision == "selected"
    assert result.input_signal_provenance["fps"] == "step7_fps_adaptation"


def test_provenance_preservation():
    engine = FuzzyAdaptiveDecisionEngine()
    fps = [create_sample_fps_signal()]
    bitrate = [create_sample_bitrate_signal()]
    resource = [create_sample_resource_signal()]

    result = engine.evaluate_candidates(fps, bitrate, resource)
    assert "fps" in result.input_signal_provenance
    assert "bitrate" in result.input_signal_provenance
    assert "edge" in result.input_signal_provenance
    assert result.rule_inference_metadata["inference_engine"] == "Mamdani_Centroid"


def test_deterministic_repeated_decisions():
    engine = FuzzyAdaptiveDecisionEngine()
    fps = [create_sample_fps_signal(real_time_ratio=1.2)]
    bitrate = [create_sample_bitrate_signal(saving_percent=45.0)]
    resource = [create_sample_resource_signal(cpu_util=30.0)]

    result1 = engine.evaluate_candidates(fps, bitrate, resource)
    result2 = engine.evaluate_candidates(fps, bitrate, resource)

    assert result1.decision == result2.decision
    assert result1.fuzzy_suitability == result2.fuzzy_suitability
    assert result1.selected_candidate == result2.selected_candidate


def test_integration_real_step7_8_9_signals():
    """Integration test constructing actual signals from Step 7, Step 8, and Step 9 modules."""
    # Step 7 Adapter
    fps_sig = FPSAdapter.evaluate(
        source_fps=30.0,
        measured_latency_ms=22.0,
        model_id="tinysr",
        scale=2,
        device="cpu",
        base_representation_id="360p",
    )

    # Step 8 Adapter
    bitrate_sig = BitrateAdapter.evaluate(
        reference_representation_id="720p",
        candidate_representation_id="360p",
        reference_bitrate_bps=4000000.0,
        candidate_bitrate_bps=1800000.0,
        base_resolution="640x360",
        target_resolution="1280x720",
        model_id="tinysr",
        scale=2,
        device="cpu",
        psnr_db=36.5,
        ssim=0.94,
        vmaf=88.0,
    )

    # Step 9 Evaluator
    edge_state = EdgeResourceState(
        edge_id="edge_01",
        cluster_id="cluster_01",
        cpu_cores_total=8,
        gpu_available=False,
        cpu_utilization_percent=30.0,
        cloud_edge_rtt_ms=18.0,
    )
    evaluator = EdgeResourceEvaluator()
    resource_sig = evaluator.evaluate_node(
        state=edge_state,
        model_id="tinysr",
        scale=2,
        device="cpu",
        base_representation_id="360p",
        target_resolution="1280x720",
        sr_processing_time_ms=22.0,
    )

    # Step 10 Decision Engine
    engine = FuzzyAdaptiveDecisionEngine(min_suitability_threshold=35.0)
    decision_signal = engine.evaluate_candidates(
        fps_signals=[fps_sig],
        bitrate_signals=[bitrate_sig],
        resource_signals=[resource_sig],
    )

    assert decision_signal.decision == "selected"
    assert decision_signal.selected_edge_id == "edge_01"
    assert decision_signal.selected_representation_id == "360p"
    assert decision_signal.model_id == "tinysr"
    assert decision_signal.scale == 2
    assert decision_signal.device == "cpu"
    assert decision_signal.fuzzy_suitability > 70.0
    assert decision_signal.baseline_comparison_ready is True
