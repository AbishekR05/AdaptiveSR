"""
tests.test_end_to_end_runtime
=============================
Step 11 — End-to-End Real-Time AdaptiveSR Integration and Runtime Tests (A–N + Hardened Regression Tests).
"""

import json
import pytest
import time
from pathlib import Path
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from adaptive_sr.runtime.orchestrator import (
    AdaptiveSRRuntime,
    DynamicConditionProfile,
    QualityEvidenceStore,
    EdgeRegistry,
)
from adaptive_sr.runtime.telemetry import ChunkTelemetry, RuntimeState
from adaptive_sr.adaptation.fps_adapter import FPSAdapter, FPSAdaptationSignal
from adaptive_sr.adaptation.bitrate_adapter import BitrateAdapter, BitrateAdaptationSignal
from adaptive_sr.adaptation.edge_evaluator import EdgeResourceEvaluator, EdgeResourceSignal, EdgeResourceState
from adaptive_sr.adaptation.fuzzy_engine import FuzzyAdaptiveDecisionEngine, FuzzyDecisionSignal


@pytest.fixture(scope="session", autouse=True)
def ensure_sample_chunks():
    """Ensures synthetic .mp4 chunks exist in cloud storage for baseline tests."""
    from adaptive_sr.shared.config import CLOUD_STORAGE_DIR

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for rep_id in ["360p", "480p"]:
        target_dir = Path(CLOUD_STORAGE_DIR) / "videos" / "sample" / rep_id
        target_dir.mkdir(parents=True, exist_ok=True)
        for chunk_id in ["0000", "0001", "0002"]:
            chunk_path = target_dir / f"{chunk_id}.mp4"
            if not chunk_path.exists() or chunk_path.stat().st_size == 0:
                out = cv2.VideoWriter(str(chunk_path), fourcc, 30, (640, 360) if rep_id == "360p" else (854, 480))
                for _ in range(60):
                    frame = np.zeros((480, 854, 3) if rep_id == "480p" else (360, 640, 3), dtype=np.uint8)
                    frame[:, :] = (100, 150, 200)
                    out.write(frame)
                out.release()


@pytest.fixture(autouse=True)
def mock_cloud_client():
    """Intercepts requests to Cloud Origin and redirects them by calling app functions directly."""
    import requests
    from adaptive_sr.services.cloud.app import get_manifest, get_chunk

    def mock_get(url, *args, **kwargs):
        response = requests.Response()
        if "/health" in url or url.endswith("/health"):
            response.status_code = 200
            response._content = b'{"status": "ok"}'
            response.headers["content-type"] = "application/json"
        elif "/manifest" in url:
            video_id = url.split("/videos/")[-1].split("/manifest")[0]
            try:
                data = get_manifest(video_id)
                response._content = json.dumps(data).encode("utf-8")
                response.status_code = 200
                response.headers["content-type"] = "application/json"
            except Exception as e:
                response.status_code = 404
                response._content = json.dumps({"detail": str(e)}).encode("utf-8")
        elif "/chunks/" in url:
            parts = url.split("/videos/")[-1].split("/")
            video_id = parts[0]
            representation_id = parts[1]
            chunk_id = parts[3]
            try:
                file_response = get_chunk(video_id, representation_id, chunk_id)
                with open(file_response.path, "rb") as f:
                    response._content = f.read()
                response.status_code = 200
                response.headers["content-type"] = "video/mp4"
            except Exception as e:
                response.status_code = 404
                response._content = json.dumps({"detail": str(e)}).encode("utf-8")
        else:
            response.status_code = 404
        return response

    with patch("requests.get", side_effect=mock_get) as mock:
        yield mock


# ----------------------------------------------------------------------------
# Test A: Single-chunk end-to-end execution
# ----------------------------------------------------------------------------
def test_A_single_chunk_end_to_end_execution():
    runtime = AdaptiveSRRuntime(video_id="sample")
    telemetry = runtime.process_chunk("0000", chunk_duration=2.0)

    assert isinstance(telemetry, ChunkTelemetry)
    assert telemetry.identity.video_id == "sample"
    assert telemetry.identity.chunk_id == "0000"
    assert telemetry.decision.decision in ["selected", "no_suitable_candidate"]

    # Verify machine readability via JSON serialization
    t_dict = telemetry.to_dict()
    t_json = json.dumps(t_dict)
    assert len(t_json) > 0
    assert "identity" in t_dict
    assert "delivery_mode" in t_dict
    assert "requested_configuration" in t_dict
    assert "executed_configuration" in t_dict
    assert "decision" in t_dict
    assert "timing" in t_dict
    assert "network" in t_dict
    assert "resource" in t_dict
    assert "buffer" in t_dict
    assert "provenance" in t_dict


# ----------------------------------------------------------------------------
# Test B: Multi-chunk sequential execution
# ----------------------------------------------------------------------------
def test_B_multi_chunk_sequential_execution():
    runtime = AdaptiveSRRuntime(video_id="sample")
    chunks = ["0000", "0001", "0002"]

    for chunk_id in chunks:
        t = runtime.process_chunk(chunk_id, chunk_duration=2.0)
        assert t.identity.chunk_id == chunk_id

    assert runtime.state.decision_count == 3
    assert len(runtime.telemetry_history) == 3
    assert runtime.state.buffer_seconds > 0.0


# ----------------------------------------------------------------------------
# Test C: Step 7/8/9 → Step 10 integration
# ----------------------------------------------------------------------------
def test_C_step7_8_9_to_step10_integration():
    q_store = QualityEvidenceStore([
        {
            "video_id": "sample",
            "chunk_id": "0000",
            "representation_id": "360p",
            "model_id": "tinysr",
            "scale": 2,
            "device": "cpu",
            "psnr": 36.0,
            "ssim": 0.95,
            "vmaf": 88.0,
        }
    ])
    runtime = AdaptiveSRRuntime(video_id="sample", quality_store=q_store)
    telemetry = runtime.process_chunk(
        "0000",
        observed_network_mbps=15.0,
        observed_rtt_seconds=0.02,
        observed_edge_cpu=20.0,
        observed_edge_gpu=30.0,
    )

    assert telemetry.provenance.fps_signal_provenance["signal_count"] > 0
    assert telemetry.provenance.bitrate_signal_provenance["signal_count"] > 0
    assert telemetry.provenance.resource_signal_provenance["signal_count"] > 0
    assert telemetry.decision.decision == "selected"


# ----------------------------------------------------------------------------
# Test D: Selected Step 10 configuration is actually honored
# ----------------------------------------------------------------------------
def test_D_selected_configuration_honored():
    executed_requests = []

    def mock_execution_handler(candidate):
        executed_requests.append(candidate)
        return {
            "bytes_received": 150000,
            "total_chunk_completion_time": 0.1,
            "download_transfer_time": 0.08,
            "sr_processing_time": 0.02,
            "edge_id": candidate.get("edge_id", "edge_01"),
            "cluster_id": "cluster_01",
            "request_id": "test_req_001",
        }

    runtime = AdaptiveSRRuntime(video_id="sample", execution_handler=mock_execution_handler)
    telemetry = runtime.process_chunk("0000")

    assert len(executed_requests) == 1
    req = executed_requests[0]
    exec_cfg = telemetry.executed_configuration

    assert exec_cfg is not None
    assert req["base_representation_id"] == exec_cfg["representation_id"]
    assert req["target_resolution"] == exec_cfg["target_resolution"]
    assert req["model_id"] == exec_cfg["model_id"]
    assert req["scale"] == exec_cfg["scale"]
    assert req["device"] == exec_cfg["device"]


# ----------------------------------------------------------------------------
# Test E: Buffer update correctness
# ----------------------------------------------------------------------------
def test_E_buffer_update_correctness():
    runtime = AdaptiveSRRuntime(video_id="sample", initial_buffer_seconds=1.0)

    def mock_handler(candidate):
        time.sleep(0.05)
        return {
            "bytes_received": 100000,
            "total_chunk_completion_time": 0.4,
            "download_transfer_time": 0.3,
            "sr_processing_time": 0.1,
            "edge_id": "edge_01",
            "cluster_id": "cluster_01",
            "request_id": "req_buf",
        }

    runtime.execution_handler = mock_handler
    telemetry = runtime.process_chunk("0000", chunk_duration=2.0)

    assert telemetry.buffer.buffer_before == pytest.approx(1.0)
    assert telemetry.timing.client_elapsed_seconds > 0.0
    assert telemetry.buffer.stall_count == 0
    assert telemetry.buffer.stall_duration == 0.0


# ----------------------------------------------------------------------------
# Test F: Stall detection
# ----------------------------------------------------------------------------
def test_F_stall_detection():
    runtime = AdaptiveSRRuntime(video_id="sample", initial_buffer_seconds=0.01)

    def mock_slow_handler(candidate):
        time.sleep(0.1)
        return {
            "bytes_received": 100000,
            "total_chunk_completion_time": 0.5,
            "download_transfer_time": 0.4,
            "sr_processing_time": 0.1,
            "edge_id": "edge_01",
            "cluster_id": "cluster_01",
            "request_id": "req_stall",
        }

    runtime.execution_handler = mock_slow_handler
    telemetry = runtime.process_chunk("0000", chunk_duration=2.0)

    assert telemetry.buffer.buffer_before == pytest.approx(0.01)
    assert telemetry.buffer.stall_count == 1
    assert telemetry.buffer.stall_duration > 0.0
    assert runtime.state.stall_count == 1


# ----------------------------------------------------------------------------
# Test G: Configuration-switch tracking
# ----------------------------------------------------------------------------
def test_G_configuration_switch_tracking():
    runtime = AdaptiveSRRuntime(video_id="sample")
    configs = [
        {"edge_id": "edge_01", "base_representation_id": "360p", "target_resolution": "720p", "model_id": "tinysr", "scale": 2, "device": "cpu"},
        {"edge_id": "edge_01", "base_representation_id": "360p", "target_resolution": "720p", "model_id": "tinysr", "scale": 2, "device": "cpu"},
        {"edge_id": "edge_01", "base_representation_id": "480p", "target_resolution": "1080p", "model_id": "tinysr", "scale": 2, "device": "cpu"},
        {"edge_id": "edge_02", "base_representation_id": "480p", "target_resolution": "1080p", "model_id": "tinysr", "scale": 2, "device": "cuda"},
    ]

    for idx, cfg in enumerate(configs):
        def mock_h(c, conf=cfg):
            return {
                "bytes_received": 100000,
                "total_chunk_completion_time": 0.1,
                "download_transfer_time": 0.08,
                "sr_processing_time": 0.02,
                "edge_id": conf["edge_id"],
                "cluster_id": "cluster_01",
                "request_id": f"req_{idx}",
            }

        with patch.object(FuzzyAdaptiveDecisionEngine, "evaluate_candidates") as mock_eval:
            mock_signal = FuzzyDecisionSignal(
                decision="selected",
                selected_candidate=cfg,
                selected_edge_id=cfg["edge_id"],
                selected_representation_id=cfg["base_representation_id"],
                target_resolution=cfg["target_resolution"],
                model_id=cfg["model_id"],
                scale=cfg["scale"],
                device=cfg["device"],
                fuzzy_suitability=75.0,
                suitability_tier="high",
                min_suitability_threshold=35.0,
                candidate_evaluations=[],
                rejected_candidates=[],
                input_signal_provenance={},
                rule_inference_metadata={},
                warnings=[],
            )
            mock_eval.return_value = mock_signal
            runtime.execution_handler = mock_h
            runtime.process_chunk(f"000{idx}")

    assert runtime.state.configuration_switch_count == 2


# ----------------------------------------------------------------------------
# Test H: Dynamic-condition adaptation
# ----------------------------------------------------------------------------
def test_H_dynamic_condition_adaptation():
    profile = DynamicConditionProfile(
        enabled=True,
        network_bandwidth_map={"0000": 20.0, "0001": 1.0, "0002": 20.0},
        rtt_map={"0001": 0.25},
    )
    runtime = AdaptiveSRRuntime(video_id="sample", dynamic_profile=profile)

    t0 = runtime.process_chunk("0000")
    t1 = runtime.process_chunk("0001")
    t2 = runtime.process_chunk("0002")

    assert t0.is_injected_test_condition is True
    assert "network_bandwidth" in t0.injected_fields
    assert t0.network.measured_bandwidth_mbps == 20.0
    assert t1.is_injected_test_condition is True
    assert t1.network.measured_bandwidth_mbps == 1.0
    assert t1.network.rtt_seconds == 0.25
    assert t2.network.measured_bandwidth_mbps == 20.0


# ----------------------------------------------------------------------------
# Test I: No-suitable-candidate handling & Native Fallback
# ----------------------------------------------------------------------------
def test_I_no_suitable_candidate_handling():
    runtime = AdaptiveSRRuntime(video_id="sample", min_suitability_threshold=99.0, fallback_representation_id="360p")

    fallback_executed = False
    def mock_handler(candidate):
        nonlocal fallback_executed
        if not candidate.get("sr_requested", True):
            fallback_executed = True
        return {
            "bytes_received": 50000,
            "total_chunk_completion_time": 0.2,
            "download_transfer_time": 0.2,
            "sr_processing_time": 0.0,
            "edge_id": "edge_01",
            "cluster_id": "cluster_01",
            "request_id": "req_nosuit",
        }

    runtime.execution_handler = mock_handler
    telemetry = runtime.process_chunk("0000")

    assert telemetry.decision.decision in ["no_suitable_candidate", "no_feasible_candidates"]
    assert telemetry.delivery_mode == "native"
    assert telemetry.requested_configuration is None
    assert telemetry.executed_configuration["representation_id"] == "360p"
    assert fallback_executed is True


# ----------------------------------------------------------------------------
# Test J: CUDA-unavailable failure handling
# ----------------------------------------------------------------------------
def test_J_cuda_unavailable_failure_handling():
    runtime = AdaptiveSRRuntime(video_id="sample")

    selected_cfg = {
        "edge_id": "edge_01",
        "base_representation_id": "360p",
        "target_resolution": "720p",
        "model_id": "tinysr",
        "scale": 2,
        "device": "cuda",
    }

    def failing_handler(candidate):
        raise RuntimeError("CUDA requested for Remote SR but CUDA is unavailable on Edge.")

    with patch.object(FuzzyAdaptiveDecisionEngine, "evaluate_candidates") as mock_eval:
        mock_eval.return_value = FuzzyDecisionSignal(
            decision="selected",
            selected_candidate=selected_cfg,
            selected_edge_id="edge_01",
            selected_representation_id="360p",
            target_resolution="720p",
            model_id="tinysr",
            scale=2,
            device="cuda",
            fuzzy_suitability=80.0,
            suitability_tier="high",
            min_suitability_threshold=35.0,
            candidate_evaluations=[],
            rejected_candidates=[],
            input_signal_provenance={},
            rule_inference_metadata={},
            warnings=[],
        )
        runtime.execution_handler = failing_handler
        telemetry = runtime.process_chunk("0000")

        assert telemetry.delivery_mode == "execution_failed"
        assert telemetry.executed_configuration is None  # Failed requested SR NOT recorded as executed
        assert telemetry.error is not None
        assert "CUDA" in telemetry.error.error_message


# ----------------------------------------------------------------------------
# Test K: SR execution failure handling
# ----------------------------------------------------------------------------
def test_K_sr_execution_failure_handling():
    runtime = AdaptiveSRRuntime(video_id="sample")

    selected_cfg = {
        "edge_id": "edge_01",
        "base_representation_id": "360p",
        "target_resolution": "720p",
        "model_id": "invalid_model",
        "scale": 2,
        "device": "cpu",
    }

    def failing_handler(candidate):
        raise ValueError("Invalid or unavailable SR model 'invalid_model'")

    with patch.object(FuzzyAdaptiveDecisionEngine, "evaluate_candidates") as mock_eval:
        mock_eval.return_value = FuzzyDecisionSignal(
            decision="selected",
            selected_candidate=selected_cfg,
            selected_edge_id="edge_01",
            selected_representation_id="360p",
            target_resolution="720p",
            model_id="invalid_model",
            scale=2,
            device="cpu",
            fuzzy_suitability=60.0,
            suitability_tier="medium",
            min_suitability_threshold=35.0,
            candidate_evaluations=[],
            rejected_candidates=[],
            input_signal_provenance={},
            rule_inference_metadata={},
            warnings=[],
        )
        runtime.execution_handler = failing_handler
        telemetry = runtime.process_chunk("0000")

        assert telemetry.delivery_mode == "execution_failed"
        assert telemetry.executed_configuration is None
        assert telemetry.error.error_code == "STEP6_EXECUTION_FAILED"


# ----------------------------------------------------------------------------
# Test L: Missing telemetry handling
# ----------------------------------------------------------------------------
def test_L_missing_telemetry_handling():
    runtime = AdaptiveSRRuntime(video_id="sample")
    telemetry = runtime.process_chunk("0000")

    assert isinstance(telemetry, ChunkTelemetry)
    assert telemetry.identity.chunk_id == "0000"


# ----------------------------------------------------------------------------
# Test M: Deterministic/reproducible telemetry structure
# ----------------------------------------------------------------------------
def test_M_deterministic_telemetry_structure():
    def mock_det_handler(c):
        return {
            "bytes_received": 100000,
            "total_chunk_completion_time": 0.1,
            "download_transfer_time": 0.08,
            "sr_processing_time": 0.02,
            "edge_id": "edge_01",
            "cluster_id": "cluster_01",
            "request_id": "req_det",
        }

    runtime1 = AdaptiveSRRuntime(video_id="sample", execution_handler=mock_det_handler)
    runtime2 = AdaptiveSRRuntime(video_id="sample", execution_handler=mock_det_handler)

    t1 = runtime1.process_chunk("0000", observed_network_mbps=10.0, observed_rtt_seconds=0.03).to_dict()
    t2 = runtime2.process_chunk("0000", observed_network_mbps=10.0, observed_rtt_seconds=0.03).to_dict()

    for section in ["identity", "delivery_mode", "requested_configuration", "executed_configuration", "decision", "network", "resource"]:
        if section in ["identity"]:
            assert t1[section]["video_id"] == t2[section]["video_id"]
            assert t1[section]["chunk_id"] == t2[section]["chunk_id"]
        else:
            assert t1[section] == t2[section]


# ----------------------------------------------------------------------------
# Test N: Existing Steps 0–10 regression compatibility
# ----------------------------------------------------------------------------
def test_N_regression_compatibility():
    fps_adapter = FPSAdapter()
    fps_sig = fps_adapter.evaluate(
        source_fps=30.0,
        measured_latency_ms=25.0,
        model_id="tinysr",
        scale=2,
        device="cpu",
        base_representation_id="360p",
        network_rtt_ms=30.0,
    )
    assert fps_sig.realtime_feasible is True

    bit_adapter = BitrateAdapter()
    bit_sig = bit_adapter.evaluate(
        reference_representation_id="720p",
        candidate_representation_id="360p",
        reference_bitrate_bps=3000000.0,
        candidate_bitrate_bps=1500000.0,
        base_resolution="640x360",
        target_resolution="1280x720",
        model_id="tinysr",
        scale=2,
        device="cpu",
        psnr_db=34.0,
        ssim=0.92,
        vmaf=82.0,
    )
    assert bit_sig.quality_evaluable is True

    edge_eval = EdgeResourceEvaluator()
    state = EdgeResourceState(
        edge_id="edge_01",
        cpu_utilization_percent=20.0,
        gpu_utilization_percent=30.0,
        gpu_available=True,
        supported_devices=["cpu", "cuda"],
        supported_models=["tinysr"],
    )
    res_sig = edge_eval.evaluate_node(
        state=state,
        model_id="tinysr",
        scale=2,
        device="cpu",
        base_representation_id="360p",
        target_resolution="1280x720",
        source_fps=30.0,
    )
    assert res_sig.resource_feasible is True

    engine = FuzzyAdaptiveDecisionEngine()
    fuzzy_sig = engine.evaluate_candidates([fps_sig], [bit_sig], [res_sig])
    assert fuzzy_sig.decision in ["selected", "no_suitable_candidate"]

    runtime = AdaptiveSRRuntime(video_id="sample")
    t = runtime.process_chunk("0000")
    assert t.identity.chunk_id == "0000"


# ============================================================================
# Hardening Requirement Regression Tests
# ============================================================================

def test_quality_evidence_from_precomputed_data():
    q_store = QualityEvidenceStore([
        {
            "video_id": "sample",
            "chunk_id": "0000",
            "representation_id": "360p",
            "model_id": "tinysr",
            "scale": 2,
            "device": "cpu",
            "psnr": 38.5,
            "ssim": 0.96,
            "vmaf": 91.0,
            "quality_provenance": "step5.6_benchmark",
        }
    ])
    runtime = AdaptiveSRRuntime(video_id="sample", quality_store=q_store)
    telemetry = runtime.process_chunk("0000")

    assert telemetry.provenance.bitrate_signal_provenance["quality_evaluable"] is True


def test_missing_quality_evidence_is_never_fabricated():
    empty_q_store = QualityEvidenceStore(records=[])
    runtime = AdaptiveSRRuntime(video_id="sample", quality_store=empty_q_store)
    telemetry = runtime.process_chunk("0000")

    assert telemetry.provenance.bitrate_signal_provenance["quality_evaluable"] is False


def test_buffer_uses_total_client_completion_elapsed_time():
    runtime = AdaptiveSRRuntime(video_id="sample", initial_buffer_seconds=0.05)

    def timing_handler(c):
        time.sleep(0.1)
        return {
            "bytes_received": 100000,
            "total_chunk_completion_time": 0.99,  # Server-reported diagnostic timing
            "download_transfer_time": 0.90,
            "sr_processing_time": 0.09,
            "edge_id": "edge_01",
            "cluster_id": "cluster_01",
            "request_id": "req_time",
        }

    runtime.execution_handler = timing_handler
    telemetry = runtime.process_chunk("0000", chunk_duration=2.0)

    # Client elapsed time is strictly measured wall-clock monotonic time (~0.1s)
    assert telemetry.timing.client_elapsed_seconds < 0.3
    # Buffer math uses client_elapsed_seconds
    assert telemetry.buffer.stall_duration == pytest.approx(telemetry.timing.client_elapsed_seconds - 0.05, abs=0.05)


def test_fallback_native_representation():
    runtime = AdaptiveSRRuntime(video_id="sample", min_suitability_threshold=99.0, fallback_representation_id="480p")
    telemetry = runtime.process_chunk("0000")

    assert telemetry.delivery_mode == "native"
    assert telemetry.executed_configuration["representation_id"] == "480p"
    assert telemetry.decision.fallback_reason in ["no_suitable_candidate", "no_feasible_candidates"]


def test_sr_to_native_switch_counted():
    runtime = AdaptiveSRRuntime(video_id="sample", fallback_representation_id="360p")

    # Chunk 0: Selected SR
    t0 = runtime.process_chunk("0000", observed_network_mbps=20.0)

    # Chunk 1: Decision failure -> Native Fallback
    runtime.min_suitability_threshold = 99.0
    t1 = runtime.process_chunk("0001", observed_network_mbps=20.0)

    assert t0.delivery_mode == "sr"
    assert t1.delivery_mode == "native"
    assert runtime.state.configuration_switch_count == 1


def test_native_to_sr_switch_counted():
    runtime = AdaptiveSRRuntime(video_id="sample", min_suitability_threshold=99.0, fallback_representation_id="360p")

    # Chunk 0: Native Fallback
    t0 = runtime.process_chunk("0000")

    # Chunk 1: Selected SR
    runtime.min_suitability_threshold = 35.0
    t1 = runtime.process_chunk("0001", observed_network_mbps=20.0)

    assert t0.delivery_mode == "native"
    assert t1.delivery_mode == "sr"
    assert runtime.state.configuration_switch_count == 1


def test_failed_requested_sr_not_recorded_as_executed():
    runtime = AdaptiveSRRuntime(video_id="sample")

    selected_cfg = {
        "edge_id": "edge_01",
        "base_representation_id": "360p",
        "target_resolution": "720p",
        "model_id": "tinysr",
        "scale": 2,
        "device": "cpu",
    }

    def failing_handler(c):
        raise RuntimeError("Edge GPU out of memory")

    with patch.object(FuzzyAdaptiveDecisionEngine, "evaluate_candidates") as mock_eval:
        mock_eval.return_value = FuzzyDecisionSignal(
            decision="selected",
            selected_candidate=selected_cfg,
            selected_edge_id="edge_01",
            selected_representation_id="360p",
            target_resolution="720p",
            model_id="tinysr",
            scale=2,
            device="cpu",
            fuzzy_suitability=80.0,
            suitability_tier="high",
            min_suitability_threshold=35.0,
            candidate_evaluations=[],
            rejected_candidates=[],
            input_signal_provenance={},
            rule_inference_metadata={},
            warnings=[],
        )
        runtime.execution_handler = failing_handler
        telemetry = runtime.process_chunk("0000")

        assert telemetry.delivery_mode == "execution_failed"
        assert telemetry.requested_configuration is not None
        assert telemetry.executed_configuration is None


def test_edge_id_resolves_to_correct_http_endpoint():
    reg = EdgeRegistry({"edge_01": "http://edge01.local:8001", "edge_02": "http://edge02.local:8002"})
    assert reg.resolve("edge_01") == "http://edge01.local:8001"
    assert reg.resolve("edge_02") == "http://edge02.local:8002"

    executed_endpoint = None
    def capturing_handler(c):
        nonlocal executed_endpoint
        executed_endpoint = reg.resolve(c["edge_id"])
        return {
            "bytes_received": 100000,
            "total_chunk_completion_time": 0.1,
            "download_transfer_time": 0.08,
            "sr_processing_time": 0.02,
            "edge_id": c["edge_id"],
            "cluster_id": "cluster_01",
            "request_id": "req_edge",
        }

    runtime = AdaptiveSRRuntime(video_id="sample", edge_registry=reg, execution_handler=capturing_handler)
    runtime.process_chunk("0000")

    assert executed_endpoint == "http://edge01.local:8001"


def test_decision_eligible_preserved():
    runtime = AdaptiveSRRuntime(video_id="sample")
    telemetry = runtime.process_chunk("0000")

    assert "decision_eligible" in telemetry.provenance.fps_signal_provenance
    assert "decision_eligible" in telemetry.provenance.bitrate_signal_provenance
    assert telemetry.provenance.fps_signal_provenance["decision_eligible"] is True


def test_injected_test_conditions_explicitly_marked():
    # 1. Disabled Profile -> No injection
    disabled_profile = DynamicConditionProfile(enabled=False, network_bandwidth_map={"0000": 1.0})
    runtime_norm = AdaptiveSRRuntime(video_id="sample", dynamic_profile=disabled_profile)
    t_norm = runtime_norm.process_chunk("0000", observed_network_mbps=10.0)

    assert t_norm.is_injected_test_condition is False
    assert len(t_norm.injected_fields) == 0

    # 2. Enabled Profile -> Injected condition marked
    enabled_profile = DynamicConditionProfile(enabled=True, network_bandwidth_map={"0000": 1.0})
    runtime_inj = AdaptiveSRRuntime(video_id="sample", dynamic_profile=enabled_profile)
    t_inj = runtime_inj.process_chunk("0000", observed_network_mbps=10.0)

    assert t_inj.is_injected_test_condition is True
    assert "network_bandwidth" in t_inj.injected_fields
