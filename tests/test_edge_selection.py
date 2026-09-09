"""
tests.test_edge_selection
=========================
Step 9 — Edge Selection & Resource Allocation Layer Unit and Integration Tests.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch
import requests
from fastapi.testclient import TestClient

from adaptive_sr.adaptation.edge_evaluator import (
    EdgeResourceEvaluator,
    EdgeResourceState,
    EdgeResourceSignal,
)
from adaptive_sr.services.edge.app import app as edge_app


# Test 1: GPU-capable feasible Edge node
def test_gpu_capable_feasible_edge():
    state = EdgeResourceState(
        edge_id="edge_gpu_01",
        gpu_available=True,
        gpu_device_name="NVIDIA RTX 5060",
        gpu_memory_total_bytes=8 * 1024**3,
        gpu_memory_free_bytes=6 * 1024**3,
        gpu_utilization_percent=25.0,
        cpu_utilization_percent=30.0,
        supported_devices=["cpu", "cuda"],
        supported_models=["tinysr", "real_esrgan"],
        cloud_edge_rtt_ms=12.5,
    )
    sig = EdgeResourceEvaluator.evaluate_node(
        state=state,
        model_id="tinysr",
        scale=2,
        device="cuda",
        required_gpu_memory_bytes=1 * 1024**3,
    )
    assert sig.edge_id == "edge_gpu_01"
    assert sig.resource_feasible is True
    assert sig.feasibility_status == "feasible"
    assert sig.hardware_capability["gpu_available"] is True


# Test 2: CPU-only Edge node handling CPU request
def test_cpu_only_edge_cpu_request():
    state = EdgeResourceState(
        edge_id="edge_cpu_01",
        gpu_available=False,
        cpu_utilization_percent=45.0,
        supported_devices=["cpu"],
        supported_models=["tinysr"],
    )
    sig = EdgeResourceEvaluator.evaluate_node(state=state, device="cpu")
    assert sig.resource_feasible is True
    assert sig.feasibility_status == "feasible"


# Test 3: Unavailable requested GPU (CUDA requested on CPU-only node -> NO silent fallback)
def test_unavailable_requested_gpu():
    state = EdgeResourceState(
        edge_id="edge_cpu_only",
        gpu_available=False,
        supported_devices=["cpu"],
    )
    sig = EdgeResourceEvaluator.evaluate_node(state=state, device="cuda")
    assert sig.resource_feasible is False
    assert sig.feasibility_status == "infeasible"
    assert any("no silent CPU fallback allowed" in w for w in sig.warnings)


# Test 4: Insufficient GPU memory
def test_insufficient_gpu_memory():
    state = EdgeResourceState(
        edge_id="edge_vram_low",
        gpu_available=True,
        gpu_memory_free_bytes=500 * 1024**2,  # 500 MB free
        supported_devices=["cpu", "cuda"],
    )
    sig = EdgeResourceEvaluator.evaluate_node(
        state=state,
        device="cuda",
        required_gpu_memory_bytes=2 * 1024**3,  # 2 GB required
    )
    assert sig.resource_feasible is False
    assert sig.feasibility_status == "infeasible"
    assert any("Insufficient free GPU memory" in w for w in sig.warnings)


# Test 5: Overloaded Edge node (CPU > 95%)
def test_overloaded_edge_node():
    state = EdgeResourceState(
        edge_id="edge_busy",
        cpu_utilization_percent=98.5,
    )
    sig = EdgeResourceEvaluator.evaluate_node(state=state, device="cpu")
    assert sig.resource_feasible is False
    assert sig.feasibility_status == "infeasible"
    assert any("CPU on Edge node 'edge_busy' is overloaded" in w for w in sig.warnings)


# Test 6: Missing resource telemetry handling
def test_missing_resource_telemetry():
    state = EdgeResourceState(
        edge_id="edge_no_telemetry",
        cpu_utilization_percent=None,
        gpu_utilization_percent=None,
    )
    sig = EdgeResourceEvaluator.evaluate_node(state=state, device="cpu")
    assert sig.resource_availability["cpu_utilization_percent"] is None
    assert sig.resource_availability["gpu_utilization_percent"] is None


# Test 7: Missing network telemetry handling
def test_missing_network_telemetry():
    state = EdgeResourceState(
        edge_id="edge_no_net",
        cloud_edge_rtt_ms=None,
    )
    sig = EdgeResourceEvaluator.evaluate_node(state=state, device="cpu")
    assert sig.network_feasible is None
    assert sig.network_telemetry["cloud_edge_rtt_ms"] is None
    assert any("Network telemetry" in w for w in sig.warnings)


# Test 8: Multiple Edge candidates evaluation & sorting
def test_multiple_edge_candidates():
    candidates = [
        EdgeResourceState(edge_id="edge_busy", cpu_utilization_percent=98.0),
        EdgeResourceState(edge_id="edge_idle", cpu_utilization_percent=15.0),
        EdgeResourceState(edge_id="edge_medium", cpu_utilization_percent=45.0),
    ]
    signals = EdgeResourceEvaluator.evaluate_candidates(candidates, device="cpu")
    assert len(signals) == 3
    # Feasible candidates first (idle: 15%, medium: 45%), infeasible candidate last (busy: 98%)
    assert signals[0].edge_id == "edge_idle"
    assert signals[0].resource_feasible is True
    assert signals[1].edge_id == "edge_medium"
    assert signals[1].resource_feasible is True
    assert signals[2].edge_id == "edge_busy"
    assert signals[2].resource_feasible is False


# Test 9: Feasible vs infeasible candidates distinction
def test_feasible_vs_infeasible_candidates():
    state_feas = EdgeResourceState(edge_id="edge_ok", cpu_utilization_percent=30.0)
    state_infeas = EdgeResourceState(edge_id="edge_bad", cpu_utilization_percent=99.0)

    sig_feas = EdgeResourceEvaluator.evaluate_node(state_feas)
    sig_infeas = EdgeResourceEvaluator.evaluate_node(state_infeas)

    assert sig_feas.feasibility_status == "feasible"
    assert sig_infeas.feasibility_status == "infeasible"


# Test 10: RTT is not treated as end-to-end streaming latency
def test_rtt_not_treated_as_end_to_end_latency():
    state = EdgeResourceState(edge_id="edge_rtt", cloud_edge_rtt_ms=8.5)
    sig = EdgeResourceEvaluator.evaluate_node(state=state)
    assert sig.network_telemetry["cloud_edge_rtt_ms"] == 8.5
    assert any("Cloud RTT does not represent end-to-end streaming latency" in w for w in sig.warnings)


# Test 11: No silent CPU fallback confirmation
def test_no_silent_cpu_fallback():
    state = EdgeResourceState(edge_id="edge_no_gpu", gpu_available=False, supported_devices=["cpu"])
    sig = EdgeResourceEvaluator.evaluate_node(state=state, device="cuda")
    assert sig.device == "cuda"  # device selection is NOT changed to cpu
    assert sig.resource_feasible is False
    assert sig.feasibility_status == "infeasible"


# Test 12: Real local integration test with Step 6 Edge runtime endpoints & telemetry
def test_real_local_step6_integration():
    client = TestClient(edge_app)

    # 1. Edge health endpoint
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert "edge_id" in health_data

    # 2. Mock Cloud Origin for chunk request
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:, :] = (100, 150, 200)
    _, png_data = cv2.imencode(".png", img)

    def mock_get(url, *args, **kwargs):
        resp = requests.Response()
        if "/health" in url:
            resp.status_code = 200
            resp._content = b'{"status": "ok"}'
            resp.headers["content-type"] = "application/json"
        elif "/chunks/" in url:
            resp.status_code = 200
            resp._content = png_data.tobytes()
            resp.headers["content-type"] = "image/png"
        else:
            resp.status_code = 404
            resp._content = b'{"detail": "Not found"}'
        return resp

    with patch("requests.get", side_effect=mock_get):
        url = "/videos/test_vid/chunks/chunk_step9_001?representation_id=360p&sr_requested=true&model_id=tinysr&scale=2&device=cpu"
        response = client.get(url)
        assert response.status_code == 200

        # Evaluate response headers into EdgeResourceSignal
        sig = EdgeResourceEvaluator.evaluate_from_step6_telemetry(
            telemetry=dict(response.headers),
            edge_id=health_data.get("edge_id", "edge_01"),
            cluster_id=health_data.get("cluster_id", "cluster_01"),
        )

        assert isinstance(sig, EdgeResourceSignal)
        assert sig.edge_id == health_data.get("edge_id", "edge_01")
        assert sig.model_id == "tinysr"
        assert sig.scale == 2
        assert sig.device == "cpu"
        assert sig.resource_feasible is True
        assert sig.measurement_provenance == "step6_edge_telemetry"
