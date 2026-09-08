"""
tests.test_fps_adaptation
=========================
Step 7 — FPS Adaptation & Feasibility Layer Unit and Integration Tests.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch
import requests
from fastapi.testclient import TestClient

from adaptive_sr.adaptation.fps_adapter import (
    FPSAdapter,
    FPSAdaptationSignal,
    classify_realtime_ratio,
)
from adaptive_sr.services.edge.app import app as edge_app


# Test 1: 30 FPS with latency below budget
def test_30fps_latency_below_budget():
    signal = FPSAdapter.evaluate(source_fps=30.0, measured_latency_ms=20.0)
    assert signal.source_fps == 30.0
    assert signal.frame_budget_ms == pytest.approx(33.3333, abs=1e-3)
    assert signal.measured_latency_ms == 20.0
    assert signal.estimated_processing_fps == 50.0
    assert signal.real_time_ratio == pytest.approx(1.6667, abs=1e-3)
    assert signal.realtime_feasible is True
    assert signal.adaptation_tier == "realtime"


# Test 2: 30 FPS with latency above budget
def test_30fps_latency_above_budget():
    signal = FPSAdapter.evaluate(source_fps=30.0, measured_latency_ms=50.0)
    assert signal.source_fps == 30.0
    assert signal.frame_budget_ms == pytest.approx(33.3333, abs=1e-3)
    assert signal.measured_latency_ms == 50.0
    assert signal.estimated_processing_fps == 20.0
    assert signal.real_time_ratio == pytest.approx(0.6667, abs=1e-3)
    assert signal.realtime_feasible is False
    assert signal.adaptation_tier == "below_realtime"


# Test 3: Exact budget boundary
def test_exact_budget_boundary():
    budget_ms = 1000.0 / 30.0
    signal = FPSAdapter.evaluate(source_fps=30.0, measured_latency_ms=budget_ms)
    assert signal.realtime_feasible is True
    assert signal.real_time_ratio == pytest.approx(1.0, abs=1e-3)
    assert signal.adaptation_tier == "realtime"


# Test 4: Invalid/zero FPS raises ValueError
def test_invalid_zero_fps():
    with pytest.raises(ValueError, match="Invalid source_fps"):
        FPSAdapter.evaluate(source_fps=0.0, measured_latency_ms=20.0)

    with pytest.raises(ValueError, match="Invalid source_fps"):
        FPSAdapter.evaluate(source_fps=-10.0, measured_latency_ms=20.0)


# Test 5: Invalid/zero latency raises ValueError
def test_invalid_zero_latency():
    with pytest.raises(ValueError, match="Invalid measured_latency_ms"):
        FPSAdapter.evaluate(source_fps=30.0, measured_latency_ms=0.0)

    with pytest.raises(ValueError, match="Invalid measured_latency_ms"):
        FPSAdapter.evaluate(source_fps=30.0, measured_latency_ms=-5.0)


# Test 6: Missing measurements handling
def test_missing_measurements():
    with pytest.raises(ValueError):
        FPSAdapter.evaluate_from_step6_telemetry({})

    with pytest.raises(ValueError):
        FPSAdapter.evaluate_from_step5_record({"config": {}})


# Test 7: Multiple models, scales, and devices
def test_multiple_models_scales_devices():
    configs = [
        {"model_id": "tinysr", "scale": 2, "device": "cpu", "latency_ms": 15.0},
        {"model_id": "tinysr", "scale": 2, "device": "cuda", "latency_ms": 5.0},
        {"model_id": "real_esrgan", "scale": 2, "device": "cuda", "latency_ms": 40.0},
        {"model_id": "real_esrgan", "scale": 4, "device": "cpu", "latency_ms": 120.0},
    ]
    signals = []
    for cfg in configs:
        sig = FPSAdapter.evaluate(
            source_fps=60.0,
            measured_latency_ms=cfg["latency_ms"],
            model_id=cfg["model_id"],
            scale=cfg["scale"],
            device=cfg["device"],
        )
        signals.append(sig)

    assert len(signals) == 4
    # 60 FPS frame budget is 16.67ms
    assert signals[0].realtime_feasible is True   # 15.0ms <= 16.67ms
    assert signals[1].realtime_feasible is True   # 5.0ms <= 16.67ms
    assert signals[2].realtime_feasible is False  # 40.0ms > 16.67ms
    assert signals[3].adaptation_tier == "severely_below_realtime" # 120ms vs 16.67ms -> ratio 0.138


# Test 8: Decision eligibility distinction
def test_decision_eligibility_distinction():
    # Feasible but NOT decision eligible (e.g. high variance / single session)
    rec = {
        "config": {"model_id": "tinysr", "scale": 2, "device": "cpu", "input_id": "360p"},
        "latency_ms": 20.0,
        "decision_eligible": False,
    }
    sig = FPSAdapter.evaluate_from_step5_record(rec, source_fps=30.0)
    assert sig.realtime_feasible is True
    assert sig.decision_eligible is False
    assert any("not decision-eligible" in w for w in sig.warnings)


# Test 9: Classification thresholds
def test_classification_thresholds():
    assert classify_realtime_ratio(1.5) == "realtime"
    assert classify_realtime_ratio(1.0) == "realtime"
    assert classify_realtime_ratio(0.85) == "near_realtime"
    assert classify_realtime_ratio(0.75) == "near_realtime"
    assert classify_realtime_ratio(0.60) == "below_realtime"
    assert classify_realtime_ratio(0.50) == "below_realtime"
    assert classify_realtime_ratio(0.40) == "severely_below_realtime"
    assert classify_realtime_ratio(0.0) == "invalid"
    assert classify_realtime_ratio(-1.0) == "invalid"


# Test 10: Real local integration with Step 6 Edge runtime and actual TinySR inference
def test_real_local_step6_integration():
    client = TestClient(edge_app)

    # Mock Cloud Origin returning a synthetic image chunk
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
        url = "/videos/test_vid/chunks/chunk_step7_001?representation_id=480p&sr_requested=true&model_id=tinysr&scale=2&device=cpu"
        response = client.get(url)
        assert response.status_code == 200
        assert response.headers.get("X-SR-Status") == "executed"
        assert "X-SR-Processing-Time" in response.headers

        # Ingest response headers into FPSAdapter
        signal = FPSAdapter.evaluate_from_step6_telemetry(dict(response.headers), source_fps=30.0)

        assert isinstance(signal, FPSAdaptationSignal)
        assert signal.source_fps == 30.0
        assert signal.measured_latency_ms > 0.0
        assert signal.model_id == "tinysr"
        assert signal.scale == 2
        assert signal.device == "cpu"
        assert signal.measurement_provenance == "step6_edge_telemetry"
        assert signal.adaptation_tier in ["realtime", "near_realtime", "below_realtime", "severely_below_realtime"]
        assert signal.end_to_end_streaming_feasible is None
        assert signal.end_to_end_status == "not_evaluated"


# Test 11: realtime_feasible=True while end_to_end_streaming_feasible=None (not_evaluated)
def test_realtime_feasible_true_while_end_to_end_streaming_feasible_none():
    signal = FPSAdapter.evaluate(source_fps=30.0, measured_latency_ms=20.0, decision_eligible=True)
    assert signal.realtime_feasible is True
    assert signal.decision_eligible is True
    assert signal.end_to_end_streaming_feasible is None
    assert signal.end_to_end_status == "not_evaluated"
    assert any("End-to-end streaming feasibility not evaluated" in w for w in signal.warnings)


# Test 12: Cloud RTT does not automatically imply end-to-end streaming feasibility
def test_cloud_rtt_does_not_imply_end_to_end_feasibility():
    signal = FPSAdapter.evaluate(source_fps=30.0, measured_latency_ms=20.0, network_rtt_ms=5.0)
    assert signal.realtime_feasible is True
    assert signal.end_to_end_streaming_feasible is None
    assert signal.end_to_end_status == "not_evaluated"
    assert any("Cloud RTT does not substitute" in w for w in signal.warnings)

