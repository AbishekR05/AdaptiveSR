"""
tests.test_remote_sr
====================
Focused unit and integration tests for Step 6 — Remote SR Inference at the Edge.
"""

import os
import io
import cv2
import json
import pytest
import numpy as np
from fastapi.testclient import TestClient

from adaptive_sr.services.edge.app import app, cache


@pytest.fixture
def test_client():
    return TestClient(app)


from unittest.mock import patch, MagicMock
import requests

@pytest.fixture
def mock_cloud_origin():
    """Mocks requests to Cloud origin using unittest.mock."""
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
        yield "http://localhost:8000"


# Test 1: test_non_sr_request_preserves_baseline_behavior
def test_non_sr_request_preserves_baseline_behavior(test_client, mock_cloud_origin):
    response = test_client.get("/videos/test_vid/chunks/chunk_001?representation_id=480p")
    assert response.status_code == 200
    assert response.headers.get("X-SR-Requested") == "False"
    assert response.headers.get("X-SR-Status") == "not_requested"
    assert "X-Request-ID" in response.headers


# Test 2: test_sr_request_invokes_tinysr_adapter_cpu
def test_sr_request_invokes_tinysr_adapter_cpu(test_client, mock_cloud_origin):
    url = "/videos/test_vid/chunks/chunk_001?representation_id=480p&sr_requested=true&model_id=tinysr&scale=2&device=cpu"
    response = test_client.get(url)
    assert response.status_code == 200
    assert response.headers.get("X-SR-Requested") == "True"
    assert response.headers.get("X-SR-Status") == "executed"
    assert response.headers.get("X-SR-Model") == "tinysr"
    assert response.headers.get("X-SR-Scale") == "2"
    assert response.headers.get("X-SR-Device") == "cpu"
    
    # Verify spatial upscaling (240x320 * scale 2 -> 480x640)
    out_img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    if out_img is None:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_f:
            tmp_f.write(response.content)
            tmp_path = tmp_f.name
        cap = cv2.VideoCapture(tmp_path)
        ret, out_img = cap.read()
        cap.release()
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    assert out_img is not None
    assert out_img.shape[:2] == (480, 640)


# Test 3: test_sr_request_cuda_device_handling
def test_sr_request_cuda_device_handling(test_client, mock_cloud_origin):
    import torch
    if not torch.cuda.is_available():
        # Requirement 6: Must fail with 500 if CUDA requested but unavailable, NO silent fallback
        url = "/videos/test_vid/chunks/chunk_001?representation_id=480p&sr_requested=true&model_id=tinysr&scale=2&device=cuda"
        response = test_client.get(url)
        assert response.status_code == 500
        assert "CUDA requested for Remote SR but CUDA is unavailable" in response.json()["detail"]
    else:
        url = "/videos/test_vid/chunks/chunk_001?representation_id=480p&sr_requested=true&model_id=tinysr&scale=2&device=cuda"
        response = test_client.get(url)
        assert response.status_code == 200
        assert response.headers.get("X-SR-Status") == "executed"
        assert response.headers.get("X-SR-Device") == "cuda"


# Test 4: test_invalid_sr_model_reports_failure
def test_invalid_sr_model_reports_failure(test_client, mock_cloud_origin):
    url = "/videos/test_vid/chunks/chunk_001?representation_id=480p&sr_requested=true&model_id=invalid_model_xyz"
    response = test_client.get(url)
    assert response.status_code == 400
    assert "Invalid or unavailable SR model" in response.json()["detail"]


# Test 5: test_sr_cache_hit_behavior
def test_sr_cache_hit_behavior(test_client, mock_cloud_origin):
    url = "/videos/test_vid/chunks/chunk_001?representation_id=480p&sr_requested=true&model_id=tinysr&scale=2&device=cpu"
    # First request: computes and caches SR chunk
    r1 = test_client.get(url)
    assert r1.status_code == 200

    # Second request: served from cache
    r2 = test_client.get(url)
    assert r2.status_code == 200
    assert r2.headers.get("X-SR-Status") == "executed"
