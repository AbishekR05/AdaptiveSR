"""
tests.test_studio_app
======================
Integration and regression test suite for the AdaptiveSR Studio Web Demonstration Prototype.

Verifies:
1. Studio index HTML endpoint.
2. System status endpoint.
3. Video file upload and metadata extraction.
4. Sample video generator endpoint.
5. End-to-end AdaptiveSR processing flow:
   - Input video parsing
   - Steps 7–9 signals -> Step 10 Mamdani fuzzy decision -> Step 6 PyTorch TinySR inference
   - Actual upscaled video frame generation (640x360 -> 1280x720)
   - Video encoding and saving to disk
   - Accurate, un-fabricated telemetry reporting
6. Media file streaming endpoints.
"""

import os
import cv2
import numpy as np
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from adaptive_sr.services.studio.app import app, STORAGE_DIR, UPLOADS_DIR, OUTPUTS_DIR


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_mp4(tmp_path):
    """Create a temporary 640x360 30fps test video file."""
    filepath = tmp_path / "test_input_640x360.mp4"
    width, height, fps = 640, 360, 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))

    for i in range(15):  # 0.5s duration
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [i * 10, 50, 100]
        cv2.circle(frame, (100 + i * 5, 100), 20, (255, 200, 50), -1)
        out.write(frame)
    out.release()

    return filepath


def test_studio_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "AdaptiveSR Studio" in response.text
    assert "dual-view" in response.text.lower() or "comparison" in response.text.lower()


def test_system_status_endpoint(client):
    response = client.get("/api/system_status")
    assert response.status_code == 200
    data = response.json()
    assert "cuda_available" in data
    assert "tinysr" in data["available_models"]


def test_sample_video_endpoint(client):
    response = client.get("/api/sample")
    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == "sample_demo"
    assert data["width"] == 640
    assert data["height"] == 360
    assert data["fps"] == 30.0
    assert "url" in data


def test_video_upload_endpoint(client, sample_mp4):
    with open(sample_mp4, "rb") as f:
        response = client.post(
            "/api/upload",
            files={"file": ("test_input_640x360.mp4", f, "video/mp4")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "video_id" in data
    assert data["width"] == 640
    assert data["height"] == 360
    assert data["fps"] == 30.0
    assert data["duration_seconds"] > 0
    assert "url" in data


def test_end_to_end_adaptive_sr_studio_processing(client, sample_mp4):
    """
    Complete Studio Demonstration acceptance test flow:
    Upload 640x360 video -> Run AdaptiveSR -> TinySR x2 upscaling -> 1280x720 upscaled video output.
    """
    # Step 1: Upload input video
    with open(sample_mp4, "rb") as f:
        up_resp = client.post(
            "/api/upload",
            files={"file": ("input_640x360.mp4", f, "video/mp4")}
        )
    assert up_resp.status_code == 200
    up_data = up_resp.json()

    # Step 2: Request AdaptiveSR processing
    proc_resp = client.post(
        "/api/process",
        json={
            "video_id": up_data["video_id"],
            "stored_filename": up_data["stored_filename"],
            "min_suitability_threshold": 35.0,
            "fallback_representation_id": "360p"
        }
    )
    assert proc_resp.status_code == 200
    res = proc_resp.json()

    assert res["status"] == "success"

    # Input metadata checks
    in_meta = res["input_metadata"]
    assert in_meta["width"] == 640
    assert in_meta["height"] == 360

    # Output metadata checks (Genuine 2x SR upscaling: 640x360 -> 1280x720)
    out_meta = res["output_metadata"]
    assert out_meta["width"] == 1280
    assert out_meta["height"] == 720
    assert out_meta["resolution"] == "1280×720"

    # Decision checks
    dec = res["decision"]
    assert dec["delivery_mode"] in ["sr", "native"]
    if dec["delivery_mode"] == "sr":
        assert dec["model_id"] == "tinysr"
        assert dec["scale"] == 2

    # Telemetry checks (No fabricated values)
    telem = res["telemetry"]
    assert telem["client_elapsed_seconds"] > 0
    assert telem["processing_latency_ms"] > 0
    assert telem["inference_fps"] != "N/A" or dec["delivery_mode"] == "native"
    assert telem["buffer_seconds"] >= 0

    # Verify generated upscaled output file exists and can be fetched via /media
    output_url = out_meta["url"]
    media_resp = client.get(output_url)
    assert media_resp.status_code == 200
    assert len(media_resp.content) > 0


def test_media_streaming_endpoint(client, sample_mp4):
    with open(sample_mp4, "rb") as f:
        up_resp = client.post(
            "/api/upload",
            files={"file": ("stream_test.mp4", f, "video/mp4")}
        )
    up_data = up_resp.json()

    media_resp = client.get(up_data["url"])
    assert media_resp.status_code == 200
    assert media_resp.headers["content-type"] == "video/mp4"
