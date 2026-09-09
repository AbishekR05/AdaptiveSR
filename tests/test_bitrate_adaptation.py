"""
tests.test_bitrate_adaptation
==============================
Step 8 — Bitrate / Quality Adaptation Layer Unit and Integration Tests.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch
import requests
from fastapi.testclient import TestClient

from adaptive_sr.adaptation.bitrate_adapter import (
    BitrateAdapter,
    BitrateAdaptationSignal,
)
from adaptive_sr.services.edge.app import app as edge_app


# Test 1: Bitrate saving calculation (3.0 Mbps ref vs 1.0 Mbps cand -> 66.67%)
def test_bitrate_saving_calculation():
    saving = BitrateAdapter.calculate_bitrate_saving(3000000.0, 1000000.0)
    assert saving == pytest.approx(66.6667, abs=1e-3)

    sig = BitrateAdapter.evaluate(
        reference_representation_id="720p",
        candidate_representation_id="360p",
        reference_bitrate_bps=3000000.0,
        candidate_bitrate_bps=1000000.0,
    )
    assert sig.bitrate_saving_percent == pytest.approx(66.6667, abs=1e-3)


# Test 2: Equal bitrate (1.0 Mbps vs 1.0 Mbps -> 0% saving)
def test_equal_bitrate():
    sig = BitrateAdapter.evaluate(
        reference_representation_id="360p",
        candidate_representation_id="360p_alt",
        reference_bitrate_bps=1000000.0,
        candidate_bitrate_bps=1000000.0,
    )
    assert sig.bitrate_saving_percent == 0.0
    assert any("equal to reference" in w for w in sig.warnings)


# Test 3: Candidate bitrate greater than reference (1.0 Mbps ref vs 1.5 Mbps cand -> -50%)
def test_candidate_bitrate_greater_than_reference():
    sig = BitrateAdapter.evaluate(
        reference_representation_id="360p",
        candidate_representation_id="360p_hq",
        reference_bitrate_bps=1000000.0,
        candidate_bitrate_bps=1500000.0,
    )
    assert sig.bitrate_saving_percent == -50.0
    assert any("higher than reference" in w for w in sig.warnings)


# Test 4: Zero or missing bitrate
def test_zero_or_missing_bitrate():
    sig = BitrateAdapter.evaluate(
        reference_representation_id="720p",
        candidate_representation_id="360p",
        reference_bitrate_bps=0.0,
        candidate_bitrate_bps=1000000.0,
    )
    assert sig.bitrate_saving_percent is None
    assert any("Reference bitrate is zero or missing" in w for w in sig.warnings)


# Test 5: Available quality metrics (PSNR, SSIM populated)
def test_available_quality_metrics():
    sig = BitrateAdapter.evaluate(
        reference_representation_id="720p",
        candidate_representation_id="360p",
        reference_bitrate_bps=3000000.0,
        candidate_bitrate_bps=1000000.0,
        psnr_db=32.45,
        ssim=0.912,
        quality_provenance="model_inference",
    )
    assert sig.psnr_db == 32.45
    assert sig.ssim == 0.912
    assert sig.quality_evaluable is True
    assert sig.quality_provenance == "model_inference"


# Test 6: Unavailable VMAF metric handling
def test_unavailable_vmaf_metric():
    sig = BitrateAdapter.evaluate(
        reference_representation_id="720p",
        candidate_representation_id="360p",
        reference_bitrate_bps=3000000.0,
        candidate_bitrate_bps=1000000.0,
        psnr_db=30.0,
        ssim=0.88,
        vmaf=None,
    )
    assert sig.vmaf is None
    assert any("VMAF metric unavailable" in w for w in sig.warnings)


# Test 7: Model inference vs bicubic simulation provenance
def test_model_inference_vs_bicubic_provenance():
    sig_model = BitrateAdapter.evaluate(
        reference_representation_id="720p",
        candidate_representation_id="360p",
        reference_bitrate_bps=3000000.0,
        candidate_bitrate_bps=1000000.0,
        quality_provenance="model_inference",
    )
    sig_bicubic = BitrateAdapter.evaluate(
        reference_representation_id="720p",
        candidate_representation_id="360p",
        reference_bitrate_bps=3000000.0,
        candidate_bitrate_bps=1000000.0,
        quality_provenance="bicubic_simulation",
    )
    assert sig_model.quality_provenance == "model_inference"
    assert sig_bicubic.quality_provenance == "bicubic_simulation"


# Test 8: Base/target representation identity preservation
def test_base_target_representation_identity():
    sig = BitrateAdapter.evaluate(
        reference_representation_id="1080p",
        candidate_representation_id="480p",
        reference_bitrate_bps=6000000.0,
        candidate_bitrate_bps=1800000.0,
        base_resolution="854x480",
        target_resolution="1920x1080",
        model_id="real_esrgan",
        scale=2,
    )
    assert sig.reference_representation_id == "1080p"
    assert sig.candidate_representation_id == "480p"
    assert sig.base_resolution == "854x480"
    assert sig.target_resolution == "1920x1080"
    assert sig.model_id == "real_esrgan"
    assert sig.quality_equivalent_to_native is False
    assert any("LOWER BITRATE + SR DOES NOT AUTOMATICALLY MEAN EQUIVALENT QUALITY" in w for w in sig.warnings)


# Test 9: Multiple representations evaluation
def test_multiple_representations():
    ref = {"representation_id": "720p", "bitrate": 3000000.0, "resolution": "1280x720"}
    cands = [
        {"representation_id": "360p", "bitrate": 1000000.0, "resolution": "640x360"},
        {"representation_id": "480p", "bitrate": 1800000.0, "resolution": "854x480"},
    ]
    signals = []
    for cand in cands:
        sig = BitrateAdapter.evaluate_from_manifest_and_quality(
            reference_rep=ref,
            candidate_rep=cand,
            quality_record={"psnr_db": 31.0, "ssim": 0.89},
        )
        signals.append(sig)

    assert len(signals) == 2
    assert signals[0].candidate_representation_id == "360p"
    assert signals[0].bitrate_saving_percent == pytest.approx(66.6667, abs=1e-3)
    assert signals[1].candidate_representation_id == "480p"
    assert signals[1].bitrate_saving_percent == pytest.approx(40.0, abs=1e-3)


# Test 10: Missing quality measurements handling
def test_missing_quality_measurements():
    sig = BitrateAdapter.evaluate(
        reference_representation_id="720p",
        candidate_representation_id="360p",
        reference_bitrate_bps=3000000.0,
        candidate_bitrate_bps=1000000.0,
        psnr_db=None,
        ssim=None,
        vmaf=None,
    )
    assert sig.quality_evaluable is False
    assert any("No genuine quality metrics" in w for w in sig.warnings)


# Test 11: Decision eligibility distinction
def test_decision_eligibility_distinction():
    sig = BitrateAdapter.evaluate(
        reference_representation_id="720p",
        candidate_representation_id="360p",
        reference_bitrate_bps=3000000.0,
        candidate_bitrate_bps=1000000.0,
        decision_eligible=False,
    )
    assert sig.decision_eligible is False
    assert any("not decision-eligible" in w for w in sig.warnings)


# Test 12: Real local integration test with Step 6 Edge runtime and manifest metadata
def test_real_local_step6_integration():
    client = TestClient(edge_app)

    # Mock Cloud Origin returning synthetic image chunk
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
        url = "/videos/test_vid/chunks/chunk_step8_001?representation_id=360p&sr_requested=true&model_id=tinysr&scale=2&device=cpu"
        response = client.get(url)
        assert response.status_code == 200
        assert response.headers.get("X-SR-Status") == "executed"

        # Manifest representations
        ref_rep = {"representation_id": "720p", "bitrate": 3000000.0, "resolution": "1280x720"}
        cand_rep = {"representation_id": "360p", "bitrate": 1000000.0, "resolution": "640x360"}

        # Ingest Step 6 headers + manifest into BitrateAdapter
        signal = BitrateAdapter.evaluate_from_manifest_and_quality(
            reference_rep=ref_rep,
            candidate_rep=cand_rep,
            quality_record={"psnr_db": 33.2, "ssim": 0.92, "evaluation_mode": "model_inference"},
            model_id=response.headers.get("X-SR-Model", "tinysr"),
            scale=int(response.headers.get("X-SR-Scale", "2")),
            device=response.headers.get("X-SR-Device", "cpu"),
        )

        assert isinstance(signal, BitrateAdaptationSignal)
        assert signal.reference_representation_id == "720p"
        assert signal.candidate_representation_id == "360p"
        assert signal.bitrate_saving_percent == pytest.approx(66.6667, abs=1e-3)
        assert signal.psnr_db == 33.2
        assert signal.ssim == 0.92
        assert signal.quality_evaluable is True
        assert signal.quality_provenance == "model_inference"
