import os
import json
import numpy as np
import pytest
import cv2

from adaptive_sr.benchmarking.quality_eval import (
    apply_divisibility_crop,
    calculate_psnr_y,
    calculate_ssim_y,
    run_vmaf_on_chunk,
    run_quality_evaluation
)


def test_apply_divisibility_crop():
    # 15x15 shape is not divisible by 2
    frame = np.zeros((15, 15, 3), dtype=np.uint8)
    cropped, meta = apply_divisibility_crop(frame, scale=2)
    assert cropped.shape == (14, 14, 3)
    assert meta["gt_crop_applied"] is True
    assert meta["crop_y"] == 0
    assert meta["crop_x"] == 0

    # 16x16 shape is divisible by 2, no crop
    frame2 = np.zeros((16, 16, 3), dtype=np.uint8)
    cropped2, meta2 = apply_divisibility_crop(frame2, scale=2)
    assert cropped2.shape == (16, 16, 3)
    assert meta2["gt_crop_applied"] is False


def test_calculate_psnr_y():
    # Identical frames -> perfect reconstruction
    y1 = np.ones((64, 64), dtype=np.uint8) * 128
    y2 = np.ones((64, 64), dtype=np.uint8) * 128
    val, reason = calculate_psnr_y(y1, y2)
    assert val is None
    assert reason == "perfect_reconstruction"

    # Different frames
    y3 = np.ones((64, 64), dtype=np.uint8) * 130
    val_diff, reason_diff = calculate_psnr_y(y1, y3)
    assert val_diff is not None
    assert val_diff > 0
    assert reason_diff is None


def test_calculate_ssim_y():
    # Identical frames -> SSIM should be 1.0
    y1 = np.ones((64, 64), dtype=np.uint8) * 128
    y2 = np.ones((64, 64), dtype=np.uint8) * 128
    val, reason = calculate_ssim_y(y1, y2)
    assert val is not None
    assert pytest.approx(val, 0.01) == 1.0
    assert reason is None


def test_run_vmaf_on_chunk_mock():
    # Verify mocked run_vmaf_on_chunk returns correct schema
    f1 = np.zeros((64, 64, 3), dtype=np.uint8)
    f2 = np.zeros((64, 64, 3), dtype=np.uint8)
    mean, per_frame, err, reason = run_vmaf_on_chunk([f1], [f2], 64, 64)
    assert mean is not None
    assert len(per_frame) == 1
    assert err is False
    assert reason is None


def test_quality_evaluation_integration(tmp_path):
    # Setup paths
    output_dir = str(tmp_path / "results")
    manifest_path = "data/benchmarks/sr/manifests/layer_b_manifest.json"

    if not os.path.exists(manifest_path):
        pytest.skip("layer_b_manifest.json not generated, skipping integration test")

    res = run_quality_evaluation(
        manifest_path=manifest_path,
        output_dir=output_dir,
        device="cpu"
    )

    assert os.path.exists(res["frames_path"])
    assert os.path.exists(res["chunks_path"])
    assert os.path.exists(res["clips_path"])

    with open(res["clips_path"], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "metadata" in data
        assert "records" in data
        # PSNR should be > 25 dB for simulated bicubic upscaling
        for record in data["records"]:
            if record["psnr_mean"] is not None:
                assert record["psnr_mean"] > 25.0
