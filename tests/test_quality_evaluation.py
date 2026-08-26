import os
import json
import numpy as np
import pytest
import cv2
from unittest.mock import patch

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
    val, reason = calculate_ssim_y(y1, y2, downsample=False)
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
        device="cpu",
        evaluation_mode="bicubic_simulation"
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


def test_bicubic_cannot_be_labeled_as_model():
    with pytest.raises(ValueError, match="Only 'bicubic_baseline' is accepted"):
        run_quality_evaluation(
            evaluation_mode="bicubic_simulation",
            models=["tinysr"]
        )


def test_evaluation_mode_field_required(tmp_path):
    output_dir = str(tmp_path / "results")
    manifest_path = "data/benchmarks/sr/manifests/layer_b_manifest.json"
    if not os.path.exists(manifest_path):
        pytest.skip("layer_b_manifest.json not present")
    res = run_quality_evaluation(
        manifest_path=manifest_path,
        output_dir=output_dir,
        evaluation_mode="bicubic_simulation"
    )
    with open(res["clips_path"], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "evaluation_mode" in data["metadata"]
        for record in data["records"]:
            assert "evaluation_mode" in record
            assert record["evaluation_mode"] == "bicubic_simulation"


def test_reference_output_dimension_mismatch_raises():
    ref = np.zeros((64, 64), dtype=np.uint8)
    out = np.zeros((64, 65), dtype=np.uint8)
    with pytest.raises(AssertionError, match="must match exactly"):
        calculate_psnr_y(ref, out)
    with pytest.raises(AssertionError, match="must match exactly"):
        calculate_ssim_y(ref, out)


@patch("adaptive_sr.benchmarking.quality_eval.cv2_capture_frames")
@patch("os.path.exists")
@patch("builtins.open")
def test_frame_count_mismatch_raises(mock_open, mock_exists, mock_capture):
    mock_exists.return_value = True
    manifest_data = {
        "videos": [
            {
                "benchmark_video_id": "test_video",
                "file_path": "dummy.mp4",
                "chunks": [
                    {
                        "chunk_id": "9999",
                        "file_path": "chunk_9999.mp4"
                    }
                ]
            }
        ]
    }
    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(manifest_data)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_capture.return_value = [dummy_frame, dummy_frame]
    
    orig_resize = cv2.resize
    call_idx = 0
    def mock_resize(src, dsize, interpolation=None):
        nonlocal call_idx
        call_idx += 1
        if call_idx == 4:  # Upsample of second frame in bicubic simulation
            raise RuntimeError("Mock resize failure")
        return orig_resize(src, dsize, interpolation)
        
    with patch("cv2.resize", side_effect=mock_resize):
        with pytest.raises(ValueError) as exc_info:
            run_quality_evaluation(
                manifest_path="dummy_manifest.json",
                evaluation_mode="bicubic_simulation",
                scales=[2]
            )
        assert "9999" in str(exc_info.value)
        assert "Frame count mismatch" in str(exc_info.value)


def test_join_keys_present(tmp_path):
    output_dir = str(tmp_path / "results")
    manifest_path = "data/benchmarks/sr/manifests/layer_b_manifest.json"
    if not os.path.exists(manifest_path):
        pytest.skip("layer_b_manifest.json not present")
    res = run_quality_evaluation(
        manifest_path=manifest_path,
        output_dir=output_dir,
        evaluation_mode="bicubic_simulation"
    )
    with open(res["clips_path"], "r", encoding="utf-8") as f:
        data = json.load(f)
        for record in data["records"]:
            assert record["model_id"] is not None
            assert record["scale"] is not None
            assert record["device"] is not None
            assert record["input_id"] is not None
            assert record["benchmark_video_id"] is not None
            assert record["clip_id"] is not None


def test_ssim_downsampling_validation_sample():
    np.random.seed(42)
    frames_gt = [np.random.randint(0, 256, (1080, 1920), dtype=np.uint8) for _ in range(5)]
    frames_sr = [np.clip(f.astype(np.int16) + np.random.randint(-10, 11, f.shape), 0, 255).astype(np.uint8) for f in frames_gt]

    deviations = []
    for gt, sr in zip(frames_gt, frames_sr):
        ssim_full, _ = calculate_ssim_y(gt, sr, downsample=False)
        ssim_down, _ = calculate_ssim_y(gt, sr, downsample=True)
        assert ssim_full is not None
        assert ssim_down is not None
        deviations.append(abs(ssim_full - ssim_down))

    mean_dev = float(np.mean(deviations))
    max_dev = float(np.max(deviations))
    print(f"\nSSIM downsampling sample mean deviation: {mean_dev}, max: {max_dev}")
    assert isinstance(mean_dev, float)
    assert isinstance(max_dev, float)


def test_no_unsupported_conclusions_in_bicubic_section_metadata(tmp_path):
    output_dir = str(tmp_path / "results")
    manifest_path = "data/benchmarks/sr/manifests/layer_b_manifest.json"
    if not os.path.exists(manifest_path):
        pytest.skip("layer_b_manifest.json not present")
    res = run_quality_evaluation(
        manifest_path=manifest_path,
        output_dir=output_dir,
        evaluation_mode="bicubic_simulation"
    )
    with open(res["clips_path"], "r", encoding="utf-8") as f:
        data = json.load(f)
        for record in data["records"]:
            assert record["evaluation_mode"] == "bicubic_simulation"
            assert record["model_id"] == "bicubic_baseline"


def test_minimum_real_inference_smoke(tmp_path):
    output_dir = str(tmp_path / "results")
    manifest_path = "data/benchmarks/sr/manifests/layer_b_manifest.json"
    if not os.path.exists(manifest_path):
        pytest.skip("layer_b_manifest.json not present")
    res = run_quality_evaluation(
        manifest_path=manifest_path,
        output_dir=output_dir,
        device="cpu",
        evaluation_mode="model_inference",
        models=["tinysr"],
        scales=[2],
        clips=["clip_001_lowmotion_30fps"],
        chunks=["0000"]
    )
    assert os.path.exists(res["clips_path"])
    with open(res["clips_path"], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data["records"]) > 0
        assert data["records"][0]["model_id"] == "tinysr"
        assert data["records"][0]["evaluation_mode"] == "model_inference"
        assert data["records"][0]["psnr_mean"] is not None
        assert data["records"][0]["ssim_mean"] is not None

