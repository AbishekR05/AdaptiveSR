"""
tests.test_benchmark_dataset
=============================
Unit and integration tests for Step 5.8 Machine-Readable Benchmark Dataset Builder.
"""

import os
import json
import tempfile
import pytest

from adaptive_sr.benchmarking.dataset_builder import (
    build_unified_dataset,
    export_csv_dataset,
    make_record_key,
    extract_step55_records,
    extract_step56_records,
    extract_step57_records,
)


def sample_step55_data():
    return [
        {
            "config": {
                "model_id": "tinysr",
                "scale": 2,
                "device": "cpu",
                "input_id": "synthetic_lowmotion_30fps",
                "cpu_config": {"cpu_ids": [1], "num_threads": 2},
                "gpu_device_id": None
            },
            "trial_latencies": [0.010, 0.011, 0.009],
            "latency_statistics": {
                "median_latency": 0.010, # 10ms (in seconds)
                "p95_latency": 0.011
            },
            "throughput_fps": 100.0,
            "decision_eligible": False,
            "eligibility_reason": "insufficient_sessions_count",
            "session_count": 1,
            "resource_summary": {"cpu_util_mean": 15.0}
        }
    ]


def sample_step56_data():
    return {
        "metadata": {"version": "1.0"},
        "records": [
            {
                "model_id": "tinysr",
                "scale": 2,
                "device": "cpu",
                "clip_id": "synthetic_lowmotion_30fps",
                "psnr_mean": 32.5,
                "ssim_mean": 0.91,
                "vmaf_mean": None,
                "vmaf_unavailable": True,
                "chunk_count": 1
            }
        ]
    }


def sample_step57_data():
    return [
        {
            "model_id": "tinysr",
            "scale": 2,
            "device": "cpu",
            "benchmark_video_id": "synthetic_lowmotion_30fps",
            "source_fps": 30.0,
            "frame_budget_ms": 33.333333333333336,
            "estimated_processing_fps": 100.0,
            "real_time_ratio": 3.3333333333333335,
            "real_time_feasible": True,
            "real_time_feasible_p95_exploratory": True,
            "p95_confidence": "exploratory",
            "caveats": ["Ineligibility reason: insufficient_sessions_count"]
        }
    ]


# Test 1: test_canonical_record_key
def test_canonical_record_key():
    key = make_record_key("tinysr", 2, "cuda:0", "clip_001")
    assert key == "tinysr_x2_cuda_0_clip_001"


# Test 2: test_full_join_correctness
def test_full_join_correctness():
    records = build_unified_dataset(
        step55_raw_data=sample_step55_data(),
        step56_raw_data=sample_step56_data(),
        step57_raw_data=sample_step57_data(),
    )
    assert len(records) == 1
    rec = records[0]

    assert rec["model_id"] == "tinysr"
    assert rec["scale"] == 2
    assert rec["device"] == "cpu"
    assert rec["input_id"] == "synthetic_lowmotion_30fps"

    # Step 5.5 fields
    b_inf = rec["benchmark_inference"]
    assert b_inf["data_present"] is True
    assert b_inf["latency_ms"] == pytest.approx(10.0)
    assert b_inf["throughput_fps"] == pytest.approx(100.0)
    assert b_inf["decision_eligible"] is False
    assert b_inf["eligibility_reason"] == "insufficient_sessions_count"
    assert b_inf["session_count"] == 1

    # Step 5.6 fields
    q_eval = rec["quality_evaluation"]
    assert q_eval["data_present"] is True
    assert q_eval["psnr_y"] == pytest.approx(32.5)
    assert q_eval["ssim_y"] == pytest.approx(0.91)
    assert q_eval["vmaf_mean"] is None
    assert q_eval["vmaf_unavailable"] is True

    # Step 5.7 fields
    r_feas = rec["realtime_feasibility"]
    assert r_feas["data_present"] is True
    assert r_feas["source_fps"] == pytest.approx(30.0)
    assert r_feas["frame_budget_ms"] == pytest.approx(33.333333333333336)
    assert r_feas["real_time_feasible"] is True


# Test 3: test_missing_data_policy_preserves_nulls
def test_missing_data_policy_preserves_nulls():
    # Only Step 5.5 present, Step 5.6 and Step 5.7 missing
    records = build_unified_dataset(
        step55_raw_data=sample_step55_data(),
        step56_raw_data=None,
        step57_raw_data=None,
    )
    assert len(records) == 1
    rec = records[0]

    assert rec["benchmark_inference"]["data_present"] is True
    assert rec["quality_evaluation"]["data_present"] is False
    assert rec["quality_evaluation"]["psnr_y"] is None
    assert rec["quality_evaluation"]["ssim_y"] is None
    assert rec["realtime_feasibility"]["data_present"] is False
    assert rec["realtime_feasibility"]["source_fps"] is None
    assert rec["realtime_feasibility"]["real_time_feasible"] is None


# Test 4: test_provenance_metadata
def test_provenance_metadata():
    records = build_unified_dataset(
        step55_file="data/benchmarks/sr/results/step55_results.json",
        step55_raw_data=sample_step55_data(),
    )
    assert len(records) == 1
    prov = records[0]["provenance"]
    assert prov["step55_source"] == "data/benchmarks/sr/results/step55_results.json"
    assert prov["step56_source"] is None
    assert "generated_timestamp" in prov


# Test 5: test_csv_export_compatibility
def test_csv_export_compatibility():
    records = build_unified_dataset(
        step55_raw_data=sample_step55_data(),
        step56_raw_data=sample_step56_data(),
        step57_raw_data=sample_step57_data(),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "test_dataset.csv")
        export_csv_dataset(records, csv_path)

        assert os.path.exists(csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2  # Header + 1 data row
        header = lines[0]
        assert "record_id" in header
        assert "psnr_y" in header
        assert "real_time_feasible" in header
