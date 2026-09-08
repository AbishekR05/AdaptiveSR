"""
tests.test_validation_report
=============================
Unit and integration tests for Step 5.9 Validation + Reproducibility Report Module.
"""

import os
import pytest

from adaptive_sr.benchmarking.validator import (
    validate_unified_record,
    validate_dataset,
    generate_validation_report,
)


def sample_valid_unified_record():
    return {
        "record_id": "tinysr_x2_cpu_synthetic_lowmotion_30fps",
        "model_id": "tinysr",
        "scale": 2,
        "device": "cpu",
        "input_id": "synthetic_lowmotion_30fps",
        "benchmark_inference": {
            "latency_ms": 10.0,
            "p95_latency_ms": 12.0,
            "throughput_fps": 100.0,
            "cpu_ids": [1],
            "num_threads": 2,
            "gpu_device_id": None,
            "resource_summary": {},
            "decision_eligible": False,
            "eligibility_reason": "insufficient_sessions_count",
            "session_count": 1,
            "data_present": True
        },
        "quality_evaluation": {
            "psnr_y": 32.5,
            "ssim_y": 0.91,
            "vmaf_mean": None,
            "vmaf_unavailable": True,
            "data_present": True
        },
        "realtime_feasibility": {
            "source_fps": 30.0,
            "frame_budget_ms": 33.333333333333336,
            "estimated_processing_fps": 100.0,
            "real_time_ratio": 3.3333333333333335,
            "real_time_feasible": True,
            "real_time_feasible_p95_exploratory": True,
            "p95_confidence": "exploratory",
            "caveats": ["Ineligibility reason: insufficient_sessions_count"],
            "data_present": True
        },
        "provenance": {
            "step55_source": "data/benchmarks/sr/results/step55_results.json",
            "step56_source": "data/benchmarks/sr/results/quality_clips.json",
            "step57_source": "data/benchmarks/sr/results/fps_feasibility.json",
            "dataset_version": "1.0",
            "generated_timestamp": "2026-09-08T22:17:00Z"
        }
    }


# Test 1: test_validate_single_valid_record
def test_validate_single_valid_record():
    rec = sample_valid_unified_record()
    res = validate_unified_record(rec)
    assert res["is_valid"] is True
    assert len(res["issues"]) == 0


# Test 2: test_detects_invalid_latency
def test_detects_invalid_latency():
    rec = sample_valid_unified_record()
    rec["benchmark_inference"]["latency_ms"] = -5.0
    res = validate_unified_record(rec)
    assert res["is_valid"] is False
    assert any("Invalid non-positive latency_ms" in issue for issue in res["issues"])


# Test 3: test_detects_inconsistent_frame_budget
def test_detects_inconsistent_frame_budget():
    rec = sample_valid_unified_record()
    rec["realtime_feasibility"]["frame_budget_ms"] = 999.0  # Incorrect for 30 FPS
    res = validate_unified_record(rec)
    assert res["is_valid"] is False
    assert any("frame_budget_ms" in issue for issue in res["issues"])


# Test 4: test_detects_ineligible_session_count
def test_detects_ineligible_session_count():
    rec = sample_valid_unified_record()
    rec["benchmark_inference"]["session_count"] = 1
    rec["benchmark_inference"]["decision_eligible"] = True  # Inconsistent: 1 session cannot be decision eligible
    res = validate_unified_record(rec)
    assert res["is_valid"] is False
    assert any("Decision eligible is True but session_count" in issue for issue in res["issues"])


# Test 5: test_dataset_level_validation
def test_dataset_level_validation():
    records = [sample_valid_unified_record()]
    summary = validate_dataset(records)
    assert summary["is_dataset_valid"] is True
    assert summary["total_records"] == 1
    assert summary["valid_records"] == 1


# Test 6: test_detects_duplicate_record_keys
def test_detects_duplicate_record_keys():
    rec1 = sample_valid_unified_record()
    rec2 = sample_valid_unified_record()
    summary = validate_dataset([rec1, rec2])
    assert summary["is_dataset_valid"] is False
    assert len(summary["duplicate_keys"]) == 1


# Test 7: test_report_generation
def test_report_generation():
    records = [sample_valid_unified_record()]
    summary = validate_dataset(records)
    report = generate_validation_report("test_path.json", summary, records)

    assert "Step 5.9 — Validation + Reproducibility Report" in report
    assert "READY TO FREEZE" in report
    assert "tinysr_x2_cpu_synthetic_lowmotion_30fps" in report
