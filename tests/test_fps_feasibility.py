"""
tests.test_fps_feasibility
==========================
Unit and integration tests for Step 5.7 FPS / Real-Time Feasibility Analysis.
"""

import pytest
import numpy as np
from typing import Dict, Any

from adaptive_sr.benchmarking.fps_analysis import (
    analyze_record,
    parse_device_id,
    MODEL_LIMITS
)
from adaptive_sr.benchmarking.harness import (
    InferenceBenchmarkHarness,
    BenchmarkConfig,
    CPUExecutionConfig
)


def create_base_config() -> Dict[str, Any]:
    return {
        "model_id": "tinysr",
        "scale": 2,
        "device": "cpu",
        "input_id": "synthetic_lowmotion_30fps",
        "cpu_config": {
            "cpu_ids": [1],
            "num_threads": 2,
            "exclude_cpu_ids": []
        }
    }


def create_base_record(config: Dict[str, Any], trial_latencies: list) -> Dict[str, Any]:
    return {
        "benchmark_id": "test_bench",
        "timestamp": "2026-08-26T00:00:00Z",
        "config": config,
        "trial_latencies": trial_latencies,
        "latency_statistics": {
            "count": len(trial_latencies),
            "mean": float(np.mean(trial_latencies)),
            "median": float(np.median(trial_latencies)),
            "min": float(np.min(trial_latencies)),
            "max": float(np.max(trial_latencies)),
            "std_dev": float(np.std(trial_latencies)),
            "p95": float(np.percentile(trial_latencies, 95)),
            "p95_sample_count": len(trial_latencies),
            "p95_confidence_note": "Exploratory fallback",
            "p95_confidence": "exploratory",
            "p95_min_recommended_n": 20,
            "min_latency": float(np.min(trial_latencies)),
            "median_latency": float(np.median(trial_latencies)),
            "mean_latency": float(np.mean(trial_latencies)),
            "max_latency": float(np.max(trial_latencies)),
            "std_latency": float(np.std(trial_latencies)),
            "p95_latency": float(np.percentile(trial_latencies, 95)),
            "measured_trial_count": len(trial_latencies)
        },
        "throughput_fps": 1.0 / float(np.median(trial_latencies)) if np.median(trial_latencies) > 0 else 0.0,
        "resource_summary": {},
        "warmup_resource_summary": {},
        "metadata": {},
        "successful_trials": len(trial_latencies),
        "failed_trials": 0,
        "failures": []
    }


# Test 1: test_frame_budget_calculation
def test_frame_budget_calculation():
    fps_map = {"synthetic_lowmotion_30fps": 30.0, "synthetic_highmotion_60fps": 60.0, "synthetic_120fps": 120.0}
    frame_count_map = {}

    rec30 = create_base_record(create_base_config(), [0.01] * 20)
    ans30 = analyze_record(rec30, fps_map, frame_count_map)
    assert ans30["frame_budget_ms"] == pytest.approx(1000.0 / 30.0)

    cfg60 = create_base_config()
    cfg60["input_id"] = "synthetic_highmotion_60fps"
    rec60 = create_base_record(cfg60, [0.01] * 20)
    ans60 = analyze_record(rec60, fps_map, frame_count_map)
    assert ans60["frame_budget_ms"] == pytest.approx(1000.0 / 60.0)


# Test 2: test_estimated_fps_calculation
def test_estimated_fps_calculation():
    fps_map = {"synthetic_lowmotion_30fps": 30.0}
    # Spatial Case
    rec_spatial = create_base_record(create_base_config(), [0.02] * 20) # 20ms median
    ans_spatial = analyze_record(rec_spatial, fps_map, {})
    assert ans_spatial["latency_interpretation"] == "per_frame"
    assert ans_spatial["estimated_processing_fps"] == pytest.approx(50.0)  # 1 / 0.02

    # Temporal Case
    cfg_temp = create_base_config()
    cfg_temp["model_id"] = "temporal_model"
    rec_temp = create_base_record(cfg_temp, [0.1] * 20) # 100ms median
    frame_count_map = {"synthetic_lowmotion_30fps": 60}
    ans_temp = analyze_record(rec_temp, fps_map, frame_count_map)
    assert ans_temp["latency_interpretation"] == "per_sequence"
    assert ans_temp["estimated_processing_fps"] == pytest.approx(600.0)  # 60 frames / 0.1s


# Test 3: test_real_time_ratio
def test_real_time_ratio():
    fps_map = {"synthetic_lowmotion_30fps": 30.0}
    # Budget is 33.33ms. Latency is 33.33ms
    rec_eq = create_base_record(create_base_config(), [1.0 / 30.0] * 20)
    ans_eq = analyze_record(rec_eq, fps_map, {})
    assert ans_eq["real_time_ratio"] == pytest.approx(1.0)
    assert ans_eq["real_time_feasible"] is True


# Test 4: test_real_time_classification_median_based
def test_real_time_classification_median_based():
    # Budget is 33.33ms. Median latency is 20ms, but mean is 40ms, and p95 is 45ms.
    # Because it is median-based, it MUST resolve as feasible.
    fps_map = {"synthetic_lowmotion_30fps": 30.0}
    latencies = [0.02] * 11 + [0.06] * 9  # median is 0.02s (20ms), mean is 38ms
    rec = create_base_record(create_base_config(), latencies)
    ans = analyze_record(rec, fps_map, {})
    assert ans["latency_ms"] == 20.0
    assert ans["real_time_feasible"] is True


# Test 5: test_p95_never_drives_classification
def test_p95_never_drives_classification():
    # p95 exceeds budget, but median does not.
    # verify real_time_feasible remains True while real_time_feasible_p95_exploratory is False.
    fps_map = {"synthetic_lowmotion_30fps": 30.0}
    latencies = [0.02] * 18 + [0.1] * 2  # median 20ms (feasible), p95 is 0.1s (100ms, unfeasible)
    rec = create_base_record(create_base_config(), latencies)
    ans = analyze_record(rec, fps_map, {})
    assert ans["real_time_feasible"] is True
    assert ans["real_time_feasible_p95_exploratory"] is False


# Test 6: test_scale_comparison_omits_unsupported_combinations
def test_scale_comparison_omits_unsupported_combinations():
    fps_map = {"synthetic_lowmotion_30fps": 30.0}
    cfg = create_base_config()
    cfg["model_id"] = "real_esrgan"
    cfg["scale"] = 3  # Invalid scale (real_esrgan only supports x2, x4)
    rec = create_base_record(cfg, [0.01] * 20)
    with pytest.raises(ValueError, match="Unsupported model\+scale combination"):
        analyze_record(rec, fps_map, {})


# Test 7: test_cpu_gpu_rows_separated
def test_cpu_gpu_rows_separated():
    # Verify parse_device_id and that device fields maintain distinct values
    assert parse_device_id("cpu") is None
    assert parse_device_id("cuda:0") == 0
    assert parse_device_id("cuda:1") == 1
    assert parse_device_id("cuda") == 0


# Test 8: test_missing_source_fps_sets_gap_flag
def test_missing_source_fps_sets_gap_flag():
    # If source_fps is missing, it sets gap flag and leaves target-dependent fields null
    rec = create_base_record(create_base_config(), [0.01] * 20)
    ans = analyze_record(rec, {}, {})
    assert ans["source_fps"] is None
    assert ans["source_fps_gap"] is True
    assert ans["real_time_feasible"] is None
    assert ans["frame_budget_ms"] is None


# Test 9: test_invalid_latency_raises
def test_invalid_latency_raises():
    fps_map = {"synthetic_lowmotion_30fps": 30.0}
    # Zero latency
    rec_zero = create_base_record(create_base_config(), [0.0] * 20)
    with pytest.raises(ValueError, match="Invalid non-positive latency observed"):
        analyze_record(rec_zero, fps_map, {})


# Test 10: test_eligibility_warnings_preserved
def test_eligibility_warnings_preserved():
    fps_map = {"synthetic_lowmotion_30fps": 30.0}
    rec = create_base_record(create_base_config(), [0.01] * 20)
    rec["failures"] = ["thermal throttling active"]
    rec["flagged"] = "high thermal state"
    ans = analyze_record(rec, fps_map, {})
    assert "Trial failure: thermal throttling active" in ans["caveats"]
    assert "Flagged session: high thermal state" in ans["caveats"]


# Test 11: test_ineligible_session_still_reports_measured_feasibility
def test_ineligible_session_still_reports_measured_feasibility():
    fps_map = {"synthetic_lowmotion_30fps": 30.0}
    rec = create_base_record(create_base_config(), [0.01] * 20)
    rec["decision_eligible"] = False
    ans = analyze_record(rec, fps_map, {})
    assert ans["real_time_feasible"] is True
    assert ans["decision_eligible"] is False


# Test 12: test_output_schema_matches_spec
def test_output_schema_matches_spec():
    fps_map = {"synthetic_lowmotion_30fps": 30.0}
    rec = create_base_record(create_base_config(), [0.01] * 20)
    ans = analyze_record(rec, fps_map, {})
    
    # Assert §10 required schema keys
    required_keys = [
        "benchmark_video_id", "model_id", "scale", "device", "cpu_ids", "num_threads",
        "gpu_device_id", "source_fps", "frame_budget_ms", "latency_ms", "p95_latency_ms",
        "p95_exploratory", "latency_interpretation", "estimated_processing_fps",
        "real_time_ratio", "real_time_feasible", "real_time_feasible_p95_exploratory",
        "budget_utilization_percent", "decision_eligible", "session_count",
        "p95_confidence", "caveats", "source_fps_gap"
    ]
    for key in required_keys:
        assert key in ans


# Test 13: test_integration_real_step5_5_fixture
def test_integration_real_step5_5_fixture():
    from adaptive_sr.benchmarking.adapters.registry import ADAPTER_MAP
    from adaptive_sr.benchmarking.adapters.base import BaseSRAdapter
    from typing import List, Optional
    import numpy as np

    class DummyFeasibilityAdapter(BaseSRAdapter):
        @property
        def model_id(self) -> str:
            return "dummy_feasibility_adapter"

        @property
        def display_name(self) -> str:
            return "Dummy Feasibility Adapter"

        @property
        def backend(self) -> str:
            return "dummy"

        @property
        def scale_factors(self) -> List[int]:
            return [2]

        @property
        def temporal_or_spatial(self) -> str:
            return "spatial"

        @property
        def precision(self) -> str:
            return "fp32"

        def is_available(self) -> bool:
            return True

        def get_unavailable_reason(self) -> Optional[str]:
            return None

        def initialize(self, device: str, scale: int, num_threads: int = None):
            self.device = device
            self.scale = scale

        def _run_inference(self, frames: list) -> list:
            import cv2
            out = []
            for f in frames:
                h, w, c = f.shape
                out.append(cv2.resize(f, (w * self.scale, h * self.scale), interpolation=cv2.INTER_NEAREST))
            return out

        def close(self):
            pass

    # Register the dummy adapter
    ADAPTER_MAP["dummy_feasibility_adapter"] = DummyFeasibilityAdapter

    harness = InferenceBenchmarkHarness()
    cpu_conf = CPUExecutionConfig(cpu_ids=[1], num_threads=2, exclude_cpu_ids=[])
    
    # Create configuration case using dummy adapter
    cfg = BenchmarkConfig(
        model_id="dummy_feasibility_adapter",
        scale=2,
        input_id="synthetic_lowmotion_30fps",
        device="cpu",
        cpu_config=cpu_conf,
        warmup_runs=0,
        measured_runs=5  # Keep it small for fast unit testing execution
    )

    # Execute harness directly to obtain a real MultiSessionResult (Step 5.5 output record)
    real_multi_session = harness.run_multi_session(cfg, num_sessions=1)
    
    # Verify the harness completed successfully
    assert real_multi_session is not None
    assert len(real_multi_session.sessions) == 1

    # Convert the MultiSessionResult to a dictionary representation
    # This matches loading serialized Step 5.5 results from disk
    record = real_multi_session.model_dump()

    # Load real maps from manifests directory
    fps_map, frame_count_map = {}, {}
    manifests_path = "data/benchmarks/sr/manifests"
    import os
    if os.path.exists(manifests_path):
        from adaptive_sr.benchmarking.fps_analysis import load_manifests_maps
        fps_map, frame_count_map = load_manifests_maps(manifests_path)
    else:
        fps_map["synthetic_lowmotion_30fps"] = 30.0

    # Run Step 5.7 analysis on the real Step 5.5 result record
    analysis = analyze_record(record, fps_map, frame_count_map)

    # Validate output analytics
    assert analysis["model_id"] == "dummy_feasibility_adapter"
    assert analysis["scale"] == 2
    assert analysis["device"] == "cpu"
    assert analysis["latency_ms"] > 0
    assert analysis["estimated_processing_fps"] > 0
    assert analysis["source_fps"] == 30.0
    assert analysis["real_time_ratio"] is not None
    assert analysis["real_time_feasible"] is not None
