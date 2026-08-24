"""
tests/test_benchmark_harness.py
================================
Step 5.5 — Inference Benchmark Harness Unit and Integration Tests.
"""

import os
import sys
import time
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch

from typing import List, Dict, Any, Tuple, Union, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adaptive_sr.benchmarking.cpu_control import CPUExecutionConfig
from adaptive_sr.benchmarking.gpu_measurement import CUDAAvailability, GPUSnapshot, GPUMeasurementBoundary
from adaptive_sr.benchmarking import (
    BenchmarkConfig,
    BenchmarkResult,
    InferenceBenchmarkHarness
)
from adaptive_sr.benchmarking.adapters.base import BaseSRAdapter
from adaptive_sr.benchmarking.adapters.registry import ADAPTER_MAP


# ---------------------------------------------------------------------------
# 1. Mock Adapter for Harness Unit Testing
# ---------------------------------------------------------------------------

class DummyHarnessAdapter(BaseSRAdapter):
    """Dummy spatial adapter to verify harness execution flow, warmup, timing, and failure paths."""

    def __init__(self, simulate_fail_at_trial: Optional[int] = None) -> None:
        self.simulate_fail_at_trial = simulate_fail_at_trial
        self._device = None
        self._scale = None
        self._initialized = False
        self.call_count = 0

    @property
    def model_id(self) -> str:
        return "dummy_harness_adapter"

    @property
    def display_name(self) -> str:
        return "Dummy Harness Adapter"

    @property
    def backend(self) -> str:
        return "dummy_backend"

    @property
    def scale_factors(self) -> List[int]:
        return [2, 3]

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

    def initialize(self, device: str, scale: int, num_threads: Optional[int] = None) -> None:
        self._device = device
        self._scale = scale
        self._initialized = True

    def _run_inference(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not self._initialized:
            raise RuntimeError("Not initialized")
        
        self.call_count += 1
        
        # Check if we should simulate a trial failure
        # In harness execution, warmup runs occur first, then measured trials.
        # Warmup = 3 runs (calls 1, 2, 3). Trials start at call 4.
        if self.simulate_fail_at_trial is not None:
            # e.g., if we fail at trial 0, it is call index 4
            trial_idx_called = self.call_count - 4  # 3 warmup runs assumed
            if trial_idx_called == self.simulate_fail_at_trial:
                raise RuntimeError(f"Simulated failure at trial {self.simulate_fail_at_trial}")

        enhanced = []
        for frame in frames:
            h, w, c = frame.shape
            out = np.zeros((h * self._scale, w * self._scale, c), dtype=np.uint8)
            enhanced.append(out)
        return enhanced

    def close(self) -> None:
        pass


@pytest.fixture
def register_dummy_adapter():
    """Temporarily registers DummyHarnessAdapter in the registry ADAPTER_MAP."""
    original_map = ADAPTER_MAP.copy()
    ADAPTER_MAP["dummy_harness_adapter"] = DummyHarnessAdapter
    yield
    # Restore registry map
    ADAPTER_MAP.clear()
    ADAPTER_MAP.update(original_map)


# ---------------------------------------------------------------------------
# 2. BenchmarkConfig validation tests
# ---------------------------------------------------------------------------

def test_benchmark_config_validation():
    """Verifies Pydantic validations for BenchmarkConfig."""
    # Correct CPU Config
    cpu_conf = CPUExecutionConfig(cpu_ids=[0], num_threads=1)
    
    cfg = BenchmarkConfig(
        model_id="dummy_harness_adapter",
        scale=2,
        input_id="synthetic_lowmotion_30fps",
        device="cpu",
        cpu_config=cpu_conf,
        warmup_runs=3,
        measured_runs=20
    )
    assert cfg.model_id == "dummy_harness_adapter"
    assert cfg.scale == 2
    assert cfg.device == "cpu"
    assert cfg.cpu_config.num_threads == 1

    # Negative scale factor
    with pytest.raises(ValueError, match="Scale must be a positive integer"):
        BenchmarkConfig(
            model_id="dummy_harness_adapter",
            scale=-1,
            input_id="synthetic_lowmotion_30fps",
            device="cpu",
            cpu_config=cpu_conf
        )

    # Missing cpu_config when device='cpu'
    with pytest.raises(ValueError, match="cpu_config must be provided when device='cpu'"):
        BenchmarkConfig(
            model_id="dummy_harness_adapter",
            scale=2,
            input_id="synthetic_lowmotion_30fps",
            device="cpu"
        )

    # CPU config provided for CUDA device
    with pytest.raises(ValueError, match="cpu_config must be None when device is CUDA"):
        BenchmarkConfig(
            model_id="dummy_harness_adapter",
            scale=2,
            input_id="synthetic_lowmotion_30fps",
            device="cuda",
            cpu_config=cpu_conf
        )

    # Unsupported device name
    with pytest.raises(ValueError, match="Unsupported device"):
        BenchmarkConfig(
            model_id="dummy_harness_adapter",
            scale=2,
            input_id="synthetic_lowmotion_30fps",
            device="tpu",
            cpu_config=cpu_conf
        )


# ---------------------------------------------------------------------------
# 3. statistics and calculations tests
# ---------------------------------------------------------------------------

def test_latency_statistics_calculations(register_dummy_adapter):
    """Verifies that mean, median, min, max, std_dev, p95 and FPS are calculated correctly."""
    harness = InferenceBenchmarkHarness()
    
    # We will patch time.perf_counter to return predictable values for trials
    # Warmup runs = 1, Measured runs = 5
    # Predictable latencies: 0.1s, 0.2s, 0.3s, 0.4s, 0.5s (average = 0.3s)
    # Perf counter increments:
    # 1. Trial 0: enter=0.0, exit=0.1 (elapsed 0.1)
    # 2. Trial 1: enter=0.1, exit=0.3 (elapsed 0.2)
    # 3. Trial 2: enter=0.3, exit=0.6 (elapsed 0.3)
    # 4. Trial 3: enter=0.6, exit=1.0 (elapsed 0.4)
    # 5. Trial 4: enter=1.0, exit=1.5 (elapsed 0.5)
    perf_times = [
        0.0, 0.1,  # Trial 0
        0.1, 0.3,  # Trial 1
        0.3, 0.6,  # Trial 2
        0.6, 1.0,  # Trial 3
        1.0, 1.5   # Trial 4
    ]
    time_idx = 0

    def mock_perf_counter():
        nonlocal time_idx
        val = perf_times[time_idx]
        time_idx += 1
        return val

    cpu_conf = CPUExecutionConfig(cpu_ids=[0], num_threads=1)
    cfg = BenchmarkConfig(
        model_id="dummy_harness_adapter",
        scale=2,
        input_id="synthetic_lowmotion_30fps",
        device="cpu",
        cpu_config=cpu_conf,
        warmup_runs=1,
        measured_runs=5
    )

    with patch("adaptive_sr.benchmarking.harness.time.perf_counter", side_effect=mock_perf_counter):
        result = harness.run_benchmark(cfg)

    assert result.successful_trials == 5
    assert result.failed_trials == 0
    assert len(result.trial_latencies) == 5
    # Raw trial latencies should match our expected times
    np.testing.assert_allclose(result.trial_latencies, [0.1, 0.2, 0.3, 0.4, 0.5])

    # Stats validation
    stats = result.latency_statistics
    assert stats.count == 5
    assert stats.mean == pytest.approx(0.3)
    assert stats.median == pytest.approx(0.3)
    assert stats.min == pytest.approx(0.1)
    assert stats.max == pytest.approx(0.5)
    assert stats.std_dev == pytest.approx(np.std([0.1, 0.2, 0.3, 0.4, 0.5]))
    
    # p95 linear percentile validation
    expected_p95 = np.percentile([0.1, 0.2, 0.3, 0.4, 0.5], 95)
    assert stats.p95 == pytest.approx(expected_p95)
    assert stats.p95_method == "numpy_linear"

    # Throughput validation (FPS = 1 / mean_latency)
    assert result.throughput_fps == pytest.approx(1.0 / 0.3)


# ---------------------------------------------------------------------------
# 4. CPU execution and ProcessMonitor integration
# ---------------------------------------------------------------------------

def test_cpu_execution_affinity_and_monitoring(register_dummy_adapter):
    """Verifies affinity restricted path and ProcessMonitor start/stop integration."""
    harness = InferenceBenchmarkHarness()
    cpu_conf = CPUExecutionConfig(cpu_ids=[0], num_threads=2)
    cfg = BenchmarkConfig(
        model_id="dummy_harness_adapter",
        scale=2,
        input_id="synthetic_lowmotion_30fps",
        device="cpu",
        cpu_config=cpu_conf,
        warmup_runs=1,
        measured_runs=2
    )

    # Let's verify that the process monitor is started and stopped, and affinity binds are applied
    # We patch psutil's Process affinity call
    with patch("psutil.Process.cpu_affinity", return_value=[0]) as mock_affinity:
        # Mock cpu monitor start and stop calls
        with patch("adaptive_sr.benchmarking.cpu_control.BenchmarkProcessMonitor.start") as mock_start, \
             patch("adaptive_sr.benchmarking.cpu_control.BenchmarkProcessMonitor.stop") as mock_stop, \
             patch("adaptive_sr.benchmarking.cpu_control.BenchmarkProcessMonitor.get_samples", return_value=[]) as mock_samples:
            
            result = harness.run_benchmark(cfg)
            
            # Verify that affinity was set to [0] during entry and restored to the previous mask on exit
            assert mock_affinity.call_count >= 2
            # Verify process monitor started and stopped
            assert mock_start.call_count == 1
            assert mock_stop.call_count == 1


# ---------------------------------------------------------------------------
# 5. GPU execution (Mocked CUDA)
# ---------------------------------------------------------------------------

def test_gpu_benchmark_cuda_synchronization_and_monitoring(register_dummy_adapter):
    """Verifies that GPU benchmarks call torch.cuda.synchronize() and integrate GPUMonitor."""
    harness = InferenceBenchmarkHarness()
    cfg = BenchmarkConfig(
        model_id="dummy_harness_adapter",
        scale=2,
        input_id="synthetic_lowmotion_30fps",
        device="cuda:0",
        warmup_runs=1,
        measured_runs=3,
        gpu_sampling_interval=0.5
    )

    # Mock get_cuda_availability to return AVAILABLE
    mock_avail = {
        "status": CUDAAvailability.AVAILABLE,
        "device_count": 1,
        "device_ids": [0]
    }

    # Mock GPU snapshot values
    dummy_snapshot = GPUSnapshot(
        timestamp="2026-08-24T12:00:00.000Z",
        device_id=0,
        gpu_name="NVIDIA GeForce RTX Test",
        gpu_utilization_percent=45.0,
        memory_utilization_percent=12.0,
        gpu_memory_total_bytes=8192 * 1024 * 1024,
        gpu_memory_free_bytes=4096 * 1024 * 1024,
        process_gpu_memory_allocated_bytes=100 * 1024 * 1024,
        process_gpu_memory_reserved_bytes=200 * 1024 * 1024,
        nvml_available=True,
        utilization_source="nvml"
    )

    with patch("adaptive_sr.benchmarking.harness.get_cuda_availability", return_value=mock_avail), \
         patch("torch.cuda.synchronize") as mock_sync, \
         patch("adaptive_sr.benchmarking.harness.take_gpu_snapshot", return_value=dummy_snapshot), \
         patch("adaptive_sr.benchmarking.gpu_measurement.GPUMonitor.start") as mock_gpu_start, \
         patch("adaptive_sr.benchmarking.gpu_measurement.GPUMonitor.stop") as mock_gpu_stop, \
         patch("adaptive_sr.benchmarking.gpu_measurement.GPUMonitor.get_samples", return_value=[dummy_snapshot]) as mock_gpu_samples:

        result = harness.run_benchmark(cfg)

        # Verify that synchronization was called to ensure CPU/GPU alignment
        # Synchronize must occur:
        # - after warmup (1 call)
        # - before trial loop (1 call, or before each trial)
        # - around each trial timing boundaries (2 calls per trial * 3 trials = 6 calls)
        # - before snapshot captures
        assert mock_sync.call_count >= 6
        assert result.successful_trials == 3

        # Verify GPU Monitor lifecycle calls
        assert mock_gpu_start.call_count == 1
        assert mock_gpu_stop.call_count == 1
        
        # Verify resources summary matches mocked snapshot
        assert result.resource_summary.gpu_utilization_mean == 45.0
        assert result.resource_summary.memory_allocated_before_mb == 100.0
        assert result.resource_summary.memory_allocated_after_mb == 100.0


def test_gpu_benchmark_skip_when_cuda_unavailable(register_dummy_adapter):
    """Verifies harness fails/skips CUDA benchmark case if CUDA is unavailable."""
    harness = InferenceBenchmarkHarness()
    cfg = BenchmarkConfig(
        model_id="dummy_harness_adapter",
        scale=2,
        input_id="synthetic_lowmotion_30fps",
        device="cuda",
        warmup_runs=1,
        measured_runs=5
    )

    # Mock availability to return UNAVAILABLE
    mock_avail = {
        "status": CUDAAvailability.UNAVAILABLE,
        "device_count": 0,
        "device_ids": []
    }

    with patch("adaptive_sr.benchmarking.harness.get_cuda_availability", return_value=mock_avail):
        with pytest.raises(RuntimeError, match="CUDA execution requested.*but CUDA is unavailable"):
            harness.run_benchmark(cfg)


# ---------------------------------------------------------------------------
# 6. Warmup and failure handling tests
# ---------------------------------------------------------------------------

def test_warmup_excluded_from_latencies(register_dummy_adapter):
    """Verifies that warmup iterations are executed but excluded from measured results list."""
    harness = InferenceBenchmarkHarness()
    
    cpu_conf = CPUExecutionConfig(cpu_ids=[0], num_threads=1)
    cfg = BenchmarkConfig(
        model_id="dummy_harness_adapter",
        scale=2,
        input_id="synthetic_lowmotion_30fps",
        device="cpu",
        cpu_config=cpu_conf,
        warmup_runs=3,
        measured_runs=5
    )

    # We will verify how many times _run_inference is called on the adapter
    # Total calls should be: 3 (warmup) + 5 (trials) = 8 calls
    adapter_instance = DummyHarnessAdapter()
    
    with patch("adaptive_sr.benchmarking.harness.get_adapter", return_value=adapter_instance):
        result = harness.run_benchmark(cfg)
        
    assert result.successful_trials == 5
    assert len(result.trial_latencies) == 5
    assert adapter_instance.call_count == 8


def test_graceful_single_trial_failure_recording(register_dummy_adapter):
    """Verifies that a trial failure does not crash the harness, but is recorded."""
    harness = InferenceBenchmarkHarness()
    
    cpu_conf = CPUExecutionConfig(cpu_ids=[0], num_threads=1)
    cfg = BenchmarkConfig(
        model_id="dummy_harness_adapter",
        scale=2,
        input_id="synthetic_lowmotion_30fps",
        device="cpu",
        cpu_config=cpu_conf,
        warmup_runs=3,
        measured_runs=5
    )

    # We tell the dummy adapter to raise a failure at measured trial index 2 (call count = 3 + 2 = 5)
    adapter_instance = DummyHarnessAdapter(simulate_fail_at_trial=2)
    
    with patch("adaptive_sr.benchmarking.harness.get_adapter", return_value=adapter_instance):
        result = harness.run_benchmark(cfg)
        
    # Out of 5 trials, 4 should succeed, 1 should fail
    assert result.successful_trials == 4
    assert result.failed_trials == 1
    assert len(result.trial_latencies) == 4
    assert len(result.failures) == 1
    assert "Simulated failure at trial 2" in result.failures[0]

    # Verify trials detail inspectability
    trials = result.metadata["trials"]
    assert len(trials) == 5
    assert trials[0]["success"] is True
    assert trials[2]["success"] is False
    assert "Simulated failure" in trials[2]["error_message"]


# ---------------------------------------------------------------------------
# 7. Real integration smoke test
# ---------------------------------------------------------------------------

def test_real_benchmark_integration_smoke():
    """Smoke test running a real tiny benchmark on CPU using tinysr (FSRCNN) model."""
    manifest_path = "data/benchmarks/sr/manifests/benchmark_manifest.json"
    if not os.path.exists(manifest_path):
        pytest.skip("Step 5.1 dataset not prepared, skipping integration smoke test.")

    harness = InferenceBenchmarkHarness()
    cpu_conf = CPUExecutionConfig(cpu_ids=[0], num_threads=1)
    cfg = BenchmarkConfig(
        model_id="tinysr",
        scale=2,
        input_id="synthetic_lowmotion_30fps",
        device="cpu",
        cpu_config=cpu_conf,
        warmup_runs=1,
        measured_runs=2
    )

    result = harness.run_benchmark(cfg)
    
    assert result.successful_trials == 2
    assert result.failed_trials == 0
    assert len(result.trial_latencies) == 2
    assert result.throughput_fps > 0.0
    
    # Check that metadata is filled properly
    assert result.metadata["input_details"]["width"] == 640
    assert result.metadata["input_details"]["height"] == 360
    assert result.metadata["input_details"]["scale"] == 2
