"""
tests/test_cpu_control.py
=========================
Step 5.3 — Logical CPU affinity and process-monitoring unit and integration tests.
"""

import os
import sys
import time
import threading
import pytest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import psutil
import torch

from adaptive_sr.benchmarking.cpu_control import (
    get_available_cpus,
    get_current_affinity,
    select_cpu_ids,
    CPUExecutionConfig,
    cpu_affinity_context,
    BenchmarkProcessMonitor,
    benchmark_execution_context
)
from adaptive_sr.benchmarking.adapters.registry import get_adapter


# ---------------------------------------------------------------------------
# 1. CPU Discovery and Selection Tests
# ---------------------------------------------------------------------------

def test_cpu_discovery():
    """Verifies that system CPUs are dynamically discovered as logical indices."""
    cpus = get_available_cpus()
    assert isinstance(cpus, list)
    assert len(cpus) > 0
    assert all(isinstance(c, int) for c in cpus)
    # Check that indices are sequential starting from 0
    assert cpus == list(range(len(cpus)))


def test_valid_affinity_selection():
    """Verifies that selecting logical CPU IDs returns a subset of size N."""
    available = get_available_cpus()
    selected = select_cpu_ids(1)
    assert len(selected) == 1
    assert selected[0] in available


def test_deterministic_cpu_selection():
    """Verifies that CPU subset selection is fully deterministic."""
    available = get_available_cpus()
    if len(available) >= 2:
        selected_1 = select_cpu_ids(2)
        selected_2 = select_cpu_ids(2)
        assert selected_1 == selected_2
        assert selected_1 == [0, 1]


def test_requested_cpu_count_too_large():
    """Verifies that requesting more cores than available on the host fails explicitly."""
    available_count = len(get_available_cpus())
    with pytest.raises(ValueError, match="exceeds available CPUs"):
        select_cpu_ids(available_count + 1)

    with pytest.raises(ValueError, match="must be greater than 0"):
        select_cpu_ids(0)

    with pytest.raises(ValueError, match="must be greater than 0"):
        select_cpu_ids(-5)


# ---------------------------------------------------------------------------
# 2. Immutable CPUExecutionConfig Validation Rules
# ---------------------------------------------------------------------------

def test_config_empty_cpu_set_rejection():
    """Verifies that an empty CPU set raises a ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        CPUExecutionConfig(cpu_ids=[])


def test_config_negative_cpu_id_rejection():
    """Verifies that negative CPU ID numbers are rejected."""
    with pytest.raises(ValueError, match="must be non-negative"):
        CPUExecutionConfig(cpu_ids=[-1])


def test_config_duplicate_cpu_rejection():
    """Verifies that duplicate CPU ID indices are rejected."""
    with pytest.raises(ValueError, match="Duplicate CPU ID"):
        CPUExecutionConfig(cpu_ids=[0, 0])


def test_config_out_of_bounds_cpu_rejection():
    """Verifies that logical CPU IDs not available on the host are rejected."""
    available_count = len(get_available_cpus())
    with pytest.raises(ValueError, match="not available on this host"):
        CPUExecutionConfig(cpu_ids=[available_count])


def test_config_num_threads_validation():
    """Verifies that thread count bounds validation rules are honored."""
    # num_threads = None is accepted
    cfg = CPUExecutionConfig(cpu_ids=[0], num_threads=None)
    assert cfg.num_threads is None

    # num_threads <= 0 raises ValueError
    with pytest.raises(ValueError, match="greater than 0"):
        CPUExecutionConfig(cpu_ids=[0], num_threads=0)

    with pytest.raises(ValueError, match="greater than 0"):
        CPUExecutionConfig(cpu_ids=[0], num_threads=-2)


# ---------------------------------------------------------------------------
# 3. CPU Affinity Restorations and Safety Guarantees
# ---------------------------------------------------------------------------

def test_cpu_affinity_applied():
    """Verifies that CPU affinity restriction is applied to the current process."""
    available = get_available_cpus()
    target_ids = [available[0]]
    
    with cpu_affinity_context(target_ids):
        current_affinity = get_current_affinity()
        assert current_affinity == target_ids


def test_cpu_affinity_restored_normal():
    """Verifies that CPU affinity is fully restored on normal context exit."""
    original_affinity = get_current_affinity()
    available = get_available_cpus()
    
    target_ids = [available[0]]
    with cpu_affinity_context(target_ids):
        assert get_current_affinity() == target_ids
        
    # Verify restoration
    assert get_current_affinity() == original_affinity


def test_cpu_affinity_restored_exception():
    """Verifies that CPU affinity is fully restored even if an exception occurs inside."""
    original_affinity = get_current_affinity()
    available = get_available_cpus()
    
    target_ids = [available[0]]
    try:
        with cpu_affinity_context(target_ids):
            assert get_current_affinity() == target_ids
            raise RuntimeError("Simulated benchmark failure")
    except RuntimeError:
        pass
        
    # Verify restoration
    assert get_current_affinity() == original_affinity


# ---------------------------------------------------------------------------
# 4. ProcessMonitor Observability and Lifecycle Verification
# ---------------------------------------------------------------------------

def test_process_monitor_starts_and_stops_cleanly():
    """Verifies that the background process monitor starts and stops cleanly."""
    monitor = BenchmarkProcessMonitor()
    assert monitor._thread is None
    
    monitor.start()
    assert monitor._thread is not None
    assert monitor._thread.is_alive()
    
    monitor.stop()
    assert monitor._thread is None


def test_process_monitor_observes_intended_process():
    """Verifies that the process monitor attaches to the intended process ID."""
    pid = os.getpid()
    monitor = BenchmarkProcessMonitor(pid=pid)
    assert monitor.pid == pid


def test_process_monitor_no_thread_leak():
    """Verifies that stopping the monitor leaves no runaway background threads."""
    initial_threads = threading.active_count()
    
    monitor = BenchmarkProcessMonitor(sample_interval=0.01)
    monitor.start()
    time.sleep(0.05)
    monitor.stop()
    
    # Wait a brief moment for the daemon thread to clean up
    time.sleep(0.05)
    assert threading.active_count() == initial_threads


def test_process_monitor_produces_telemetry_samples():
    """Verifies that the background process monitor records valid snapshot telemetry."""
    monitor = BenchmarkProcessMonitor(sample_interval=0.02)
    monitor.start()
    time.sleep(0.08)
    monitor.stop()
    
    samples = monitor.get_samples()
    assert len(samples) > 0
    
    # Verify schema properties
    snap = samples[0]
    assert snap.process_id == os.getpid()
    assert isinstance(snap.cpu_percent, float)
    assert snap.cpu_percent >= 0.0
    assert snap.memory_used_bytes > 0
    assert snap.timestamp.endswith("Z")


# ---------------------------------------------------------------------------
# 5. Integrated Execution Context Lifecycle
# ---------------------------------------------------------------------------

def test_integrated_execution_context_lifecycle():
    """Verifies the complete monitor-start -> affinity-apply -> affinity-restore -> monitor-stop lifecycle."""
    original_affinity = get_current_affinity()
    available = get_available_cpus()
    
    config = CPUExecutionConfig(cpu_ids=[available[0]], num_threads=1)
    monitor = BenchmarkProcessMonitor(sample_interval=0.01)
    
    with benchmark_execution_context(config, monitor=monitor):
        # 1. Verify affinity is applied inside
        assert get_current_affinity() == [available[0]]
        # 2. Verify monitor is active inside
        assert monitor._thread is not None
        assert monitor._thread.is_alive()
        time.sleep(0.05)
        
    # 3. Verify affinity restored outside
    assert get_current_affinity() == original_affinity
    # 4. Verify monitor stopped outside
    assert monitor._thread is None
    assert len(monitor.get_samples()) > 0


# ---------------------------------------------------------------------------
# 6. PyTorch & ONNX Runtime Thread Observations
# ---------------------------------------------------------------------------

def test_pytorch_thread_config_observable():
    """Verifies that PyTorch thread configuration is applied and observable via backend calls."""
    adapter = get_adapter("tinysr")
    adapter.initialize(device="cpu", scale=2, num_threads=3)
    
    assert torch.get_num_threads() == 3
    assert adapter._num_threads == 3


def test_onnx_runtime_thread_config_mocked():
    """Verifies ONNX Runtime session thread configuration setting is verified inside the adapter."""
    mock_ort = MagicMock()
    mock_backend = MagicMock()
    
    # We patch import systems and path locations to trigger FSRCNNInt8Adapter execution configuration
    with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
        with patch("adaptive_sr.benchmarking.adapters.fsrcnn_int8.HAS_ORT", True):
            with patch("os.path.exists", return_value=True):
                from adaptive_sr.benchmarking.adapters.fsrcnn_int8 import FSRCNNInt8Adapter
                
                adapter = FSRCNNInt8Adapter()
                # Inject mocked backend imports
                adapter._backend_module = mock_backend
                
                # Verify that initializing ORT adapter sets options intra_op_num_threads
                adapter.initialize(device="cpu", scale=2, num_threads=4)
                
                # Check that SessionOptions was constructed and intra_op_num_threads was configured
                assert mock_ort.SessionOptions.called
                opts = mock_ort.SessionOptions.return_value
                assert opts.intra_op_num_threads == 4


# ---------------------------------------------------------------------------
# 7. Step 5.2 Adapter Smoke Test Compatibility
# ---------------------------------------------------------------------------

def test_step5_2_adapter_smoke_works():
    """Verifies that the Step 5.2 model execution smoke test continues to pass."""
    from tests.test_model_adapters import test_smoke_integration_with_step5_1_dataset
    test_smoke_integration_with_step5_1_dataset()
