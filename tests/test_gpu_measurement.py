"""
tests/test_gpu_measurement.py
==============================
Step 5.4 — GPU Measurement Infrastructure Tests.

Test coverage (14 areas per step5.4.md §26):

 1. CUDA availability detection
 2. GPU enumeration
 3. Device metadata retrieval
 4. Invalid GPU ID rejection
 5. GPU memory snapshot schema
 6. GPU utilization availability handling
 7. Sampling lifecycle
 8. Sampling interval configuration
 9. Clean monitor shutdown
10. Exception-safe shutdown
11. Multi-GPU device selection logic
12. No-GPU graceful behavior
13. Step 5.2 CUDA adapter compatibility
14. Existing Step 0–4 tests remain passing (regression marker)

Design constraints:
- Do NOT assert exact GPU utilization percentages.
- Do NOT assert exact memory byte values.
- Use structural / range / invariant checks only.
- Do NOT fake a GPU.  GPU-specific integration tests are skipped
  when CUDA hardware is unavailable via pytest.mark.skipif.
- Tests must pass on a CPU-only development machine.
"""

import os
import sys
import time
import threading
from typing import Optional
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from unittest.mock import MagicMock, patch, PropertyMock

from adaptive_sr.benchmarking.gpu_measurement import (
    CUDAAvailability,
    get_cuda_availability,
    require_cuda,
    GPUDeviceInfo,
    get_gpu_info,
    list_gpus,
    validate_device_id,
    take_gpu_snapshot,
    GPUMeasurementBoundary,
    capture_gpu_boundary,
    GPUMonitor,
    gpu_measurement_context,
    NVMLContext,
    _NVML_IMPORTABLE,
)
from adaptive_sr.shared.schemas import GPUSnapshot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CUDA_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0

cuda_only = pytest.mark.skipif(
    not _CUDA_AVAILABLE,
    reason="No CUDA GPU available on this host — GPU runtime test skipped."
)

multi_gpu_only = pytest.mark.skipif(
    not (_CUDA_AVAILABLE and torch.cuda.device_count() >= 2),
    reason="Requires at least 2 CUDA GPUs."
)


# ===========================================================================
# 1. CUDA AVAILABILITY DETECTION
# ===========================================================================

class TestCUDAAvailabilityDetection:
    """§26.1 — CUDA availability detection."""

    def test_returns_dict_with_required_keys(self):
        info = get_cuda_availability()
        assert "status" in info
        assert "device_count" in info
        assert "device_ids" in info

    def test_status_is_cuda_availability_enum(self):
        info = get_cuda_availability()
        assert isinstance(info["status"], CUDAAvailability)

    def test_device_count_is_non_negative_int(self):
        info = get_cuda_availability()
        assert isinstance(info["device_count"], int)
        assert info["device_count"] >= 0

    def test_device_ids_is_list_of_ints(self):
        info = get_cuda_availability()
        assert isinstance(info["device_ids"], list)
        assert all(isinstance(i, int) for i in info["device_ids"])

    def test_device_ids_length_matches_count(self):
        info = get_cuda_availability()
        assert len(info["device_ids"]) == info["device_count"]

    def test_unavailable_when_no_cuda(self):
        """Simulate CUDA being unavailable — status must be UNAVAILABLE."""
        with patch("torch.cuda.is_available", return_value=False):
            info = get_cuda_availability()
        assert info["status"] == CUDAAvailability.UNAVAILABLE
        assert info["device_count"] == 0
        assert info["device_ids"] == []

    def test_no_device_when_cuda_present_but_zero_gpus(self):
        """Simulate CUDA present but no devices visible."""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=0):
            info = get_cuda_availability()
        assert info["status"] == CUDAAvailability.NO_DEVICE
        assert info["device_count"] == 0

    def test_available_status_with_gpu(self):
        """Simulate CUDA available with two GPUs."""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=2):
            info = get_cuda_availability()
        assert info["status"] == CUDAAvailability.AVAILABLE
        assert info["device_count"] == 2
        assert info["device_ids"] == [0, 1]

    def test_enum_values_are_strings(self):
        """All enum values must be meaningful strings, not blank."""
        for member in CUDAAvailability:
            assert isinstance(member.value, str)
            assert len(member.value) > 0


# ===========================================================================
# 2. GPU ENUMERATION
# ===========================================================================

class TestGPUEnumeration:
    """§26.2 — GPU enumeration."""

    def test_list_gpus_returns_list(self):
        result = list_gpus()
        assert isinstance(result, list)

    def test_list_gpus_empty_when_no_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            result = list_gpus()
        assert result == []

    def test_list_gpus_returns_gpu_device_info_instances(self):
        if not _CUDA_AVAILABLE:
            result = list_gpus()
            assert result == []
            return
        result = list_gpus()
        for item in result:
            assert isinstance(item, GPUDeviceInfo)

    @cuda_only
    def test_list_gpus_count_matches_device_count(self):
        result = list_gpus()
        assert len(result) == torch.cuda.device_count()

    @cuda_only
    def test_list_gpus_device_ids_are_sequential(self):
        result = list_gpus()
        ids = [g.device_id for g in result]
        assert ids == list(range(len(result)))

    @cuda_only
    def test_list_gpus_names_are_nonempty_strings(self):
        result = list_gpus()
        for gpu in result:
            assert isinstance(gpu.device_name, str)
            assert len(gpu.device_name) > 0
            # Must not be just a generic "GPU" label
            assert gpu.device_name.lower() != "gpu"

    @cuda_only
    def test_list_gpus_total_memory_positive(self):
        result = list_gpus()
        for gpu in result:
            assert gpu.total_memory_bytes > 0


# ===========================================================================
# 3. DEVICE METADATA RETRIEVAL
# ===========================================================================

class TestDeviceMetadataRetrieval:
    """§26.3 — Device metadata retrieval."""

    @cuda_only
    def test_get_gpu_info_returns_gpu_device_info(self):
        info = get_gpu_info(0)
        assert isinstance(info, GPUDeviceInfo)

    @cuda_only
    def test_get_gpu_info_device_id_correct(self):
        info = get_gpu_info(0)
        assert info.device_id == 0

    @cuda_only
    def test_get_gpu_info_device_name_nonempty(self):
        info = get_gpu_info(0)
        assert isinstance(info.device_name, str)
        assert len(info.device_name) > 0

    @cuda_only
    def test_get_gpu_info_total_memory_positive(self):
        info = get_gpu_info(0)
        assert isinstance(info.total_memory_bytes, int)
        assert info.total_memory_bytes > 0

    @cuda_only
    def test_get_gpu_info_compute_capability_format(self):
        """Compute capability should be 'major.minor' format when available."""
        info = get_gpu_info(0)
        if info.compute_capability is not None:
            parts = info.compute_capability.split(".")
            assert len(parts) == 2
            assert all(p.isdigit() for p in parts)

    @cuda_only
    def test_get_gpu_info_optional_fields_none_or_string(self):
        """Optional identity fields must be either None or a non-empty string."""
        info = get_gpu_info(0)
        optional_fields = [
            info.compute_capability,
            info.cuda_runtime_version,
            info.pytorch_cuda_version,
            info.driver_version,
        ]
        for val in optional_fields:
            assert val is None or (isinstance(val, str) and len(val) > 0)

    @cuda_only
    def test_gpu_device_info_is_frozen(self):
        """GPUDeviceInfo must be immutable (frozen dataclass)."""
        info = get_gpu_info(0)
        with pytest.raises((AttributeError, TypeError)):
            info.device_id = 99  # type: ignore


# ===========================================================================
# 4. INVALID GPU ID REJECTION
# ===========================================================================

class TestInvalidGPUIDRejection:
    """§26.4 — Invalid GPU ID rejection."""

    def test_negative_device_id_raises_value_error(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_device_id(-1)

    def test_negative_large_device_id_raises_value_error(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_device_id(-100)

    def test_out_of_range_device_id_raises_runtime_error(self):
        """Requesting a device ID beyond available count must raise RuntimeError."""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1):
            with pytest.raises(RuntimeError, match="device 0 is"):
                require_cuda(device_id=1)

    def test_cuda_unavailable_raises_runtime_error(self):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="not available"):
                require_cuda(device_id=0)

    def test_error_message_mentions_requested_id(self):
        """Error message must name the requested and available device counts."""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=2):
            with pytest.raises(RuntimeError) as exc_info:
                require_cuda(device_id=5)
        assert "5" in str(exc_info.value)

    def test_get_gpu_info_negative_id_raises(self):
        with pytest.raises(ValueError):
            get_gpu_info(-1)

    def test_get_gpu_info_no_cuda_raises(self):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError):
                get_gpu_info(0)

    def test_gpu_monitor_negative_id_raises(self):
        with pytest.raises(ValueError):
            GPUMonitor(device_id=-1)

    def test_gpu_monitor_no_cuda_raises(self):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError):
                GPUMonitor(device_id=0)


# ===========================================================================
# 5. GPU MEMORY SNAPSHOT SCHEMA
# ===========================================================================

class TestGPUSnapshotSchema:
    """§26.5 — GPU memory snapshot schema validation."""

    def _make_snapshot(self, **overrides) -> GPUSnapshot:
        """Helper: build a minimal valid GPUSnapshot."""
        defaults = dict(
            timestamp="2026-01-01T00:00:00.000000Z",
            device_id=0,
            gpu_name="NVIDIA Test GPU",
            gpu_utilization_percent=None,
            memory_utilization_percent=None,
            gpu_memory_total_bytes=None,
            gpu_memory_free_bytes=None,
            process_gpu_memory_allocated_bytes=None,
            process_gpu_memory_reserved_bytes=None,
            nvml_available=False,
            utilization_source="unavailable",
        )
        defaults.update(overrides)
        return GPUSnapshot(**defaults)

    def test_snapshot_creates_successfully(self):
        snap = self._make_snapshot()
        assert isinstance(snap, GPUSnapshot)

    def test_snapshot_timestamp_ends_with_z(self):
        snap = self._make_snapshot()
        assert snap.timestamp.endswith("Z")

    def test_snapshot_device_id_is_int(self):
        snap = self._make_snapshot(device_id=0)
        assert isinstance(snap.device_id, int)

    def test_snapshot_gpu_name_is_str(self):
        snap = self._make_snapshot()
        assert isinstance(snap.gpu_name, str)

    def test_snapshot_utilization_none_when_nvml_absent(self):
        """When nvml_available=False, utilization fields MUST be None."""
        snap = self._make_snapshot(nvml_available=False)
        assert snap.gpu_utilization_percent is None
        assert snap.memory_utilization_percent is None

    def test_snapshot_utilization_zero_is_distinct_from_none(self):
        """0.0 utilization is a meaningful value — must not be treated as None."""
        snap = self._make_snapshot(
            nvml_available=True,
            gpu_utilization_percent=0.0,
            utilization_source="nvml",
        )
        assert snap.gpu_utilization_percent == 0.0
        assert snap.gpu_utilization_percent is not None

    def test_snapshot_nvml_available_bool(self):
        snap = self._make_snapshot(nvml_available=False)
        assert isinstance(snap.nvml_available, bool)

    def test_snapshot_utilization_source_valid_values(self):
        valid_sources = {"nvml", "pytorch_allocator", "unavailable"}
        for src in valid_sources:
            snap = self._make_snapshot(utilization_source=src)
            assert snap.utilization_source in valid_sources

    def test_snapshot_memory_fields_none_or_positive_int(self):
        """Memory byte fields must be either None or a non-negative int."""
        snap = self._make_snapshot(
            gpu_memory_total_bytes=8_000_000_000,
            gpu_memory_free_bytes=7_000_000_000,
            process_gpu_memory_allocated_bytes=512_000,
            process_gpu_memory_reserved_bytes=1_024_000,
        )
        for val in [
            snap.gpu_memory_total_bytes,
            snap.gpu_memory_free_bytes,
            snap.process_gpu_memory_allocated_bytes,
            snap.process_gpu_memory_reserved_bytes,
        ]:
            assert val is None or (isinstance(val, int) and val >= 0)

    def test_snapshot_memory_total_gte_free_when_both_present(self):
        snap = self._make_snapshot(
            gpu_memory_total_bytes=8_000_000_000,
            gpu_memory_free_bytes=7_000_000_000,
        )
        assert snap.gpu_memory_total_bytes >= snap.gpu_memory_free_bytes

    def test_snapshot_reserved_gte_allocated_when_both_present(self):
        """PyTorch reserved ≥ allocated is a fundamental allocator invariant."""
        snap = self._make_snapshot(
            process_gpu_memory_allocated_bytes=512_000,
            process_gpu_memory_reserved_bytes=1_024_000,
        )
        assert snap.process_gpu_memory_reserved_bytes >= snap.process_gpu_memory_allocated_bytes


# ===========================================================================
# 6. GPU UTILIZATION AVAILABILITY HANDLING
# ===========================================================================

class TestGPUUtilizationHandling:
    """§26.6 — GPU utilization availability handling."""

    def test_nvml_unavailable_yields_none_not_zero(self):
        """Critically: missing utilization must be None, never 0."""
        with patch(
            "adaptive_sr.benchmarking.gpu_measurement._NVML_IMPORTABLE",
            False
        ):
            # Simulate a CUDA environment with one device
            mock_props = MagicMock()
            mock_props.name = "Mock GPU"
            mock_props.total_memory = 4_000_000_000

            with patch("torch.cuda.get_device_properties", return_value=mock_props), \
                 patch("torch.cuda.memory_allocated", return_value=0), \
                 patch("torch.cuda.memory_reserved", return_value=0):
                snap = take_gpu_snapshot(0)

        assert snap.gpu_utilization_percent is None
        assert snap.memory_utilization_percent is None
        assert snap.nvml_available is False

    def test_utilization_source_is_unavailable_without_nvml(self):
        with patch(
            "adaptive_sr.benchmarking.gpu_measurement._NVML_IMPORTABLE",
            False
        ):
            mock_props = MagicMock()
            mock_props.name = "Mock GPU"
            mock_props.total_memory = 4_000_000_000

            with patch("torch.cuda.get_device_properties", return_value=mock_props), \
                 patch("torch.cuda.memory_allocated", return_value=0), \
                 patch("torch.cuda.memory_reserved", return_value=0):
                snap = take_gpu_snapshot(0)

        # Without NVML, source is 'pytorch_allocator' (if torch available) or 'unavailable'
        assert snap.utilization_source in {"pytorch_allocator", "unavailable"}

    def test_nvml_context_available_false_when_not_importable(self):
        with patch(
            "adaptive_sr.benchmarking.gpu_measurement._NVML_IMPORTABLE",
            False
        ):
            with NVMLContext() as ctx:
                assert ctx.available is False

    def test_utilization_source_nvml_when_nvml_works(self):
        """When NVML succeeds, utilization_source must be 'nvml'."""
        mock_util = MagicMock()
        mock_util.gpu = 42
        mock_util.memory = 10
        mock_mem = MagicMock()
        mock_mem.total = 8_000_000_000
        mock_mem.free = 6_000_000_000
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem

        mock_props = MagicMock()
        mock_props.name = "Mock NVML GPU"
        mock_props.total_memory = 8_000_000_000

        with patch("adaptive_sr.benchmarking.gpu_measurement._NVML_IMPORTABLE", True), \
             patch("torch.cuda.get_device_properties", return_value=mock_props), \
             patch("torch.cuda.memory_allocated", return_value=1_000_000), \
             patch("torch.cuda.memory_reserved", return_value=2_000_000), \
             patch(
                 "adaptive_sr.benchmarking.gpu_measurement.NVMLContext.__enter__",
                 lambda self: setattr(self, "available", True) or
                              setattr(self, "pynvml", mock_pynvml) or self,
             ):
            snap = take_gpu_snapshot(0)

        assert snap.utilization_source == "nvml"
        assert snap.nvml_available is True
        assert snap.gpu_utilization_percent == 42.0
        assert snap.gpu_memory_free_bytes == 6_000_000_000


# ===========================================================================
# 7. SAMPLING LIFECYCLE
# ===========================================================================

class TestSamplingLifecycle:
    """§26.7 — Sampling lifecycle: start → sample → stop."""

    @cuda_only
    def test_monitor_start_stop_collects_samples(self):
        monitor = GPUMonitor(device_id=0, sample_interval=0.1)
        monitor.start()
        time.sleep(0.35)  # Allow ~3 samples
        monitor.stop()
        samples = monitor.get_samples()
        assert len(samples) >= 1

    @cuda_only
    def test_all_samples_are_gpu_snapshot_instances(self):
        monitor = GPUMonitor(device_id=0, sample_interval=0.1)
        monitor.start()
        time.sleep(0.25)
        monitor.stop()
        for snap in monitor.get_samples():
            assert isinstance(snap, GPUSnapshot)

    @cuda_only
    def test_samples_have_correct_device_id(self):
        monitor = GPUMonitor(device_id=0, sample_interval=0.1)
        monitor.start()
        time.sleep(0.25)
        monitor.stop()
        for snap in monitor.get_samples():
            assert snap.device_id == 0

    @cuda_only
    def test_samples_timestamps_are_utc_strings(self):
        monitor = GPUMonitor(device_id=0, sample_interval=0.1)
        monitor.start()
        time.sleep(0.25)
        monitor.stop()
        for snap in monitor.get_samples():
            assert isinstance(snap.timestamp, str)
            assert snap.timestamp.endswith("Z")

    @cuda_only
    def test_context_manager_collects_samples(self):
        with GPUMonitor(device_id=0, sample_interval=0.1) as monitor:
            time.sleep(0.25)
        samples = monitor.get_samples()
        assert len(samples) >= 1

    @cuda_only
    def test_gpu_measurement_context_collects_samples(self):
        with gpu_measurement_context(device_id=0, sample_interval=0.1) as monitor:
            time.sleep(0.25)
        samples = monitor.get_samples()
        assert len(samples) >= 1

    @cuda_only
    def test_sample_count_method(self):
        monitor = GPUMonitor(device_id=0, sample_interval=0.1)
        monitor.start()
        time.sleep(0.25)
        monitor.stop()
        assert monitor.sample_count() == len(monitor.get_samples())


# ===========================================================================
# 8. SAMPLING INTERVAL CONFIGURATION
# ===========================================================================

class TestSamplingIntervalConfiguration:
    """§26.8 — Sampling interval configuration."""

    @cuda_only
    def test_custom_interval_is_stored(self):
        monitor = GPUMonitor(device_id=0, sample_interval=1.23)
        assert monitor.sample_interval == 1.23

    @cuda_only
    def test_default_interval_is_0_5_seconds(self):
        """Default sample_interval must be 0.5 s (documented in spec §13)."""
        monitor = GPUMonitor(device_id=0)
        assert monitor.sample_interval == 0.5

    @cuda_only
    def test_faster_interval_produces_more_samples(self):
        """A faster interval should yield more samples in the same wall time."""
        monitor_fast = GPUMonitor(device_id=0, sample_interval=0.05)
        monitor_slow = GPUMonitor(device_id=0, sample_interval=0.4)
        monitor_fast.start()
        monitor_slow.start()
        time.sleep(0.5)
        monitor_fast.stop()
        monitor_slow.stop()
        assert monitor_fast.sample_count() >= monitor_slow.sample_count()


# ===========================================================================
# 9. CLEAN MONITOR SHUTDOWN
# ===========================================================================

class TestCleanMonitorShutdown:
    """§26.9 — Clean monitor shutdown — no zombie threads."""

    @cuda_only
    def test_stop_joins_thread(self):
        monitor = GPUMonitor(device_id=0, sample_interval=0.1)
        monitor.start()
        thread_ref = monitor._thread
        monitor.stop()
        assert not thread_ref.is_alive()

    @cuda_only
    def test_thread_is_none_after_stop(self):
        monitor = GPUMonitor(device_id=0, sample_interval=0.1)
        monitor.start()
        monitor.stop()
        assert monitor._thread is None

    @cuda_only
    def test_stop_idempotent(self):
        """Calling stop() twice must not raise."""
        monitor = GPUMonitor(device_id=0, sample_interval=0.1)
        monitor.start()
        monitor.stop()
        monitor.stop()  # second call — should be no-op

    @cuda_only
    def test_stop_without_start_is_no_op(self):
        monitor = GPUMonitor(device_id=0, sample_interval=0.1)
        monitor.stop()  # never started — must be a no-op

    @cuda_only
    def test_double_start_raises(self):
        monitor = GPUMonitor(device_id=0, sample_interval=0.1)
        monitor.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                monitor.start()
        finally:
            monitor.stop()


# ===========================================================================
# 10. EXCEPTION-SAFE SHUTDOWN
# ===========================================================================

class TestExceptionSafeShutdown:
    """§26.10 — Monitor stops cleanly even if an exception occurs inside context."""

    @cuda_only
    def test_context_manager_stops_on_exception(self):
        """Monitor thread must stop even if the workload raises."""
        monitor = GPUMonitor(device_id=0, sample_interval=0.1)
        with pytest.raises(ValueError, match="intentional"):
            with monitor:
                time.sleep(0.1)
                raise ValueError("intentional test exception")
        # After exception, the thread must be stopped
        assert monitor._thread is None

    @cuda_only
    def test_gpu_measurement_context_stops_on_exception(self):
        """Convenience context manager must stop on exception."""
        captured_monitor = None
        with pytest.raises(RuntimeError, match="simulated"):
            with gpu_measurement_context(device_id=0, sample_interval=0.1) as mon:
                captured_monitor = mon
                time.sleep(0.1)
                raise RuntimeError("simulated failure")
        assert captured_monitor is not None
        assert captured_monitor._thread is None


# ===========================================================================
# 11. MULTI-GPU DEVICE SELECTION
# ===========================================================================

class TestMultiGPUDeviceSelection:
    """§26.11 — Multi-GPU device selection logic."""

    def test_validate_device_id_0_succeeds_with_mocked_gpu(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=2):
            validate_device_id(0)  # must not raise

    def test_validate_device_id_1_succeeds_with_two_gpus(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=2):
            validate_device_id(1)  # must not raise

    def test_validate_device_id_2_fails_with_two_gpus(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=2):
            with pytest.raises(RuntimeError):
                validate_device_id(2)

    @multi_gpu_only
    def test_real_gpu1_snapshot_has_correct_device_id(self):
        snap = take_gpu_snapshot(1)
        assert snap.device_id == 1

    @multi_gpu_only
    def test_gpu0_and_gpu1_snapshots_independent(self):
        snap0 = take_gpu_snapshot(0)
        snap1 = take_gpu_snapshot(1)
        assert snap0.device_id == 0
        assert snap1.device_id == 1

    def test_list_gpus_returns_independent_identities(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=3):
            # Mock get_gpu_info by mocking get_device_properties for each device
            def mock_props(dev_id):
                m = MagicMock()
                m.name = f"Mock GPU {dev_id}"
                m.total_memory = (dev_id + 1) * 1_000_000_000
                m.major = 8
                m.minor = 6
                return m
            with patch("torch.cuda.get_device_properties", side_effect=mock_props), \
                 patch(
                     "adaptive_sr.benchmarking.gpu_measurement._get_driver_version",
                     return_value=None
                 ):
                result = list_gpus()
        assert len(result) == 3
        for i, gpu in enumerate(result):
            assert gpu.device_id == i


# ===========================================================================
# 12. NO-GPU GRACEFUL BEHAVIOR
# ===========================================================================

class TestNoGPUGracefulBehavior:
    """§26.12 — No-GPU graceful behavior."""

    def test_list_gpus_returns_empty_list_without_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            result = list_gpus()
        assert result == []

    def test_get_cuda_availability_unavailable_status(self):
        with patch("torch.cuda.is_available", return_value=False):
            info = get_cuda_availability()
        assert info["status"] == CUDAAvailability.UNAVAILABLE
        assert info["device_count"] == 0

    def test_gpu_monitor_raises_clear_error_without_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                GPUMonitor(device_id=0)
        assert "not available" in str(exc_info.value).lower() or \
               "unavailable" in str(exc_info.value).lower()

    def test_require_cuda_raises_with_no_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError):
                require_cuda(0)

    def test_validate_device_id_raises_with_no_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError):
                validate_device_id(0)

    def test_nvml_context_available_false_without_importable(self):
        with patch(
            "adaptive_sr.benchmarking.gpu_measurement._NVML_IMPORTABLE",
            False
        ):
            with NVMLContext() as ctx:
                assert ctx.available is False
                assert ctx.pynvml is None


# ===========================================================================
# 13. STEP 5.2 CUDA ADAPTER COMPATIBILITY
# ===========================================================================

class TestStep52AdapterCompatibility:
    """§26.13 — Step 5.2 CUDA adapter compatibility check."""

    def test_tinysr_adapter_is_importable(self):
        from adaptive_sr.benchmarking.adapters.registry import get_adapter
        adapter = get_adapter("tinysr")
        assert adapter is not None

    @cuda_only
    def test_gpu_info_device_name_matches_adapter_cuda_device(self):
        """GPUDeviceInfo name must reflect the same device the adapter would use."""
        gpu_info = get_gpu_info(0)
        # The adapter uses torch's cuda:0; verify device name is obtainable
        props = torch.cuda.get_device_properties(0)
        assert gpu_info.device_name == props.name

    @cuda_only
    def test_tinysr_adapter_available_on_cuda(self):
        """On a CUDA host, FSRCNN adapter should be available for cuda execution."""
        from adaptive_sr.benchmarking.adapters.registry import get_adapter
        adapter = get_adapter("tinysr")
        assert adapter.is_available()

    def test_cuda_availability_check_consistent_with_torch(self):
        """get_cuda_availability() must be consistent with torch.cuda.is_available()."""
        info = get_cuda_availability()
        torch_available = torch.cuda.is_available()
        if torch_available:
            assert info["status"] in (
                CUDAAvailability.AVAILABLE, CUDAAvailability.NO_DEVICE
            )
        else:
            assert info["status"] == CUDAAvailability.UNAVAILABLE


# ===========================================================================
# 14. EXISTING STEP 0–4 REGRESSION MARKER
# ===========================================================================

class TestRegressionMarker:
    """§26.14 — Existing Step 0–4 tests must remain passing.

    This class acts as a sanity import test: if any of the frozen modules
    are accidentally broken by Step 5.4 changes, imports here will fail.
    """

    def test_resource_monitor_importable(self):
        from adaptive_sr.monitoring.resource_monitor import (
            ResourceMonitor, ProcessMonitor
        )
        assert ResourceMonitor is not None
        assert ProcessMonitor is not None

    def test_schemas_importable(self):
        from adaptive_sr.shared.schemas import (
            EdgeResourceTelemetry,
            ProcessResourceSnapshot,
            GPUSnapshot,
        )
        assert GPUSnapshot is not None

    def test_cpu_control_importable(self):
        from adaptive_sr.benchmarking.cpu_control import (
            CPUExecutionConfig,
            cpu_affinity_context,
            BenchmarkProcessMonitor,
            benchmark_execution_context,
        )
        assert CPUExecutionConfig is not None

    def test_adapters_registry_importable(self):
        from adaptive_sr.benchmarking.adapters.registry import (
            get_adapter,
            list_available_models,
        )
        assert get_adapter is not None
        assert list_available_models is not None

    def test_process_monitor_snapshot_returns_valid_schema(self):
        from adaptive_sr.monitoring.resource_monitor import ProcessMonitor
        from adaptive_sr.shared.schemas import ProcessResourceSnapshot
        pm = ProcessMonitor()
        snap = pm.snapshot(interval=0.05)
        assert isinstance(snap, ProcessResourceSnapshot)
        assert snap.cpu_percent >= 0.0
        assert snap.memory_used_bytes > 0


# ===========================================================================
# GPU SMOKE TEST (§27 — only runs if real CUDA GPU is present)
# ===========================================================================

class TestGPUSmokeTest:
    """§27 — Real GPU smoke test.

    Verifies:
        - CUDA device is selectable
        - GPU measurement can start
        - At least one valid snapshot can be obtained
        - Measurement can stop cleanly

    This is NOT a benchmark.  No latency, FPS, or model comparison.
    """

    @cuda_only
    def test_gpu_smoke_test(self):
        """Full lifecycle smoke test on a real CUDA GPU."""
        # 1. Verify device is selectable
        validate_device_id(0)

        # 2. Start measurement
        monitor = GPUMonitor(device_id=0, sample_interval=0.1)
        monitor.start()

        # 3. Run a minimal bounded GPU operation (NOT a benchmark)
        x = torch.ones(64, 64, device="cuda:0")
        y = x + x
        del x, y

        # Allow at least one sample to be collected
        time.sleep(0.25)

        # 4. Stop measurement
        monitor.stop()

        # 5. Verify at least one valid snapshot was obtained
        samples = monitor.get_samples()
        assert len(samples) >= 1

        snap = samples[0]
        assert isinstance(snap, GPUSnapshot)
        assert snap.device_id == 0
        assert isinstance(snap.gpu_name, str) and len(snap.gpu_name) > 0
        assert snap.timestamp.endswith("Z")

        # Memory fields: if populated, must be non-negative integers
        for field_val in [
            snap.gpu_memory_total_bytes,
            snap.gpu_memory_free_bytes,
            snap.process_gpu_memory_allocated_bytes,
            snap.process_gpu_memory_reserved_bytes,
        ]:
            if field_val is not None:
                assert isinstance(field_val, int)
                assert field_val >= 0

        # Utilization: if populated, must be in [0, 100]
        for util_val in [snap.gpu_utilization_percent, snap.memory_utilization_percent]:
            if util_val is not None:
                assert 0.0 <= util_val <= 100.0


# ===========================================================================
# CORRECTION PASS — ADDITIONAL TESTS
# ===========================================================================

# ---------------------------------------------------------------------------
# C1. nvidia-ml-py import / integration
# ---------------------------------------------------------------------------

class TestNvidiaMlPyIntegration:
    """Correction §4 — nvidia-ml-py package import and integration path."""

    def test_nvml_importable_flag_is_bool(self):
        """_NVML_IMPORTABLE must be a bool, set at module import time."""
        assert isinstance(_NVML_IMPORTABLE, bool)

    def test_pynvml_module_importable_via_nvidia_ml_py(self):
        """nvidia-ml-py exposes its API through the pynvml module name."""
        try:
            import pynvml
            # If import succeeds, verify it exposes core NVML functions
            assert hasattr(pynvml, "nvmlInit")
            assert hasattr(pynvml, "nvmlShutdown")
            assert hasattr(pynvml, "nvmlDeviceGetHandleByIndex")
            assert hasattr(pynvml, "nvmlDeviceGetUtilizationRates")
            assert hasattr(pynvml, "nvmlDeviceGetMemoryInfo")
        except ImportError:
            pytest.skip("pynvml (nvidia-ml-py) is not installed in this environment")

    def test_nvml_context_sets_available_correctly(self):
        """NVMLContext.available reflects actual NVML initialisation result."""
        with NVMLContext() as ctx:
            assert isinstance(ctx.available, bool)
            if ctx.available:
                assert ctx.pynvml is not None
            else:
                # unavailable is valid on CPU-only or driver-absent machines
                assert ctx.pynvml is None or not ctx.available

    @cuda_only
    def test_nvml_can_query_device_0_when_gpu_present(self):
        """On a real GPU host, NVML should successfully open device 0."""
        with NVMLContext() as ctx:
            if not ctx.available:
                pytest.skip("NVML not functional on this GPU host (driver issue?)")
            handle = ctx.pynvml.nvmlDeviceGetHandleByIndex(0)
            name = ctx.pynvml.nvmlDeviceGetName(handle)
            assert isinstance(name, str)
            assert len(name) > 0


# ---------------------------------------------------------------------------
# C2. Synchronous snapshot mode (Mode B)
# ---------------------------------------------------------------------------

class TestSynchronousSnapshotMode:
    """Correction §1/§2 — Mode B synchronous snapshot retrieval and semantics."""

    @cuda_only
    def test_take_gpu_snapshot_returns_gpu_snapshot(self):
        snap = take_gpu_snapshot(0)
        assert isinstance(snap, GPUSnapshot)

    @cuda_only
    def test_snapshot_is_immediate_not_periodic(self):
        """take_gpu_snapshot() is a synchronous call — it returns immediately."""
        t_start = time.monotonic()
        snap = take_gpu_snapshot(0)
        elapsed = time.monotonic() - t_start
        assert isinstance(snap, GPUSnapshot)
        # Should complete in well under 1 second (no periodic wait)
        assert elapsed < 1.0

    @cuda_only
    def test_two_snapshots_have_increasing_timestamps(self):
        """Consecutive snapshots must have non-decreasing timestamps."""
        snap1 = take_gpu_snapshot(0)
        time.sleep(0.05)
        snap2 = take_gpu_snapshot(0)
        assert snap2.timestamp >= snap1.timestamp

    @cuda_only
    def test_snapshot_memory_fields_are_integers_or_none(self):
        snap = take_gpu_snapshot(0)
        for val in [
            snap.gpu_memory_total_bytes,
            snap.gpu_memory_free_bytes,
            snap.process_gpu_memory_allocated_bytes,
            snap.process_gpu_memory_reserved_bytes,
        ]:
            assert val is None or (isinstance(val, int) and val >= 0)

    @cuda_only
    def test_before_after_snapshot_pattern(self):
        """Demonstrates the synchronous before/after snapshot boundary pattern."""
        before = take_gpu_snapshot(0)
        # Allocate a small tensor to change memory state
        t = torch.zeros(1024, 1024, device="cuda:0")
        after = take_gpu_snapshot(0)
        del t

        assert isinstance(before, GPUSnapshot)
        assert isinstance(after, GPUSnapshot)
        assert before.device_id == after.device_id == 0
        # Both snapshots must be structurally valid
        assert before.timestamp.endswith("Z")
        assert after.timestamp.endswith("Z")


# ---------------------------------------------------------------------------
# C3. GPUMeasurementBoundary schema and memory deltas
# ---------------------------------------------------------------------------

class TestGPUMeasurementBoundary:
    """Correction §1/§2/§5 — GPUMeasurementBoundary schema validity."""

    def _make_snapshot(self, allocated: Optional[int], reserved: Optional[int],
                       free: Optional[int] = None, util: Optional[float] = None) -> GPUSnapshot:
        return GPUSnapshot(
            timestamp="2026-01-01T00:00:00.000000Z",
            device_id=0,
            gpu_name="Test GPU",
            gpu_utilization_percent=util,
            memory_utilization_percent=None,
            gpu_memory_total_bytes=8_000_000_000,
            gpu_memory_free_bytes=free,
            process_gpu_memory_allocated_bytes=allocated,
            process_gpu_memory_reserved_bytes=reserved,
            nvml_available=(util is not None),
            utilization_source="nvml" if util is not None else "pytorch_allocator",
        )

    def test_boundary_creation(self):
        before = self._make_snapshot(allocated=0, reserved=0)
        after = self._make_snapshot(allocated=1_000_000, reserved=2_000_000)
        b = GPUMeasurementBoundary(before=before, after=after)
        assert isinstance(b, GPUMeasurementBoundary)

    def test_memory_allocated_delta_positive(self):
        """Delta is positive when memory grows."""
        before = self._make_snapshot(allocated=0, reserved=0)
        after = self._make_snapshot(allocated=1_000_000, reserved=2_000_000)
        b = GPUMeasurementBoundary(before=before, after=after)
        assert b.memory_allocated_delta_bytes == 1_000_000

    def test_memory_allocated_delta_negative_on_free(self):
        """Delta is negative when memory is freed."""
        before = self._make_snapshot(allocated=1_000_000, reserved=2_000_000)
        after = self._make_snapshot(allocated=0, reserved=2_000_000)
        b = GPUMeasurementBoundary(before=before, after=after)
        assert b.memory_allocated_delta_bytes == -1_000_000

    def test_memory_reserved_delta(self):
        before = self._make_snapshot(allocated=0, reserved=0)
        after = self._make_snapshot(allocated=0, reserved=4_000_000)
        b = GPUMeasurementBoundary(before=before, after=after)
        assert b.memory_reserved_delta_bytes == 4_000_000

    def test_free_memory_delta_with_nvml(self):
        before = self._make_snapshot(allocated=0, reserved=0, free=7_000_000_000)
        after = self._make_snapshot(allocated=1_000_000, reserved=0, free=6_999_000_000)
        b = GPUMeasurementBoundary(before=before, after=after)
        assert b.free_memory_delta_bytes == -1_000_000

    def test_delta_is_none_when_field_absent(self):
        """Delta must be None when either snapshot lacks the field."""
        before = self._make_snapshot(allocated=None, reserved=None)
        after = self._make_snapshot(allocated=1_000_000, reserved=2_000_000)
        b = GPUMeasurementBoundary(before=before, after=after)
        assert b.memory_allocated_delta_bytes is None
        assert b.memory_reserved_delta_bytes is None

    def test_boundary_is_frozen(self):
        before = self._make_snapshot(allocated=0, reserved=0)
        after = self._make_snapshot(allocated=0, reserved=0)
        b = GPUMeasurementBoundary(before=before, after=after)
        with pytest.raises((AttributeError, TypeError)):
            b.before = after  # type: ignore

    def test_operation_label_stored(self):
        before = self._make_snapshot(allocated=0, reserved=0)
        after = self._make_snapshot(allocated=0, reserved=0)
        b = GPUMeasurementBoundary(before=before, after=after, operation_label="load_model")
        assert b.operation_label == "load_model"

    def test_utilization_in_boundary_is_instantaneous_not_peak(self):
        """Explicit structural check: utilization in before/after != peak inference util.

        The boundary before/after util values reflect instantaneous device state,
        not peak load during the bounded operation.  This test documents that
        constraint by verifying the field names and None handling are correct.
        """
        before = self._make_snapshot(allocated=0, reserved=0, util=5.0)
        after = self._make_snapshot(allocated=0, reserved=0, util=2.0)
        b = GPUMeasurementBoundary(before=before, after=after)
        # Neither before nor after utilization = peak during operation
        # We only assert type/range, NOT that they equal inference load
        for u in [b.before.gpu_utilization_percent, b.after.gpu_utilization_percent]:
            if u is not None:
                assert 0.0 <= u <= 100.0


# ---------------------------------------------------------------------------
# C4. capture_gpu_boundary context manager (Mode B)
# ---------------------------------------------------------------------------

class TestCaptureGpuBoundary:
    """Correction §1 — capture_gpu_boundary context manager."""

    @cuda_only
    def test_capture_boundary_produces_boundary_object(self):
        with capture_gpu_boundary(0, "test_op") as holder:
            x = torch.zeros(256, device="cuda:0")
            del x
        assert holder.result is not None
        assert isinstance(holder.result, GPUMeasurementBoundary)

    @cuda_only
    def test_capture_boundary_before_after_device_ids(self):
        with capture_gpu_boundary(0) as holder:
            pass
        b = holder.result
        assert b.before.device_id == 0
        assert b.after.device_id == 0

    @cuda_only
    def test_capture_boundary_stops_on_exception(self):
        """After snapshot must be taken even when exception occurs."""
        holder_ref = None
        with pytest.raises(ValueError, match="boundary_test"):
            with capture_gpu_boundary(0, "exc_test") as holder:
                holder_ref = holder
                raise ValueError("boundary_test")
        assert holder_ref is not None
        assert holder_ref.result is not None
        assert isinstance(holder_ref.result, GPUMeasurementBoundary)

    @cuda_only
    def test_capture_boundary_operation_label(self):
        with capture_gpu_boundary(0, "model_forward") as holder:
            pass
        assert holder.result.operation_label == "model_forward"

    @cuda_only
    def test_capture_boundary_no_timing_in_result(self):
        """GPUMeasurementBoundary must not contain any latency/timing fields."""
        with capture_gpu_boundary(0) as holder:
            pass
        b = holder.result
        # Structural check: no latency/timing attribute should exist
        assert not hasattr(b, "latency_ms")
        assert not hasattr(b, "elapsed_seconds")
        assert not hasattr(b, "duration")
        assert not hasattr(b, "throughput")
        assert not hasattr(b, "fps")


# ---------------------------------------------------------------------------
# C5. Measurement mode distinctions
# ---------------------------------------------------------------------------

class TestMeasurementModeDistinctions:
    """Correction §1/§2 — explicit distinction between Mode A and Mode B."""

    def test_utilization_fields_distinct_from_memory_fields(self):
        """Utilization and memory fields are independent — cannot be interchanged."""
        snap = GPUSnapshot(
            timestamp="2026-01-01T00:00:00.000000Z",
            device_id=0,
            gpu_name="Test",
            gpu_utilization_percent=80.0,       # device-wide SM utilization
            memory_utilization_percent=50.0,    # device-wide memory bus util
            gpu_memory_total_bytes=8_000_000_000,
            gpu_memory_free_bytes=4_000_000_000,
            process_gpu_memory_allocated_bytes=100_000,   # process-level
            process_gpu_memory_reserved_bytes=200_000,    # process-level
            nvml_available=True,
            utilization_source="nvml",
        )
        # All four memory quantities must be independently readable
        assert snap.gpu_memory_total_bytes != snap.process_gpu_memory_allocated_bytes
        assert snap.gpu_utilization_percent != snap.memory_utilization_percent  # may differ
        # NVML utilization fields are device-wide
        assert snap.utilization_source == "nvml"
        assert snap.nvml_available is True
        # Process-level fields are separate from device-wide
        assert snap.process_gpu_memory_allocated_bytes < snap.gpu_memory_total_bytes

    def test_reserved_always_gte_allocated_when_both_present(self):
        """PyTorch caching allocator invariant: reserved >= allocated."""
        snap = GPUSnapshot(
            timestamp="2026-01-01T00:00:00.000000Z",
            device_id=0,
            gpu_name="Test",
            process_gpu_memory_allocated_bytes=100_000,
            process_gpu_memory_reserved_bytes=500_000,
            nvml_available=False,
            utilization_source="pytorch_allocator",
        )
        assert snap.process_gpu_memory_reserved_bytes >= snap.process_gpu_memory_allocated_bytes

    def test_mode_a_monitor_exists_and_mode_b_snapshot_exists(self):
        """Both measurement modes must be independently importable."""
        assert callable(take_gpu_snapshot)         # Mode B primitive
        assert callable(GPUMeasurementBoundary)    # Mode B boundary schema
        assert callable(capture_gpu_boundary)      # Mode B context manager
        assert callable(GPUMonitor)                # Mode A periodic sampler
        assert callable(gpu_measurement_context)   # Mode A context manager

    def test_nvml_utilization_none_does_not_contaminate_memory_fields(self):
        """When NVML is absent, memory fields (from PyTorch) must still be populated."""
        snap = GPUSnapshot(
            timestamp="2026-01-01T00:00:00.000000Z",
            device_id=0,
            gpu_name="Test",
            gpu_utilization_percent=None,    # NVML absent
            memory_utilization_percent=None, # NVML absent
            gpu_memory_total_bytes=8_000_000_000,
            process_gpu_memory_allocated_bytes=50_000,
            process_gpu_memory_reserved_bytes=100_000,
            nvml_available=False,
            utilization_source="pytorch_allocator",
        )
        # Utilization absent but memory still populated
        assert snap.gpu_utilization_percent is None
        assert snap.process_gpu_memory_allocated_bytes is not None
        assert snap.gpu_memory_total_bytes is not None


# ---------------------------------------------------------------------------
# C6. CPU / GPU sampling interval independence
# ---------------------------------------------------------------------------

class TestCPUGPUSamplingIntervalIndependence:
    """Correction §3 — CPU and GPU sampling intervals are independently configurable."""

    @cuda_only
    def test_gpu_monitor_has_configurable_interval(self):
        monitor = GPUMonitor(device_id=0, sample_interval=0.3)
        assert monitor.sample_interval == 0.3

    @cuda_only
    def test_gpu_default_interval_is_independent_of_cpu_default(self):
        """GPU default is 0.5s; CPU default (BenchmarkProcessMonitor) is 0.05s.
        These are explicitly different and independently justified."""
        from adaptive_sr.benchmarking.cpu_control import BenchmarkProcessMonitor
        cpu_monitor = BenchmarkProcessMonitor()
        gpu_monitor = GPUMonitor(device_id=0)
        # Deliberately different defaults
        assert gpu_monitor.sample_interval != cpu_monitor.sample_interval
        # GPU default
        assert gpu_monitor.sample_interval == 0.5
        # CPU default
        assert cpu_monitor.sample_interval == 0.05

    @cuda_only
    def test_can_set_gpu_interval_independent_of_cpu(self):
        """GPU monitor interval can be changed without affecting CPU monitor."""
        from adaptive_sr.benchmarking.cpu_control import BenchmarkProcessMonitor
        gpu_monitor = GPUMonitor(device_id=0, sample_interval=0.2)
        cpu_monitor = BenchmarkProcessMonitor(sample_interval=0.1)
        assert gpu_monitor.sample_interval == 0.2
        assert cpu_monitor.sample_interval == 0.1
        # They are independent
        assert gpu_monitor.sample_interval != cpu_monitor.sample_interval


# ---------------------------------------------------------------------------
# C7. Step 5.3 regression marker
# ---------------------------------------------------------------------------

class TestStep53Regression:
    """Correction §8 — Step 5.3 tests must remain passing."""

    def test_cpu_control_still_importable(self):
        from adaptive_sr.benchmarking.cpu_control import (
            CPUExecutionConfig,
            cpu_affinity_context,
            BenchmarkProcessMonitor,
            benchmark_execution_context,
            get_available_cpus,
            select_cpu_ids,
        )
        assert CPUExecutionConfig is not None
        assert BenchmarkProcessMonitor is not None

    def test_cpu_monitor_unaffected_by_gpu_module_import(self):
        """Importing gpu_measurement must not alter Step 5.3 behavior."""
        from adaptive_sr.benchmarking.cpu_control import BenchmarkProcessMonitor
        from adaptive_sr.monitoring.resource_monitor import ProcessMonitor
        pm = ProcessMonitor()
        snap = pm.snapshot(interval=0.05)
        assert snap.cpu_percent >= 0.0
        assert snap.memory_used_bytes > 0

    def test_cpu_affinity_context_still_works(self):
        import psutil
        from adaptive_sr.benchmarking.cpu_control import cpu_affinity_context, get_available_cpus
        cpus = get_available_cpus()
        original = psutil.Process().cpu_affinity()
        with cpu_affinity_context([cpus[0]]):
            active = psutil.Process().cpu_affinity()
            assert active == [cpus[0]]
        restored = psutil.Process().cpu_affinity()
        assert restored == original
