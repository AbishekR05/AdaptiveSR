"""
adaptive_sr.benchmarking.gpu_measurement
=========================================
Step 5.4 — GPU Measurement Infrastructure.

PURPOSE
-------
Establishes the GPU-side observability layer needed to compare SR inference
under GPU execution in later benchmark steps.

This module is the GPU equivalent of Step 5.3 cpu_control.py:

    cpu_control.py          →   CPU affinity + ProcessMonitor integration
    gpu_measurement.py      →   GPU discovery + identity + memory + utilization

SCOPE
-----
This module PROVIDES:
    - CUDA availability detection
    - GPU device discovery (list_gpus)
    - GPU device identity metadata (GPUDeviceInfo)
    - GPU memory measurement (device-wide and process-level)
    - GPU utilization sampling via NVML where available
    - GPUSnapshot structured observation schema
    - GPUMeasurementBoundary: before/after snapshot pair (Mode B)
    - GPUMonitor: background periodic sampling lifecycle (Mode A)
    - gpu_measurement_context() context manager

This module does NOT implement:
    - Inference timing or latency measurement
    - CUDA timing events or torch.cuda.synchronize() around benchmark calls
    - FPS / throughput benchmarking
    - Warmup loops or repeated benchmark trials
    - PSNR, SSIM, LPIPS, VMAF
    - Real-time feasibility analysis
    - Benchmark result datasets
    - GPU resource allocation or multi-GPU load balancing
    - Dynamic GPU scheduling or MIG partitioning

============================================================
GPU MEASUREMENT MODES
============================================================

Two distinct measurement modes are provided.  They serve different purposes
and MUST NOT be confused.

MODE A — PERIODIC MONITORING (GPUMonitor)
------------------------------------------
A background thread calls take_gpu_snapshot() at a configurable interval.

Purpose:
    Sustained workload characterisation.
    Aggregate utilisation observation over longer-running or repeated
    inference workloads (Step 5.5+).

Limitation:
    Periodic sampling may MISS a short-lived SR inference entirely if the
    inference completes faster than the sampling interval.

    Therefore: periodic utilisation samples MUST NOT be treated as exact
    per-operation utilisation measurements.

    Periodic sampling is appropriate for characterising sustained GPU load
    across a batch of inferences, not a single short call.

Default interval: 0.5 s (conservative; avoids significant overhead).
The Step 5.5 benchmark harness may choose a shorter interval if needed.

MODE B — SYNCHRONOUS SNAPSHOTS (take_gpu_snapshot / GPUMeasurementBoundary)
-----------------------------------------------------------------------------
A single point-in-time GPU state capture via take_gpu_snapshot().

Purpose:
    Operation-boundary GPU memory/state observation.
    Capturing before/after GPU memory state around a bounded operation.

What a before/after snapshot pair CAN reliably provide:
    - Allocated GPU memory delta (process-level PyTorch allocator)
    - Reserved GPU memory delta (process-level PyTorch allocator)
    - Device-wide free memory change
    - Device-wide total VRAM (constant)

What a before/after snapshot pair CANNOT reliably provide:
    - Peak GPU utilisation during a short inference.
    - NVML utilisation is sampled at a point in time; a snapshot taken
      immediately after a kernel completes will NOT reflect the peak
      utilisation that occurred during execution.

Therefore:
    synchronous before/after snapshots = per-operation memory/state boundaries
    periodic utilisation samples        = sustained utilisation characterisation

    NEITHER is a substitute for explicit inference latency measurement.
    Step 5.5 must use torch.cuda.synchronize() around explicit timing
    boundaries for accurate latency.

GPUMeasurementBoundary encapsulates a before/after snapshot pair and
exposes memory delta helpers.  It does NOT compute or report timing.

============================================================
CPU / GPU SAMPLING INTERVAL POLICY
============================================================

CPU telemetry (Step 5.3 BenchmarkProcessMonitor) and GPU telemetry
(Step 5.4 GPUMonitor) operate with INDEPENDENTLY configurable sampling
intervals.

Rationale:
    CPU and GPU telemetry have different collection mechanisms and different
    overhead characteristics:

    - CPU measurement (psutil.Process.cpu_percent) uses a blocking OS call
      with a configurable interval (default: 50 ms in Step 5.3).
    - GPU measurement (NVML nvmlDeviceGetUtilizationRates + PyTorch allocator)
      has different latency and overhead characteristics.

    Forcing identical intervals would artificially couple two independent
    subsystems with different measurement semantics.

Therefore:
    - Step 5.3 CPU monitor interval and Step 5.4 GPU monitor interval MAY
      differ.
    - CPU and GPU utilisation curves MUST NOT be assumed to have identical
      temporal resolution merely because both are labelled "utilisation".
    - The primary latency measurement in Step 5.5 MUST be performed using
      explicit timing boundaries around inference, NOT inferred from either
      monitor's sampling interval.

============================================================
IMPORTANT — CUDA ASYNCHRONOUS EXECUTION NOTE
============================================================
CUDA operations are asynchronous: the CPU-side call to a model's forward()
method may return before the GPU has finished executing the operation.

Therefore, future benchmark timing in Step 5.5 MUST use explicit CUDA
synchronization (torch.cuda.synchronize()) around timing boundaries.

Step 5.4 does NOT implement benchmark timing and does NOT need to call
synchronize().  This note is here for Step 5.5 implementers.

============================================================
IMPORTANT — GPU WARMUP NOTE
============================================================
Future benchmark execution in Step 5.5 MUST distinguish:
    - CUDA context creation overhead  (first ever CUDA call in a process)
    - Model loading overhead           (weight transfer to VRAM)
    - First-inference overhead         (JIT compilation, cache misses)
    - Warmed-up inference              (steady-state execution)

Step 5.4 provides measurement infrastructure only; warmup is NOT implemented.

============================================================
IMPORTANT — MEASUREMENT OVERHEAD
============================================================
GPU monitoring via NVML or the PyTorch allocator introduces non-zero overhead.
This overhead is NOT subtracted from measurements.  To maintain comparability:
    - Use the same GPUMonitor configuration across all comparable benchmark runs.
    - Use a conservative sampling interval (default: 0.5 s).
    - Document the sampling configuration in the benchmark record.

============================================================
NVIDIA-ML-PY vs. pynvml — PACKAGE NOTE
============================================================
This module uses the pynvml API, which is provided by the maintained
'nvidia-ml-py' package (listed in requirements.txt).

The legacy 'pynvml' package (pypi: pynvml) is deprecated.  Both packages
expose the same 'pynvml' Python module name and identical API.  The migration
from pynvml to nvidia-ml-py requires only a requirements.txt change; all
'import pynvml' statements remain valid.

Source: PyTorch FutureWarning in torch/cuda/__init__.py recommending
nvidia-ml-py over pynvml.

============================================================
NVML vs. PyTorch allocator — what each provides
============================================================
Source                       What it measures
---------------------------------------------------------------------------
nvidia-ml-py (NVML)          Device-wide GPU utilization %, free/total VRAM
torch.cuda.memory_allocated  Process-level: bytes used by live tensors
torch.cuda.memory_reserved   Process-level: bytes held by caching allocator

These are DIFFERENT quantities and MUST NOT be interchanged.
See GPUSnapshot docstring in schemas.py for full details.

============================================================
PROCESS vs. DEVICE GPU MEMORY
============================================================
NVML reports memory for the ENTIRE device across all processes.
PyTorch allocator reports memory for THIS process only.

The GPUSnapshot schema distinguishes them with explicit field names:
    gpu_memory_*             → device-wide (NVML source)
    process_gpu_memory_*     → this-process only (PyTorch allocator source)

Do not conflate these values.

============================================================
NO-GPU ENVIRONMENTS
============================================================
This module is designed to function on machines without CUDA hardware.
On a CPU-only machine:
    - get_cuda_availability()  returns CUDAAvailability.UNAVAILABLE
    - list_gpus()              returns []
    - GPUMonitor(device_id=0)  raises RuntimeError with a clear message
    - Tests skip GPU-specific assertions with pytest.mark.skipif
"""

from __future__ import annotations

import time
import threading
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Generator

import torch

from adaptive_sr.shared.schemas import GPUSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NVML AVAILABILITY CHECK  (nvidia-ml-py package, pynvml module)
# ---------------------------------------------------------------------------

def _check_nvml_importable() -> bool:
    """Returns True if the pynvml module is importable.

    The pynvml module is provided by the 'nvidia-ml-py' package (requirements.txt).
    This check does NOT mean NVML will successfully initialise — NVIDIA drivers
    must also be present and functional.
    """
    try:
        import pynvml  # noqa: F401  — provided by nvidia-ml-py
        return True
    except ImportError:
        return False


_NVML_IMPORTABLE: bool = _check_nvml_importable()


# ---------------------------------------------------------------------------
# NVML SAFE CONTEXT MANAGER
# ---------------------------------------------------------------------------

class NVMLContext:
    """Context manager that initialises and shuts down the NVML library safely.

    Uses the pynvml module provided by the 'nvidia-ml-py' package.

    On environments without NVIDIA hardware or drivers, initialisation will
    fail.  The context manager catches this and sets ``available=False`` so
    callers can branch on it rather than catching exceptions themselves.

    Usage
    -----
    with NVMLContext() as nvml_ctx:
        if nvml_ctx.available:
            handle = nvml_ctx.pynvml.nvmlDeviceGetHandleByIndex(0)
            ...
    """

    def __init__(self) -> None:
        self.available: bool = False
        self.pynvml = None

    def __enter__(self) -> "NVMLContext":
        if not _NVML_IMPORTABLE:
            return self
        try:
            import pynvml  # provided by nvidia-ml-py
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.available = True
        except Exception as exc:
            logger.debug("NVML initialisation failed (not an error on CPU-only hosts): %s", exc)
            self.available = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.available and self.pynvml is not None:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass
        # Do not suppress exceptions
        return False


# ---------------------------------------------------------------------------
# CUDA AVAILABILITY
# ---------------------------------------------------------------------------

class CUDAAvailability(Enum):
    """Describes the state of CUDA on the current host.

    UNAVAILABLE  — torch.cuda.is_available() is False.  No CUDA runtime.
    NO_DEVICE    — CUDA runtime present but no device is visible.
    AVAILABLE    — At least one CUDA device is present and usable.
    """
    UNAVAILABLE = "unavailable"
    NO_DEVICE   = "no_device"
    AVAILABLE   = "available"


def get_cuda_availability() -> dict:
    """Detects CUDA availability and enumerates visible devices.

    Returns
    -------
    dict with keys:
        status      : CUDAAvailability
        device_count: int
        device_ids  : List[int]

    Notes
    -----
    This function does NOT fall back silently from CUDA to CPU.
    If callers request CUDA execution, they should check status first
    and raise an error if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return {
            "status": CUDAAvailability.UNAVAILABLE,
            "device_count": 0,
            "device_ids": [],
        }
    count = torch.cuda.device_count()
    if count == 0:
        return {
            "status": CUDAAvailability.NO_DEVICE,
            "device_count": 0,
            "device_ids": [],
        }
    return {
        "status": CUDAAvailability.AVAILABLE,
        "device_count": count,
        "device_ids": list(range(count)),
    }


def require_cuda(device_id: int = 0) -> None:
    """Raises RuntimeError with a clear message if CUDA or the requested device
    is unavailable.

    Parameters
    ----------
    device_id : int
        The CUDA device index being requested.

    Raises
    ------
    RuntimeError
        If CUDA is not available or the device_id is out of range.
    """
    info = get_cuda_availability()
    if info["status"] == CUDAAvailability.UNAVAILABLE:
        raise RuntimeError(
            "CUDA is not available on this host.  "
            "Ensure NVIDIA drivers and a CUDA-capable GPU are installed."
        )
    if info["status"] == CUDAAvailability.NO_DEVICE:
        raise RuntimeError(
            "CUDA runtime is present but no CUDA device is visible."
        )
    count = info["device_count"]
    if device_id >= count:
        raise RuntimeError(
            f"Requested CUDA device {device_id}, but only "
            f"{'device 0 is' if count == 1 else f'devices 0–{count - 1} are'} available."
        )


# ---------------------------------------------------------------------------
# GPU DEVICE IDENTITY
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPUDeviceInfo:
    """Stable identity metadata for a single CUDA device.

    Used to uniquely identify GPUs across benchmark records.  The combination
    of (device_id, device_name, total_memory_bytes) is the minimum required
    identity.  Optional fields are populated where the backend exposes them.

    A benchmark record MUST NOT identify a GPU only as "GPU" because multiple
    devices may exist.
    """
    device_id: int
    device_name: str
    total_memory_bytes: int
    compute_capability: Optional[str] = None
    cuda_runtime_version: Optional[str] = None
    pytorch_cuda_version: Optional[str] = None
    driver_version: Optional[str] = None


def _get_pytorch_cuda_version() -> Optional[str]:
    """Returns the PyTorch CUDA build version or None."""
    v = getattr(torch.version, "cuda", None)
    return str(v) if v else None


def _get_driver_version() -> Optional[str]:
    """Returns the NVIDIA driver version from NVML (nvidia-ml-py), or None."""
    with NVMLContext() as ctx:
        if not ctx.available:
            return None
        try:
            return ctx.pynvml.nvmlSystemGetDriverVersion()
        except Exception:
            return None


def get_gpu_info(device_id: int) -> GPUDeviceInfo:
    """Returns stable identity metadata for a CUDA device.

    Parameters
    ----------
    device_id : int
        The CUDA device index to query.

    Raises
    ------
    RuntimeError
        If CUDA is unavailable.
    ValueError
        If device_id is negative or out of range.
    """
    validate_device_id(device_id)

    props = torch.cuda.get_device_properties(device_id)
    device_name = props.name
    total_memory_bytes = props.total_memory
    compute_capability = f"{props.major}.{props.minor}"
    pytorch_cuda_version = _get_pytorch_cuda_version()
    driver_version = _get_driver_version()

    return GPUDeviceInfo(
        device_id=device_id,
        device_name=device_name,
        total_memory_bytes=total_memory_bytes,
        compute_capability=compute_capability,
        cuda_runtime_version=pytorch_cuda_version,
        pytorch_cuda_version=pytorch_cuda_version,
        driver_version=driver_version,
    )


def list_gpus() -> List[GPUDeviceInfo]:
    """Enumerates all visible CUDA devices and returns their identity metadata.

    Returns
    -------
    List[GPUDeviceInfo]
        One entry per visible CUDA device.  Returns an empty list if CUDA is
        unavailable — this is NOT an error condition.
    """
    info = get_cuda_availability()
    if info["status"] != CUDAAvailability.AVAILABLE:
        return []
    result: List[GPUDeviceInfo] = []
    for dev_id in info["device_ids"]:
        try:
            result.append(get_gpu_info(dev_id))
        except Exception as exc:
            logger.warning("Failed to query GPU device %d metadata: %s", dev_id, exc)
    return result


# ---------------------------------------------------------------------------
# DEVICE VALIDATION
# ---------------------------------------------------------------------------

def validate_device_id(device_id: int) -> None:
    """Validates that device_id is a legal and available CUDA device index.

    Raises
    ------
    ValueError
        If device_id is negative.
    RuntimeError
        If CUDA is unavailable or device_id exceeds the discovered range.
    """
    if device_id < 0:
        raise ValueError(
            f"GPU device_id must be non-negative, got: {device_id}"
        )
    require_cuda(device_id)


# ---------------------------------------------------------------------------
# MODE B — SYNCHRONOUS SNAPSHOT
# ---------------------------------------------------------------------------

def take_gpu_snapshot(device_id: int) -> GPUSnapshot:
    """Captures a single point-in-time GPU measurement (Mode B — synchronous).

    This is the primitive for synchronous snapshot mode.  Callers invoke it
    explicitly before and after a bounded operation to observe GPU memory/state
    changes across that boundary.

    What this function reliably captures
    -------------------------------------
    - process_gpu_memory_allocated_bytes : process-level tensor allocations
    - process_gpu_memory_reserved_bytes  : process-level allocator reservation
    - gpu_memory_free_bytes              : device-wide free VRAM (NVML)
    - gpu_memory_total_bytes             : device-wide total VRAM

    What this function does NOT reliably capture
    ---------------------------------------------
    - Peak GPU utilisation during a short inference.
      NVML utilisation is sampled instantaneously.  If the GPU kernel has
      already completed by the time this call executes, the reported
      utilisation may be 0% even if peak utilisation was 100% during the
      kernel.  Do NOT use a post-inference snapshot utilisation value as
      a proxy for inference-time GPU load.

    For accurate utilisation over a workload, use GPUMonitor (Mode A).
    For accurate inference latency, use explicit timing in Step 5.5.

    Parameters
    ----------
    device_id : int
        CUDA device index to measure.

    Returns
    -------
    GPUSnapshot
        Structured observation.  Fields that cannot be obtained are None (not 0).

    Notes
    -----
    - Does NOT synchronize CUDA streams.
    - Does NOT measure inference latency.
    - Does NOT fake any values; missing data is explicitly None.
    - Provided by nvidia-ml-py for NVML metrics.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

    # GPU name from PyTorch (always available when CUDA is up)
    try:
        props = torch.cuda.get_device_properties(device_id)
        gpu_name = props.name
        gpu_memory_total_bytes: Optional[int] = props.total_memory
    except Exception as exc:
        logger.debug("Could not read GPU properties for device %d: %s", device_id, exc)
        gpu_name = f"cuda:{device_id}"
        gpu_memory_total_bytes = None

    # NVML — device-wide metrics (nvidia-ml-py)
    gpu_utilization_percent: Optional[float] = None
    memory_utilization_percent: Optional[float] = None
    gpu_memory_free_bytes: Optional[int] = None
    nvml_available: bool = False
    utilization_source: str = "unavailable"

    with NVMLContext() as ctx:
        if ctx.available:
            try:
                handle = ctx.pynvml.nvmlDeviceGetHandleByIndex(device_id)
                util = ctx.pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization_percent = float(util.gpu)
                memory_utilization_percent = float(util.memory)
                mem_info = ctx.pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_total_bytes = int(mem_info.total)
                gpu_memory_free_bytes = int(mem_info.free)
                nvml_available = True
                utilization_source = "nvml"
            except Exception as exc:
                logger.debug(
                    "NVML query for device %d failed (non-fatal): %s", device_id, exc
                )
                nvml_available = False

    # PyTorch allocator — process-level memory
    process_gpu_memory_allocated_bytes: Optional[int] = None
    process_gpu_memory_reserved_bytes: Optional[int] = None

    try:
        process_gpu_memory_allocated_bytes = int(torch.cuda.memory_allocated(device_id))
        process_gpu_memory_reserved_bytes = int(torch.cuda.memory_reserved(device_id))
        if not nvml_available and utilization_source == "unavailable":
            utilization_source = "pytorch_allocator"
    except Exception as exc:
        logger.debug(
            "PyTorch allocator query for device %d failed: %s", device_id, exc
        )

    return GPUSnapshot(
        timestamp=ts,
        device_id=device_id,
        gpu_name=gpu_name,
        gpu_utilization_percent=gpu_utilization_percent,
        memory_utilization_percent=memory_utilization_percent,
        gpu_memory_total_bytes=gpu_memory_total_bytes,
        gpu_memory_free_bytes=gpu_memory_free_bytes,
        process_gpu_memory_allocated_bytes=process_gpu_memory_allocated_bytes,
        process_gpu_memory_reserved_bytes=process_gpu_memory_reserved_bytes,
        nvml_available=nvml_available,
        utilization_source=utilization_source,
    )


# ---------------------------------------------------------------------------
# GPU MEASUREMENT BOUNDARY — before/after snapshot pair (Mode B)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPUMeasurementBoundary:
    """A before/after pair of GPU snapshots bracketing a bounded operation.

    Captures the GPU memory/state change across a single operation boundary.

    This is the Mode B (synchronous) measurement abstraction.

    IMPORTANT LIMITATIONS
    ---------------------
    Memory deltas (allocated/reserved/free) CAN be reliably observed from a
    before/after snapshot pair.

    GPU utilisation reported in 'before' and 'after' snapshots reflects the
    instantaneous device state at those two moments — NOT the peak utilisation
    during the operation.

    If the bounded operation completes very quickly (shorter than NVML's
    internal sampling window), the post-snapshot utilisation may be 0% even
    if the GPU was fully saturated during the operation.

    Therefore:
        before.gpu_utilization_percent  → state before operation
        after.gpu_utilization_percent   → state after operation
        NEITHER = peak utilisation during the operation

    For peak/sustained utilisation observation, use GPUMonitor (Mode A).
    For inference latency measurement, use Step 5.5 with explicit timing.

    Fields
    ------
    before : GPUSnapshot
        GPU state captured immediately before the bounded operation.
    after : GPUSnapshot
        GPU state captured immediately after the bounded operation.
    operation_label : str
        A human-readable label for the bounded operation.
        Used for benchmark record identification.  Must NOT encode timing.
    """
    before: GPUSnapshot
    after: GPUSnapshot
    operation_label: str = "unnamed_operation"

    @property
    def memory_allocated_delta_bytes(self) -> Optional[int]:
        """Change in process-level allocated GPU memory across the boundary.

        Positive = more memory allocated after the operation.
        Negative = memory freed during the operation.
        None if either snapshot could not obtain the allocator value.
        """
        if (
            self.before.process_gpu_memory_allocated_bytes is not None
            and self.after.process_gpu_memory_allocated_bytes is not None
        ):
            return (
                self.after.process_gpu_memory_allocated_bytes
                - self.before.process_gpu_memory_allocated_bytes
            )
        return None

    @property
    def memory_reserved_delta_bytes(self) -> Optional[int]:
        """Change in process-level reserved GPU memory across the boundary.

        Positive = allocator reserved more memory from the OS.
        Negative = allocator returned memory to the OS.
        None if either snapshot could not obtain the value.
        """
        if (
            self.before.process_gpu_memory_reserved_bytes is not None
            and self.after.process_gpu_memory_reserved_bytes is not None
        ):
            return (
                self.after.process_gpu_memory_reserved_bytes
                - self.before.process_gpu_memory_reserved_bytes
            )
        return None

    @property
    def free_memory_delta_bytes(self) -> Optional[int]:
        """Change in device-wide free GPU memory across the boundary (NVML).

        Negative = free memory decreased (device consumed more VRAM).
        Positive = free memory increased (VRAM returned to device pool).
        None if NVML was unavailable for either snapshot.
        """
        if (
            self.before.gpu_memory_free_bytes is not None
            and self.after.gpu_memory_free_bytes is not None
        ):
            return (
                self.after.gpu_memory_free_bytes
                - self.before.gpu_memory_free_bytes
            )
        return None


@contextmanager
def capture_gpu_boundary(
    device_id: int,
    operation_label: str = "unnamed_operation",
) -> Generator[None, None, GPUMeasurementBoundary]:
    """Context manager capturing a before/after GPU measurement boundary (Mode B).

    Captures a GPUSnapshot immediately on entry (before) and immediately on
    exit (after), regardless of whether an exception occurred.

    IMPORTANT: This does NOT measure inference latency.  No timing is performed.
    Do NOT use the timestamps in before/after as a timing mechanism.

    Parameters
    ----------
    device_id : int
        CUDA device index to capture.
    operation_label : str
        Human-readable label for the operation being bracketed.

    Yields
    ------
    None
        The bounded operation runs inside the with-block.

    Returns (via generator send/result pattern)
    -------------------------------------------
    The GPUMeasurementBoundary is accessible via the context variable:
        with capture_gpu_boundary(0, "load_weights") as boundary_ref:
            load_model_weights()
        boundary = boundary_ref.result

    Alternatively, use take_gpu_snapshot() directly:
        before = take_gpu_snapshot(device_id)
        do_operation()
        after = take_gpu_snapshot(device_id)
        boundary = GPUMeasurementBoundary(before=before, after=after, ...)

    Notes
    -----
    - Exception-safe: after snapshot is always captured.
    - No CUDA synchronization is performed.  For accurate memory state after
      async GPU operations, Step 5.5 must add torch.cuda.synchronize() before
      calling take_gpu_snapshot().
    """
    # Simple approach: use a mutable container so callers can access the result
    class BoundaryHolder:
        result: Optional[GPUMeasurementBoundary] = None

    holder = BoundaryHolder()
    before = take_gpu_snapshot(device_id)
    try:
        yield holder
    finally:
        after = take_gpu_snapshot(device_id)
        holder.result = GPUMeasurementBoundary(
            before=before,
            after=after,
            operation_label=operation_label,
        )


# ---------------------------------------------------------------------------
# MODE A — BACKGROUND PERIODIC SAMPLING (GPUMonitor)
# ---------------------------------------------------------------------------

class GPUMonitor:
    """Background GPU telemetry sampler with start/stop lifecycle (Mode A).

    Polls take_gpu_snapshot() at a configurable interval in a daemon thread.
    Produces a time-series of GPUSnapshot observations for characterising
    sustained GPU load across longer-running or repeated inference workloads.

    MEASUREMENT MODE
    ----------------
    This is Mode A — PERIODIC MONITORING.

    Purpose:
        Sustained workload characterisation.
        Aggregate utilisation observation over a benchmark campaign.

    Limitation:
        Periodic sampling may miss a single short-lived SR inference entirely
        if the inference completes faster than sample_interval.

        Therefore: GPUMonitor utilisation samples MUST NOT be treated as
        exact per-operation GPU utilisation.

        For per-operation memory state boundaries, use take_gpu_snapshot()
        directly or capture_gpu_boundary() (Mode B).

    CPU/GPU Interval Independence:
        This monitor's sample_interval is independent of the Step 5.3
        BenchmarkProcessMonitor's interval.  CPU and GPU telemetry have
        different collection mechanisms and overhead characteristics.
        Do not assume they have equivalent temporal resolution.

    Parameters
    ----------
    device_id : int
        CUDA device index to monitor.
    sample_interval : float
        Seconds between consecutive snapshots.
        Default: 0.5 s — conservative enough to avoid significant monitoring
        overhead while still producing useful time-series data.
        This interval is INDEPENDENT of Step 5.3 CPU monitor interval.
        Step 5.5 may override this value.

    Lifecycle
    ---------
    monitor = GPUMonitor(device_id=0)
    monitor.start()
    # ... sustained workload ...
    monitor.stop()
    samples = monitor.get_samples()  # List[GPUSnapshot]

    Or via context manager (preferred — guaranteed cleanup on exception):
    with GPUMonitor(device_id=0) as monitor:
        # ... workload ...
    samples = monitor.get_samples()

    Notes
    -----
    - Background thread is a daemon; will not prevent process exit.
    - stop() joins the thread with a 5-second timeout.
    - stop() is idempotent.
    - Do NOT use this monitor to measure inference latency.  CUDA operations
      are asynchronous; timing belongs in Step 5.5.
    """

    def __init__(self, device_id: int, sample_interval: float = 0.5) -> None:
        validate_device_id(device_id)
        self.device_id = device_id
        self.sample_interval = sample_interval
        self._samples: List[GPUSnapshot] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Starts the background GPU sampling thread.

        Raises RuntimeError if already running.
        """
        if self._thread is not None:
            raise RuntimeError("GPUMonitor is already running.")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"GPUMonitorThread-device{self.device_id}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        """Internal periodic sampling loop."""
        while not self._stop_event.is_set():
            try:
                snap = take_gpu_snapshot(self.device_id)
                with self._lock:
                    self._samples.append(snap)
            except Exception as exc:
                logger.debug(
                    "GPUMonitor sampling error for device %d (non-fatal): %s",
                    self.device_id, exc
                )
            self._stop_event.wait(timeout=self.sample_interval)

    def stop(self) -> None:
        """Stops the background sampling thread and joins it.

        Safe to call multiple times.  No-op if not running.
        """
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._thread = None

    def get_samples(self) -> List[GPUSnapshot]:
        """Returns a thread-safe copy of all collected GPUSnapshots."""
        with self._lock:
            return list(self._samples)

    def sample_count(self) -> int:
        """Returns the number of snapshots collected so far (thread-safe)."""
        with self._lock:
            return len(self._samples)

    def __enter__(self) -> "GPUMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
        return False


# ---------------------------------------------------------------------------
# CONVENIENCE CONTEXT MANAGER (Mode A)
# ---------------------------------------------------------------------------

@contextmanager
def gpu_measurement_context(
    device_id: int,
    sample_interval: float = 0.5,
) -> Generator[GPUMonitor, None, None]:
    """Context manager wrapping GPUMonitor lifecycle (Mode A — periodic).

    Starts a GPUMonitor on entry and guarantees stop on exit (even on exception).

    Parameters
    ----------
    device_id : int
        CUDA device index to monitor.
    sample_interval : float
        Seconds between snapshots.  Default: 0.5 s.
        Independent of Step 5.3 CPU monitor interval.

    Yields
    ------
    GPUMonitor
        The running monitor.

    Usage
    -----
    with gpu_measurement_context(device_id=0) as monitor:
        run_sustained_workload()
    samples = monitor.get_samples()

    Notes
    -----
    - No benchmark timing is performed here.
    - On a CPU-only machine, raises RuntimeError at entry.
    """
    monitor = GPUMonitor(device_id=device_id, sample_interval=sample_interval)
    try:
        monitor.start()
        yield monitor
    finally:
        monitor.stop()
