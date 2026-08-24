"""
adaptive_sr.benchmarking.harness
================================
Step 5.5 — Inference Benchmark Harness.

Converts the execution-control and measurement infrastructure from Steps 5.2–5.4
into a reproducible inference benchmark harness.
"""

from __future__ import annotations

import os
import time
import json
import platform
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator, ConfigDict
import numpy as np
import torch

from adaptive_sr.benchmarking.cpu_control import (
    CPUExecutionConfig,
    BenchmarkProcessMonitor,
    benchmark_execution_context,
    get_available_cpus
)
from adaptive_sr.benchmarking.gpu_measurement import (
    GPUMonitor,
    take_gpu_snapshot,
    GPUMeasurementBoundary,
    get_cuda_availability,
    CUDAAvailability,
    _get_driver_version
)
from adaptive_sr.benchmarking.adapters.registry import get_adapter
from src.modules.video_loader import VideoLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Schemas for Benchmark cases and results
# ---------------------------------------------------------------------------

class BenchmarkConfig(BaseModel):
    """Configuration case identifying the execution options for benchmarking."""
    model_id: str
    scale: int
    input_id: str
    device: str
    cpu_config: Optional[CPUExecutionConfig] = None
    warmup_runs: int = Field(default=3, ge=0)
    measured_runs: int = Field(default=20, gt=0)
    gpu_sampling_interval: float = Field(default=0.5, gt=0.0)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_device_and_cpu_config(self) -> "BenchmarkConfig":
        if self.scale <= 0:
            raise ValueError("Scale must be a positive integer.")

        dev_lower = self.device.lower()
        if not (dev_lower == "cpu" or dev_lower.startswith("cuda")):
            raise ValueError(f"Unsupported device: {self.device}. Must be 'cpu' or 'cuda[:N]'.")

        if dev_lower == "cpu":
            if self.cpu_config is None:
                raise ValueError("cpu_config must be provided when device='cpu'")
        else:
            if self.cpu_config is not None:
                raise ValueError("cpu_config must be None when device is CUDA")

        return self


class TrialRecord(BaseModel):
    """Individually inspectable record of a single benchmark trial."""
    trial_idx: int
    latency: Optional[float] = None  # in seconds; None if trial failed
    success: bool
    error_message: Optional[str] = None

    # VRAM allocated/reserved boundary telemetry for GPU (if applicable)
    gpu_memory_allocated_before: Optional[int] = None
    gpu_memory_allocated_after: Optional[int] = None
    gpu_memory_reserved_before: Optional[int] = None
    gpu_memory_reserved_after: Optional[int] = None


class LatencyStatistics(BaseModel):
    """Aggregated latency statistics for a benchmark case."""
    count: int
    mean: float
    median: float
    min: float
    max: float
    std_dev: float
    p95: float
    p95_method: str = "numpy_linear"


class ResourceSummary(BaseModel):
    """Summarized process/device resource telemetry observations."""
    cpu_utilization_mean: Optional[float] = None
    cpu_utilization_peak: Optional[float] = None
    memory_used_mean_mb: Optional[float] = None
    memory_used_peak_mb: Optional[float] = None

    # GPU observed metrics
    gpu_utilization_mean: Optional[float] = None
    gpu_utilization_peak: Optional[float] = None
    memory_allocated_before_mb: Optional[float] = None
    memory_allocated_after_mb: Optional[float] = None
    memory_reserved_before_mb: Optional[float] = None
    memory_reserved_after_mb: Optional[float] = None
    gpu_memory_free_before_mb: Optional[float] = None
    gpu_memory_free_after_mb: Optional[float] = None


class HostMetadata(BaseModel):
    """System-level hardware and runtime build version information."""
    operating_system: str
    python_version: str
    pytorch_version: str
    cpu_count: int
    gpu_name: Optional[str] = None
    gpu_vram_total_mb: Optional[float] = None
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None


class BenchmarkResult(BaseModel):
    """Comprehensive structured outcome of a benchmark run."""
    benchmark_id: str
    timestamp: str
    config: BenchmarkConfig
    trial_latencies: List[float]
    latency_statistics: LatencyStatistics
    throughput_fps: float
    resource_summary: ResourceSummary
    metadata: Dict[str, Any]
    successful_trials: int
    failed_trials: int
    failures: List[str]


# ---------------------------------------------------------------------------
# Benchmark Harness Engine
# ---------------------------------------------------------------------------

class InferenceBenchmarkHarness:
    """Reproducible benchmarking manager orchestrating inputs, adapters, contexts, and statistics."""

    def __init__(self, manifest_path: str = "data/benchmarks/sr/manifests/benchmark_manifest.json") -> None:
        self.manifest_path = os.path.abspath(manifest_path)
        self.base_dir = os.path.dirname(os.path.dirname(self.manifest_path))
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Benchmark manifest not found at: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

    def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Executes a benchmark case according to the validated configuration.

        Parameters
        ----------
        config : BenchmarkConfig
            The configuration detailing the benchmark options.

        Returns
        -------
        BenchmarkResult
            The structured benchmark results.
        """
        # Step 45.1: Validate configuration
        # Verify model is registered
        adapter = get_adapter(config.model_id)

        # Verify model supports scale
        if config.scale not in adapter.scale_factors:
            raise ValueError(
                f"Model '{config.model_id}' does not support scale {config.scale}. "
                f"Supported: {adapter.scale_factors}"
            )

        # Verify device compatibility / CUDA availability
        is_gpu = config.device.lower().startswith("cuda")
        device_id = 0
        if is_gpu:
            cuda_info = get_cuda_availability()
            if cuda_info["status"] != CUDAAvailability.AVAILABLE:
                raise RuntimeError(
                    f"CUDA execution requested ('{config.device}'), but CUDA is unavailable on this host."
                )
            if ":" in config.device:
                device_id = int(config.device.split(":")[1])
                if device_id >= cuda_info["device_count"]:
                    raise RuntimeError(
                        f"Requested CUDA device {device_id} is out of bounds. Discovered visible count: {cuda_info['device_count']}"
                    )
        else:
            if config.cpu_config is None:
                raise ValueError("cpu_config must be provided when running on CPU.")

        # Find target video in corpus manifest
        video_metadata = None
        for v in self.manifest.get("videos", []):
            if v.get("benchmark_video_id") == config.input_id:
                video_metadata = v
                break

        if video_metadata is None:
            raise ValueError(f"Input sample ID '{config.input_id}' not found in benchmark manifest.")

        # Get first chunk path for isolated compute benchmark (Step 5.1 input preparation)
        chunks = video_metadata.get("chunks", [])
        if not chunks:
            raise ValueError(f"No chunks associated with video '{config.input_id}' in manifest.")
        chunk_rel_path = chunks[0]["file_path"]
        input_file_path = os.path.join(self.base_dir, chunk_rel_path)

        # Step 45.2: Prepare input (load frames outside measurement boundary)
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Corpus video chunk file not found at: {input_file_path}")

        # Extract frames to memory
        cap = cv2_capture_frames(input_file_path)
        if not cap:
            raise RuntimeError(f"Failed to read frames from {input_file_path}")

        # Isolate inputs based on model sequence demands (spatial vs temporal)
        if adapter.temporal_or_spatial == "spatial":
            # Spatial models benchmark on a single prepared frame to isolate compute
            prepared_input = cap[0]
            input_shape = prepared_input.shape
        else:
            # Temporal models benchmark on the full frame sequence list
            prepared_input = cap
            input_shape = cap[0].shape

        # Step 45.3: Initialize adapter (exclude weight loading/compilation from timing)
        if not is_gpu:
            adapter.initialize(device=config.device, scale=config.scale, num_threads=config.cpu_config.num_threads)
        else:
            adapter.initialize(device=config.device, scale=config.scale)

        # Monitored execution setups
        trial_records: List[TrialRecord] = []
        trial_latencies: List[float] = []
        failures: List[str] = []
        successful_trials = 0
        failed_trials = 0

        # Capture Host Metadata
        host_meta = self._get_host_metadata(is_gpu, device_id)

        # CPU Exec Path
        if not is_gpu:
            cpu_monitor = BenchmarkProcessMonitor(sample_interval=0.05)
            with benchmark_execution_context(config.cpu_config, cpu_monitor):
                # Warm up runs
                for _ in range(config.warmup_runs):
                    adapter.process(prepared_input, scale=config.scale)

                # Steady-state trials timing boundary
                for trial_idx in range(config.measured_runs):
                    try:
                        t0 = time.perf_counter()
                        out = adapter.process(prepared_input, scale=config.scale)
                        t1 = time.perf_counter()
                        latency = t1 - t0

                        # Validate output outside timed boundary
                        out_list = [out] if isinstance(out, np.ndarray) else out
                        adapter.validate_outputs(out_list, input_shape, config.scale)

                        trial_latencies.append(latency)
                        successful_trials += 1
                        trial_records.append(TrialRecord(trial_idx=trial_idx, latency=latency, success=True))
                    except Exception as e:
                        failed_trials += 1
                        failures.append(str(e))
                        trial_records.append(TrialRecord(trial_idx=trial_idx, success=False, error_message=str(e)))

            cpu_samples = cpu_monitor.get_samples()
            gpu_samples = []
            boundary_snapshots = []

        # GPU Exec Path (CUDA synchronization rules)
        else:
            gpu_monitor = GPUMonitor(device_id=device_id, sample_interval=config.gpu_sampling_interval)
            boundary_snapshots = []

            with gpu_monitor:
                # Warm up runs
                for _ in range(config.warmup_runs):
                    adapter.process(prepared_input, scale=config.scale)

                torch.cuda.synchronize(device_id)

                # Steady-state trials timing boundary
                for trial_idx in range(config.measured_runs):
                    torch.cuda.synchronize(device_id)
                    mem_before = take_gpu_snapshot(device_id)

                    try:
                        # Mandated pre-timing synchronize boundary
                        torch.cuda.synchronize(device_id)
                        t0 = time.perf_counter()
                        out = adapter.process(prepared_input, scale=config.scale)
                        # Mandated post-timing synchronize boundary
                        torch.cuda.synchronize(device_id)
                        t1 = time.perf_counter()
                        latency = t1 - t0

                        # Validate output outside timed boundary
                        out_list = [out] if isinstance(out, np.ndarray) else out
                        adapter.validate_outputs(out_list, input_shape, config.scale)

                        torch.cuda.synchronize(device_id)
                        mem_after = take_gpu_snapshot(device_id)

                        boundary_snapshots.append(
                            GPUMeasurementBoundary(before=mem_before, after=mem_after, operation_label=f"trial_{trial_idx}")
                        )

                        trial_latencies.append(latency)
                        successful_trials += 1
                        trial_records.append(
                            TrialRecord(
                                trial_idx=trial_idx,
                                latency=latency,
                                success=True,
                                gpu_memory_allocated_before=mem_before.process_gpu_memory_allocated_bytes,
                                gpu_memory_allocated_after=mem_after.process_gpu_memory_allocated_bytes,
                                gpu_memory_reserved_before=mem_before.process_gpu_memory_reserved_bytes,
                                gpu_memory_reserved_after=mem_after.process_gpu_memory_reserved_bytes,
                            )
                        )
                    except Exception as e:
                        failed_trials += 1
                        failures.append(str(e))
                        trial_records.append(TrialRecord(trial_idx=trial_idx, success=False, error_message=str(e)))

            gpu_samples = gpu_monitor.get_samples()
            cpu_samples = []

        # Cleanup model backend sessions
        adapter.close()

        # Compute stats (precision linear interpolation percentile)
        if trial_latencies:
            latencies_arr = np.array(trial_latencies)
            stats = LatencyStatistics(
                count=len(trial_latencies),
                mean=float(np.mean(latencies_arr)),
                median=float(np.median(latencies_arr)),
                min=float(np.min(latencies_arr)),
                max=float(np.max(latencies_arr)),
                std_dev=float(np.std(latencies_arr)) if len(trial_latencies) > 1 else 0.0,
                p95=float(np.percentile(latencies_arr, 95))
            )
            # Sequential FPS calculation
            throughput_fps = 1.0 / stats.mean
        else:
            stats = LatencyStatistics(
                count=0,
                mean=0.0,
                median=0.0,
                min=0.0,
                max=0.0,
                std_dev=0.0,
                p95=0.0
            )
            throughput_fps = 0.0

        # Aggregate Resource Snapshot telemetry
        res_summary = self._summarize_resources(cpu_samples, gpu_samples, boundary_snapshots)

        # Build full output record
        benchmark_id = f"bench_{config.model_id}_x{config.scale}_{config.device.replace(':', '_')}_{int(time.time())}"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

        meta_details = {
            "input_details": {
                "input_id": config.input_id,
                "width": input_shape[1],
                "height": input_shape[0],
                "channels": input_shape[2],
                "scale": config.scale,
            },
            "host_metadata": host_meta.model_dump(),
            "trials": [tr.model_dump() for tr in trial_records],
            "software_versions": {
                "operating_system": platform.system(),
                "python": platform.python_version(),
                "pytorch": torch.__version__,
            }
        }

        return BenchmarkResult(
            benchmark_id=benchmark_id,
            timestamp=timestamp,
            config=config,
            trial_latencies=trial_latencies,
            latency_statistics=stats,
            throughput_fps=throughput_fps,
            resource_summary=res_summary,
            metadata=meta_details,
            successful_trials=successful_trials,
            failed_trials=failed_trials,
            failures=failures,
        )

    def _get_host_metadata(self, is_gpu: bool, device_id: int) -> HostMetadata:
        import psutil

        gpu_name = None
        gpu_vram = None
        cuda_version = None
        driver_version = None

        if is_gpu:
            try:
                props = torch.cuda.get_device_properties(device_id)
                gpu_name = props.name
                gpu_vram = props.total_memory
                cuda_version = torch.version.cuda
                driver_version = _get_driver_version()
            except Exception:
                pass

        return HostMetadata(
            operating_system=platform.system(),
            python_version=platform.python_version(),
            pytorch_version=torch.__version__,
            cpu_count=psutil.cpu_count(logical=True) or 1,
            gpu_name=gpu_name,
            gpu_vram_total_mb=gpu_vram / (1024 * 1024) if gpu_vram is not None else None,
            cuda_version=cuda_version,
            driver_version=driver_version,
        )

    def _summarize_resources(
        self,
        cpu_samples: list,
        gpu_samples: list,
        boundary_snapshots: List[GPUMeasurementBoundary]
    ) -> ResourceSummary:
        cpu_utils = [s.cpu_percent for s in cpu_samples if s.cpu_percent is not None]
        cpu_mems = [s.memory_used_bytes / (1024 * 1024) for s in cpu_samples if s.memory_used_bytes is not None]

        gpu_utils = [s.gpu_utilization_percent for s in gpu_samples if s.gpu_utilization_percent is not None]

        cpu_mean = float(np.mean(cpu_utils)) if cpu_utils else None
        cpu_peak = float(np.max(cpu_utils)) if cpu_utils else None

        mem_mean = float(np.mean(cpu_mems)) if cpu_mems else None
        mem_peak = float(np.max(cpu_mems)) if cpu_mems else None

        gpu_mean = float(np.mean(gpu_utils)) if gpu_utils else None
        gpu_peak = float(np.max(gpu_utils)) if gpu_utils else None

        # Snapshot memory boundaries (Mode B)
        allocated_before = None
        allocated_after = None
        reserved_before = None
        reserved_after = None
        free_before = None
        free_after = None

        if boundary_snapshots:
            first_boundary = boundary_snapshots[0]
            last_boundary = boundary_snapshots[-1]

            if first_boundary.before.process_gpu_memory_allocated_bytes is not None:
                allocated_before = first_boundary.before.process_gpu_memory_allocated_bytes / (1024 * 1024)
            if last_boundary.after.process_gpu_memory_allocated_bytes is not None:
                allocated_after = last_boundary.after.process_gpu_memory_allocated_bytes / (1024 * 1024)

            if first_boundary.before.process_gpu_memory_reserved_bytes is not None:
                reserved_before = first_boundary.before.process_gpu_memory_reserved_bytes / (1024 * 1024)
            if last_boundary.after.process_gpu_memory_reserved_bytes is not None:
                reserved_after = last_boundary.after.process_gpu_memory_reserved_bytes / (1024 * 1024)

            if first_boundary.before.gpu_memory_free_bytes is not None:
                free_before = first_boundary.before.gpu_memory_free_bytes / (1024 * 1024)
            if last_boundary.after.gpu_memory_free_bytes is not None:
                free_after = last_boundary.after.gpu_memory_free_bytes / (1024 * 1024)

        return ResourceSummary(
            cpu_utilization_mean=cpu_mean,
            cpu_utilization_peak=cpu_peak,
            memory_used_mean_mb=mem_mean,
            memory_used_peak_mb=mem_peak,
            gpu_utilization_mean=gpu_mean,
            gpu_utilization_peak=gpu_peak,
            memory_allocated_before_mb=allocated_before,
            memory_allocated_after_mb=allocated_after,
            memory_reserved_before_mb=reserved_before,
            memory_reserved_after_mb=reserved_after,
            gpu_memory_free_before_mb=free_before,
            gpu_memory_free_after_mb=free_after,
        )


def cv2_capture_frames(file_path: str) -> List[np.ndarray]:
    """Helper to extract frames sequentially using OpenCV cap."""
    import cv2
    cap = cv2.VideoCapture(file_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames
