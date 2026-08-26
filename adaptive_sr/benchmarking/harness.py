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
    is_decision_run: bool = False
    cpu_0_intentional: bool = False

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

            if self.is_decision_run and not self.cpu_0_intentional:
                import psutil as _psutil
                logical_core_count = _psutil.cpu_count(logical=True) or 1
                excl = self.cpu_config.exclude_cpu_ids or []
                if 0 not in excl:
                    if logical_core_count <= 2:
                        # D2: On ≤2-core hosts, OS interrupt contention on core 0 is
                        # proportionally large — enforce the isolated split strictly.
                        raise ValueError(
                            "BenchmarkConfig declared as is_decision_run on CPU must explicitly "
                            "exclude CPU 0 (exclude_cpu_ids must contain 0) to minimize noise "
                            "on this ≤2-core host, unless cpu_0_intentional is set to True."
                        )
                    else:
                        # D2: Above 2 cores, core-0 contention becomes proportionally smaller.
                        # The baseline/isolated split is still available on request, but
                        # we only warn here rather than blocking execution.
                        import logging as _logging
                        _logging.getLogger(__name__).warning(
                            "BenchmarkConfig is_decision_run=True on a >2-core host with CPU 0 "
                            "not excluded. On hosts with >2 logical cores, CPU-0 exclusion is "
                            "recommended but not required (D2). Consider setting "
                            "exclude_cpu_ids=[0] for the most rigorous decision-quality data."
                        )
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
    flagged: Optional[str] = None

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
    p95_sample_count: int
    p95_confidence_note: str
    p95_confidence: str
    p95_min_recommended_n: int

    # Explicit statistical fields required by downstream tools/analysis
    min_latency: float
    median_latency: float
    mean_latency: float
    max_latency: float
    std_latency: float
    p95_latency: float
    measured_trial_count: int


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
    thermal_state: str = "not_measured"


class BenchmarkResult(BaseModel):
    """Comprehensive structured outcome of a benchmark run."""
    benchmark_id: str
    timestamp: str
    config: BenchmarkConfig
    trial_latencies: List[float]
    latency_statistics: LatencyStatistics
    throughput_fps: float
    resource_summary: ResourceSummary
    warmup_resource_summary: ResourceSummary
    metadata: Dict[str, Any]
    successful_trials: int
    failed_trials: int
    failures: List[str]
    flagged: Optional[str] = None


class MultiSessionResult(BaseModel):
    """Encapsulates outcomes of executing a benchmark configuration over multiple independent sessions."""
    benchmark_id: str
    config: BenchmarkConfig
    sessions: List[BenchmarkResult]
    metadata: Dict[str, Any]
    decision_eligible: Optional[bool] = None
    eligibility_reason: Optional[str] = None
    coefficient_of_variation: Optional[float] = None


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

    def run_benchmark(self, config: BenchmarkConfig, _skip_decision_split: bool = False) -> BenchmarkResult:
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
        # Step 5.5 D2: CPU-0 Isolation split check for decision-quality runs
        if not _skip_decision_split and config.device.lower() == "cpu" and config.is_decision_run:
            from adaptive_sr.benchmarking.cpu_control import select_cpu_ids

            # 1. Isolated configuration (CPU 0 excluded)
            excl_iso = sorted(list(set((config.cpu_config.exclude_cpu_ids or []) + [0])))
            cpu_ids_iso = select_cpu_ids(count=len(config.cpu_config.cpu_ids), exclude_cpu_ids=excl_iso)
            cpu_config_iso = CPUExecutionConfig(
                cpu_ids=cpu_ids_iso,
                num_threads=config.cpu_config.num_threads,
                exclude_cpu_ids=excl_iso
            )
            config_iso = config.model_copy(update={"cpu_config": cpu_config_iso})

            # 2. Baseline configuration (CPU 0 included)
            excl_base = [cid for cid in (config.cpu_config.exclude_cpu_ids or []) if cid != 0]
            cpu_ids_base = select_cpu_ids(count=len(config.cpu_config.cpu_ids), exclude_cpu_ids=excl_base)
            cpu_config_base = CPUExecutionConfig(
                cpu_ids=cpu_ids_base,
                num_threads=config.cpu_config.num_threads,
                exclude_cpu_ids=excl_base
            )
            config_base = config.model_copy(update={"cpu_config": cpu_config_base, "cpu_0_intentional": True})

            # Run baseline and isolated configurations
            res_isolated = self.run_benchmark(config_iso, _skip_decision_split=True)
            res_baseline = self.run_benchmark(config_base, _skip_decision_split=True)

            # Compute delta
            mean_iso = res_isolated.latency_statistics.mean_latency
            mean_base = res_baseline.latency_statistics.mean_latency
            delta = mean_base - mean_iso

            flag_status = None
            if delta > 0.10 * mean_iso:
                flag_status = "core0_noise_significant"

            # Merge results into isolated metadata
            res_isolated.metadata["cpu_config_baseline"] = res_baseline.model_dump()
            res_isolated.metadata["cpu_config_isolated"] = res_isolated.model_dump()
            res_isolated.metadata["core0_contention_delta"] = delta
            if flag_status:
                res_isolated.flagged = flag_status
                res_isolated.metadata["flagged"] = flag_status

            return res_isolated

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
            # Warm up phase
            warmup_monitor = BenchmarkProcessMonitor(sample_interval=0.05)
            with benchmark_execution_context(config.cpu_config, warmup_monitor):
                for _ in range(config.warmup_runs):
                    adapter.process(prepared_input, scale=config.scale)
            warmup_cpu_samples = warmup_monitor.get_samples()
            warmup_gpu_samples = []

            # Measured trials phase
            measured_monitor = BenchmarkProcessMonitor(sample_interval=0.05)
            with benchmark_execution_context(config.cpu_config, measured_monitor):
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
                        
                        trial_flag = None
                        if hasattr(adapter, "get_last_inference_metadata"):
                            crop_meta = adapter.get_last_inference_metadata()
                            if crop_meta and not crop_meta.get("crop_within_tolerance", True):
                                trial_flag = "anomalous_crop"
                                
                        trial_records.append(TrialRecord(trial_idx=trial_idx, latency=latency, success=True, flagged=trial_flag))
                    except Exception as e:
                        failed_trials += 1
                        failures.append(str(e))
                        trial_records.append(TrialRecord(trial_idx=trial_idx, success=False, error_message=str(e)))

            cpu_samples = measured_monitor.get_samples()
            gpu_samples = []
            boundary_snapshots = []

        # GPU Exec Path (CUDA synchronization rules)
        else:
            boundary_snapshots = []

            # Warm up phase
            warmup_monitor = GPUMonitor(device_id=device_id, sample_interval=config.gpu_sampling_interval)
            with warmup_monitor:
                for _ in range(config.warmup_runs):
                    adapter.process(prepared_input, scale=config.scale)
            warmup_gpu_samples = warmup_monitor.get_samples()
            warmup_cpu_samples = []

            # Measured trials phase
            gpu_monitor = GPUMonitor(device_id=device_id, sample_interval=config.gpu_sampling_interval)
            with gpu_monitor:
                torch.cuda.synchronize(device_id)

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
                        
                        trial_flag = None
                        if hasattr(adapter, "get_last_inference_metadata"):
                            crop_meta = adapter.get_last_inference_metadata()
                            if crop_meta and not crop_meta.get("crop_within_tolerance", True):
                                trial_flag = "anomalous_crop"
                                
                        trial_records.append(
                            TrialRecord(
                                trial_idx=trial_idx,
                                latency=latency,
                                success=True,
                                flagged=trial_flag,
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
            cnt = len(trial_latencies)
            conf_note = "Exploratory p95; limited tail resolution at n=20." if cnt <= 20 else "Sufficient tail resolution."
            p95_conf = "exploratory" if cnt < 100 else "reliable"
            p95_min_n = 100
            mean_val = float(np.mean(latencies_arr))
            median_val = float(np.median(latencies_arr))
            min_val = float(np.min(latencies_arr))
            max_val = float(np.max(latencies_arr))
            std_val = float(np.std(latencies_arr)) if len(trial_latencies) > 1 else 0.0
            p95_val = float(np.percentile(latencies_arr, 95))
            
            stats = LatencyStatistics(
                count=cnt,
                mean=mean_val,
                median=median_val,
                min=min_val,
                max=max_val,
                std_dev=std_val,
                p95=p95_val,
                p95_sample_count=cnt,
                p95_confidence_note=conf_note,
                p95_confidence=p95_conf,
                p95_min_recommended_n=p95_min_n,
                min_latency=min_val,
                median_latency=median_val,
                mean_latency=mean_val,
                max_latency=max_val,
                std_latency=std_val,
                p95_latency=p95_val,
                measured_trial_count=cnt
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
                p95=0.0,
                p95_sample_count=0,
                p95_confidence_note="No trials executed.",
                p95_confidence="exploratory",
                p95_min_recommended_n=100,
                min_latency=0.0,
                median_latency=0.0,
                mean_latency=0.0,
                max_latency=0.0,
                std_latency=0.0,
                p95_latency=0.0,
                measured_trial_count=0
            )
            throughput_fps = 0.0

        # Determine overall case flags
        flagged_res = None
        anomalous_crop_detected = any(tr.flagged == "anomalous_crop" for tr in trial_records)
        if anomalous_crop_detected:
            flagged_res = "anomalous_crop"

        # Aggregate Resource Snapshot telemetry
        res_summary = self._summarize_resources(cpu_samples, gpu_samples, boundary_snapshots)
        warmup_summary = self._summarize_resources(warmup_cpu_samples, warmup_gpu_samples, [])

        # Build full output record
        benchmark_id = f"bench_{config.model_id}_x{config.scale}_{config.device.replace(':', '_')}_{int(time.time())}"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

        meta_details = {
            "session_id": benchmark_id,
            "timestamp": timestamp,
            "benchmark_config": config.model_dump(),
            "flagged": flagged_res,
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

        # Include crop info in metadata if model has it
        if hasattr(adapter, "get_last_inference_metadata"):
            meta_details["crop_metadata"] = adapter.get_last_inference_metadata()

        return BenchmarkResult(
            benchmark_id=benchmark_id,
            timestamp=timestamp,
            config=config,
            trial_latencies=trial_latencies,
            latency_statistics=stats,
            throughput_fps=throughput_fps,
            resource_summary=res_summary,
            warmup_resource_summary=warmup_summary,
            metadata=meta_details,
            successful_trials=successful_trials,
            failed_trials=failed_trials,
            failures=failures,
            flagged=flagged_res,
        )

    def run_multi_session(self, config: BenchmarkConfig, num_sessions: int = 1) -> MultiSessionResult:
        """Executes a benchmark config over multiple independent sessions, preserving session isolation."""
        sessions = []
        for i in range(num_sessions):
            result = self.run_benchmark(config)
            session_id = f"{result.benchmark_id}_session_{i + 1}"
            result.benchmark_id = session_id
            result.metadata["session_id"] = session_id
            sessions.append(result)

        benchmark_id = f"multisession_{config.model_id}_x{config.scale}_{config.device.replace(':', '_')}_{int(time.time())}"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

        # Calculate CV and eligibility
        session_means = [s.latency_statistics.mean_latency for s in sessions]
        cv = None
        decision_eligible = False
        eligibility_reason = None

        if len(sessions) < 3:
            decision_eligible = False
            eligibility_reason = "insufficient_sessions_count"
        else:
            mean_of_means = np.mean(session_means)
            std_of_means = np.std(session_means, ddof=1) if len(session_means) > 1 else 0.0
            cv = std_of_means / mean_of_means if mean_of_means > 0.0 else 0.0
            if cv <= 0.15:
                decision_eligible = True
            else:
                decision_eligible = False
                eligibility_reason = "between_session_variance_exceeds_threshold"

        # I2: Build eligibility record with explicit (model, device, cpu-affinity-config) tuple
        # so downstream decision documents can cite the exact configuration scope.
        cpu_affinity_config = None
        if config.cpu_config is not None:
            cpu_affinity_config = {
                "cpu_ids": config.cpu_config.cpu_ids,
                "num_threads": config.cpu_config.num_threads,
                "exclude_cpu_ids": config.cpu_config.exclude_cpu_ids or [],
            }

        eligibility_record = {
            "scope_tuple": {
                "model_id": config.model_id,
                "device": config.device,
                "cpu_affinity_config": cpu_affinity_config,  # I2: per-config tuple key
            },
            "session_means": session_means,
            "session_thermal_states": [
                s.metadata.get("host_metadata", {}).get("thermal_state", "not_measured")
                for s in sessions
            ],
            "coefficient_of_variation": cv,
            "decision_eligible": decision_eligible,
            "eligibility_reason": eligibility_reason
        }

        return MultiSessionResult(
            benchmark_id=benchmark_id,
            config=config,
            sessions=sessions,
            metadata={
                "num_sessions": num_sessions,
                "timestamp": timestamp,
                "eligibility_record": eligibility_record
            },
            decision_eligible=decision_eligible,
            eligibility_reason=eligibility_reason,
            coefficient_of_variation=cv
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
            thermal_state="not_measured"
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


class BenchmarkReportGenerator:
    """Report generator formatting headline tables, hiding exploratory p95 values, and listing footnotes."""

    @staticmethod
    def generate_comparison_table(results: List[BenchmarkResult]) -> str:
        lines = []
        lines.append("| Model ID | Device | Mean Latency (s) | Median Latency (s) | Max Latency (s) | p95 Latency (s) | Notes |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        footnotes = []
        footnote_counter = 1

        for res in results:
            stats = res.latency_statistics
            model = res.config.model_id
            device = res.config.device
            mean_l = f"{stats.mean_latency:.4f}"
            median_l = f"{stats.median_latency:.4f}"
            max_l = f"{stats.max_latency:.4f}"

            if stats.p95_confidence == "exploratory":
                p95_l = f"Fallback (med={median_l}/max={max_l})"
                note = f"Footnote [{footnote_counter}]"
                footnotes.append(f"[{footnote_counter}] Exploratory p95 for {model} on {device}: {stats.p95_latency:.4f}s (n={stats.count})")
                footnote_counter += 1
            else:
                p95_l = f"{stats.p95_latency:.4f}"
                note = "Reliable p95"

            lines.append(f"| {model} | {device} | {mean_l} | {median_l} | {max_l} | {p95_l} | {note} |")

        report = "\n".join(lines)
        if footnotes:
            report += "\n\n**Footnotes:**\n" + "\n".join(f"- {fn}" for fn in footnotes)

        return report
