# Step 5 — Empirical Model Benchmarking

## Step 5.1 — Benchmark Dataset / Test-Video Preparation

### 1. Purpose & Benchmark Corpus Scope
This step establishes a deterministic, reproducible benchmark dataset (manifest, videos, chunks, and metadata) to support empirical Super-Resolution (SR) model evaluation. Every model analyzed in later steps will run on the exact same inputs (resolutions, frame counts, FPS, and dynamic content patterns).

> [!IMPORTANT]
> **Step 5.1 prepares benchmark INPUTS. It does NOT load SR models, define model runner adapters, or run SR inference.**

#### Benchmark Corpus Scope Clarification
The synthetic corpus generated in this step is intentionally suitable for:
- Inference latency benchmarking.
- Throughput / FPS benchmarking.
- CPU/GPU resource benchmarking.
- Spatial resolution/scale computational benchmarking.
- Pipeline correctness testing and verification of edge-to-client/edge-to-cloud interfaces.
- Target FPS/chunk handling validation.

The synthetic corpus is **NOT** considered sufficient evidence for:
- Natural-video SR perceptual quality comparison.
- Production-quality PSNR/SSIM/VMAF conclusions.
- Content-aware SR quality decisions.

Therefore, **Step 5.6 quality evaluation MUST use suitable real-world reference video clips** (natural video with full-reference frames) when making claims about SR quality.

---

### 2. Benchmark Corpus Layers
To support independent analysis of resources, performance, and video quality, the benchmark corpus is divided into three distinct layers:

| Layer | Title | Purpose | Status in Step 5.1 |
|-------|-------|---------|--------------------|
| **Layer A** | Controlled Synthetic Corpus | Latency, throughput, resource, and pipeline correctness testing under controlled conditions. | **Fully Implemented** |
| **Layer B** | Real-World Reference Corpus | Quantitative and qualitative SR quality evaluation (PSNR/SSIM/VMAF) in Step 5.6. | Deferred to Step 5.6 (to be introduced when quality comparisons are performed) |
| **Layer C** | Production-Like Streaming Inputs | End-to-end validation against the actual production streaming representation pipeline. | Deferred to end-to-end integration |

No quality metrics, model runners, or Layer B/C video files are introduced in Step 5.1.

---

### 3. Real-World Quality Dataset Policy
- The initial Step 5.1 synthetic corpus serves exclusively as the controlled latency/resource benchmark corpus (Layer A).
- A separate real-world reference corpus (Layer B) containing natural video with reference frames suitable for full-reference quality evaluation will be introduced for Step 5.6 when perceptual/quality comparisons are performed. 
- No quality evaluation or external real-world downloads are performed at this stage.

---

### 4. Step 1 Feature-Diversity Warning
- The synthetic corpus is **not** intended to provide a representative distribution of real-world content-complexity features.
- While it may be used to verify that the Step 1 profiler runs correctly on controlled inputs, it **must NOT** be treated as the sole dataset for learning or validating content-aware scheduling relationships later in Step 10.

---

### 5. Codec Methodology & Mismatch Analysis
- **Production/Reference Codec**: The production streaming pipeline (e.g., in `VideoEncoder` under `src/modules/encoder.py`) encodes raw frames using FFmpeg's `libx264` to generate **H.264** video streams.
- **Synthetic Benchmark Codec**: The Step 5.1 synthetic corpus generator utilizes OpenCV's `cv2.VideoWriter` with the `mp4v` codec (**MPEG-4**) to guarantee a highly portable, cross-platform, and deterministic generation path on the development machine.
- **Codec Mismatch**: **Yes**.
- **Methodology Impact**: Because of this mismatch, Step 5.1 benchmark timings that include decode/container processing **must NOT** be interpreted as production H.264 decode timings. They represent Layer A comparative processing overhead only.

---

### 6. Benchmark Corpus Design
The benchmark corpus is designed to cover:
- **Frame Rates**: 30 FPS, 60 FPS, and 120 FPS.
- **Content Diversity**: Exercises three distinct motion profiles:
  - `lowmotion`: Moves dynamic targets at 20 pixels/second.
  - `moderatemotion`: Moves dynamic targets at 100 pixels/second.
  - `highmotion`: Moves dynamic targets at 300 pixels/second.
- **Duration**: Configured to exactly 4.0 seconds by default. Under a target chunk duration of 2.0 seconds, this guarantees exactly 2 chunks per video, allowing fast unit test execution while retaining multiple-chunk behavior.

---

### 7. Synthetic Video Generation
Videos are deterministically generated frame-by-frame using OpenCV (`cv2.VideoWriter`) with the `mp4v` codec.
The frames contain:
- A dark blue-gray background with a high-contrast grid pattern (high spatial frequencies).
- Multiple text lines showing the target frame rate and current frame indices.
- A dynamic intersecting crosshair pattern and concentric circles (vibrant cyan/magenta colors) moving along a diagonal bouncing path. The movement speed (pixels/sec) scales with the target frame rate so that spatial displacement per second remains constant across all FPS variants.

---

### 8. Source Metadata
For every benchmark video, the following metadata is verified and recorded:
- `benchmark_video_id`
- `filename`
- `source_fps`
- `width`
- `height`
- `duration_seconds`
- `frame_count`
- `codec`
- `pixel_format` (default: `yuv420p` for OpenCV containers)
- `source_bitrate` (null if not available)
- `audio_presence`

---

### 9. Chunk Association Mechanism
Step 5.1 reuses the Step 1 profiling pipeline (`run_profiler` from `profile_video.py`) to segment the generated video into dynamic chunks and profile its content complexity.
The chunks are associated by reading the profiler's output artifacts (`{video_id}_profile.json` and `{video_id}_manifest.json`). We merge:
- The chunk timeline ranges (`start_frame`, `end_frame`, `start_time_seconds`, etc.)
- The chunk files relative paths and SHA-256 hashes.

This ensures we consume the authoritative Step 1 timeline without duplicate computation or conflicting definitions.

---

### 10. Dataset Hashing & Integrity
For every generated source video and chunk, a stable SHA-256 cryptographic hash is calculated. Files are checked against these hashes during validation to catch files modified or corrupted in transport. Timestamps and file modification times are not used for identity, ensuring absolute reproducibility across machines.

---

### 11. Directory Structure
All benchmark files are located inside `data/benchmarks/sr/` to keep them cleanly separated from runtime Edge caches and network emulation logs:
```
data/
    benchmarks/
        sr/
            videos/      # High-quality synthetic source MP4s
            chunks/      # Dynamic FFmpeg copy-mode segmented chunk files
            profiles/    # Step 1 content profiles (JSON files)
            manifests/   # Step 1 manifests & the main benchmark_manifest.json
```

---

### 12. CLI Usage

To generate or overwrite the benchmark dataset:
```powershell
python -m adaptive_sr.benchmarking.prepare_dataset --output data/benchmarks/sr/ --overwrite
```

**Supported Options**:
- `--output`: Target folder (defaults to `data/benchmarks/sr`).
- `--duration`: Bounded duration in seconds (defaults to 4.0).
- `--width` / `--height`: Generation resolution (defaults to 640x360).
- `--seed`: Deterministic seed (defaults to 42).
- `--overwrite`: Forces regeneration of existing videos and profiles.
- `--validate`: Path to a dataset manifest to validate.

---

### 13. Validation Mechanism
A dataset validation routine is exposed via:
```powershell
python -m adaptive_sr.benchmarking.prepare_dataset --validate data/benchmarks/sr/manifests/benchmark_manifest.json
```

The validator checks:
1. Manifest structure complies with the Step 5.1 schema.
2. Every video file and chunk file exists.
3. Actual file SHA-256 hashes match the hashes written in the manifest.
4. Actual video container properties (resolution, FPS, frames) match manifest specifications.
5. Chunk timelines start at frame 0, end at the last frame, and have no gaps or overlaps.
6. All `benchmark_video_id`s are unique.

---

### 14. Automated Tests
Unit tests in [`tests/test_benchmark_preparation.py`](file:///d:/Full%20Stack/AdaptiveSR/tests/test_benchmark_preparation.py) cover all 16 specified requirements (FPS detection, hash checks, chunk continuity, duplicate prevention, and tamper detection).

---

### 15. Limitations
- **OpenCV VideoWriter Container timing**: On some operating systems, VideoWriter might introduce tiny floating-point rounding differences in duration. The validator handles this using a small floating-point tolerance check.
- **FFmpeg copy-segmentation boundaries**: Since synthetic videos have keyframes at every frame (by default in raw mpeg4), the chunks segment exactly at the requested 2.0s boundary.

---
---

## Step 5.2 — SR Model Runner Adapter Interface

### 1. Purpose & Core Abstraction
Step 5.2 establishes a common, model-independent adapter interface (`BaseSRAdapter`) that decouples model-specific initialization, execution, and verification pipelines from the future benchmarking harness. 

> [!IMPORTANT]
> **Step 5.2 provides model execution abstraction. It does NOT benchmark models.**
> Timing statistics, ProcessMonitor resource checks, and GPU profiling belong strictly to later Steps 5.3–5.5.

---

### 2. The Base Adapter Contract
All models implement the Abstract Base Class `BaseSRAdapter` which encapsulates:
- **Initialization** (`initialize(device, scale)`): Wakes up the target model backend on the specified hardware target and scale.
- **Inference Execution** (`process(inputs, scale)`): Performs the backend-specific runtime forward pass.
- **Cleanup** (`close()`): Releases backend sessions or clears GPU VRAM caches.
- **Capability Discovery**: Exposes whether the model can execute in the current environment (`is_available() -> bool`) and documents the dependency blocks if not (`get_unavailable_reason() -> Optional[str]`).

---

### 3. Spatial vs. Temporal Input/Output Contracts

#### Input Validation Contract
- **Spatial models** (e.g. FSRCNN, Real-ESRGAN): Accept either a single frame (`np.ndarray` of shape `(H, W, 3)`) or a list of frames to process independently.
- **Temporal models** (e.g. BasicVSR++): Must accept a list of frames representing a temporal sequence. Passing a single frame to a temporal adapter raises a `ValueError`.
- Validation checks confirm that:
  - Input objects are standard `uint8` numpy arrays with 3 color channels (BGR).
  - All frames inside a sequence list share identical spatial dimensions.

#### Output Validation Contract
- The output structure mirrors the input structure (a single array or a list of arrays).
- Every output frame must satisfy the strict scale factor equation:
  $$\text{output\_height} = \text{input\_height} \times \text{scale}$$
  $$\text{output\_width} = \text{input\_width} \times \text{scale}$$
- **No silent resizing is permitted**: If the backend produces an unexpected dimension mismatch, a validation `ValueError` is raised, preventing silent inference bugs.

---

### 4. Registered Adapters

#### 1. FSRCNN FP32 (`tinysr`)
- **Backend**: PyTorch (`pytorch`).
- **Precision**: `fp32`.
- **Supported Scales**: `[2, 3, 4]`.
- **Availability**: Always `True` (PyTorch is a core package requirement).

#### 2. FSRCNN INT8 (`tinysr_int8`)
- **Backend**: ONNX Runtime (`onnxruntime`).
- **Precision**: `int8`.
- **Supported Scales**: `[2]`.
- **Availability**: Dynamic. Requires `onnxruntime` importable and the weights file `models/tinysr/fsrcnn_x2_int8.onnx` to exist. 

#### 3. Real-ESRGAN (`real_esrgan`)
- **Backend**: RealESRGAN (`realesrgan` upsampler using PyTorch RRDBNet).
- **Precision**: `fp32`.
- **Supported Scales**: `[2, 4]`.
- **Availability**: Requires `basicsr` and `realesrgan` importable. Exposes a pre-import compatibility patch for `torchvision.transforms.functional_tensor` to resolve torchvision deprecations on Python 3.10+.

#### 4. BasicVSR++ (`basicvsr++`)
- **Backend**: BasicVSR (`basicvsr_backend`).
- **Precision**: `fp32`.
- **Supported Scales**: `[4]`.
- **Temporal/Spatial**: `temporal` (sequence-aware).
- **Availability**: Always `False` (MMCV compilation is blocked on Windows; operates as a capability discovery stub with the reason documented).

---

### 5. Device & CPU Thread Configuration
- **Device Selection**: Configured via `initialize(device="cpu" | "cuda", scale, num_threads)`. Requesting CUDA when unavailable raises a clear `ValueError`.
- **CPU Thread/Core Control**: The `initialize` signature supports `num_threads=None` (uses default backend settings) or `num_threads=N` (requests controlled CPU parallelism).
  - *PyTorch backends* (`tinysr`, `real_esrgan`): Invoke `torch.set_num_threads(N)` when executing on CPU.
  - *ONNX Runtime backend* (`tinysr_int8`): Sets `intra_op_num_threads` via `ort.SessionOptions` and dynamically registers the session.
  - *Unsupported device configuration*: Requesting `num_threads` with `device="cuda"` raises a `ValueError` to ensure thread count constraints are explicitly honored.
  - *Deferred details*: CPU core affinity binding and active process monitoring are deferred to Step 5.3.

---

### 6. Preprocessing / Postprocessing & Output Cropping
- Preprocessing and postprocessing steps are isolated inside the adapter, keeping raw BGR frames as the clean input/output boundary.
- **Real-ESRGAN Output Cropping**: Real-ESRGAN's upsampler can introduce padding or border mismatches when dividing frames into tiles. To protect the strict output validation contract, `RealESRGANAdapter` crops the output frame to the exact dimensions ($H_{out} = H_{in} \times S$, $W_{out} = W_{in} \times S$) before returning it. Generic caller-side resizing is completely avoided.

---

### 7. Real-ESRGAN x2 Compute Path Finding
- **Native vs. Downsampled**: Real-ESRGAN scale=2 operates on a **genuine native x2 compute path**.
- **Evidence**:
  - The model weights are loaded from `RealESRGAN_x2plus.pth`.
  - The underlying generator architecture `RRDBNet` is initialized with `scale=2` parameters.
  - The upsampler wrapper `RealESRGANer` runs with `scale=2`.
- Since the model construction and inference scale match the target output scale factor, no post-inference downsampling is used.

---

### 8. Registry Integration & Discovery
The registry interface (`adaptive_sr/benchmarking/adapters/registry.py`) links registered model IDs to their adapter wrappers. 
Exposed endpoints:
- `get_adapter(model_id: str) -> BaseSRAdapter`
- `list_available_models() -> List[str]` (Filters out unavailable stubs dynamically).
- `get_model_status_report() -> Dict[str, Dict[str, Any]]` (Returns registration and availability details for all models).

---

### 9. Verification & Test Suite
Unit tests in [`tests/test_model_adapters.py`](file:///d:/Full%20Stack/AdaptiveSR/tests/test_model_adapters.py) cover:
- Standard execution flow, input type check, and shape checks.
- Spatial sequence handling vs. temporal sequence requirements.
- Target upscaled dimension validation and silent upscaling prevention.
- Registry mapping and dynamic capability status checks.
- CPU/GPU device validation and uninitialized process errors.
- **CPU Thread Control verification**: Tests that `initialize` accepts `num_threads`, `num_threads=None` preserves default execution, and unsupported non-CPU thread settings fail explicitly.
- **Odd Resolution Cropping verification**: Feeds a non-standard odd resolution frame (`123 x 125`) to the `RealESRGANAdapter` to verify that the internal tiling/padding doesn't cause shape mismatch errors and crops correctly.
- **Real-ESRGAN scale=2 verification**: Confirms that scale=2 is supported by the adapter metadata.
- **Step 5.1 Dataset Smoke Test**: Integration test that loads a real chunk file from the Step 5.1 benchmark manifest, extracts a frame, processes it through the FSRCNN adapter on CPU, and verifies output dimensions match the expected scale factor.

---
---

## Step 5.3 — CPU Affinity + ProcessMonitor Integration

### 1. Purpose & Objectives
Step 5.3 establishes a controlled, reproducible execution environment for CPU benchmarking experiments by restricting the SR inference process to a known set of logical CPU cores while monitoring its resource consumption via process-level telemetry.

> [!IMPORTANT]
> **Step 5.3 provides benchmarking instrumentation and execution control. It does NOT make dynamic resource allocation or runtime placement scheduling decisions.**

---

### 2. CPU Affinity vs. num_threads Distinction
To support empirical experiments mapping latency to CPU resources, Step 5.3 separates hardware core bounds from backend thread counts:
- **Logical CPU Affinity**: The operating system scheduler constraint restricting the process to a specific set of logical core indices.
- **Model thread count (`num_threads`)**: The size of the intra-op compute thread pool initialized by PyTorch or ONNX Runtime.
- **Oversubscription Control**: Restricting a process to $M$ logical cores while requesting $N$ threads ($N > M$) allows testing the effects of thread oversubscription under controlled hardware constraints.

---

### 3. Logical CPU Discovery & Selection
- **Discovery**: Exposes `get_available_cpus() -> List[int]` which dynamically fetches logical indices (e.g. `[0, 1, 2, ..., 11]`) via `psutil`.
- **Deterministic Selection**: Exposes `select_cpu_ids(count: int) -> List[int]`. To ensure reproducibility across runs, the selected core list is fully deterministic (e.g., requesting 4 logical cores on a 12-core host consistently returns `[0, 1, 2, 3]`).

---

### 4. Validation Rules
The configuration model `CPUExecutionConfig` enforces strict validation checks to prevent invalid hardware binds:
- Rejects negative CPU IDs.
- Rejects duplicate CPU ID entries (e.g. `[0, 0]`).
- Rejects empty CPU ID sets.
- Rejects CPU counts or IDs exceeding available host limits (prevents silent core clamping).
- Rejects `num_threads <= 0`.

---

### 5. Guarantees for CPU Affinity Restoration
To prevent the Python runner or development environment from remaining permanently core-bound after a test exits or encounters a failure:
- The context manager `cpu_affinity_context` captures the process's initial affinity mask on enter.
- It applies the restricted mask for the duration of the context.
- Under a `finally` block, it guarantees restoration of the original affinity mask, even if exceptions, test assertions, or model runtime crashes occur inside.

---

### 6. ProcessMonitor Integration & Observability Lifecycle
Step 5.3 integrates the frozen Step 4 `ProcessMonitor` via a background thread wrapper `BenchmarkProcessMonitor`:
- **Background Sampling**: On `start()`, a daemon thread periodically queries the process state via the frozen `ProcessMonitor.snapshot(interval)` and accumulates `ProcessResourceSnapshot` metrics.
- **Termination Cleanup**: On `stop()`, the thread is joined and cleaned up to prevent thread leaks.
- **Lifecycle Coordination**: The `benchmark_execution_context` manager coordinates the complete start-stop lifecycle:
  1. Starts the background process monitor.
  2. Sets the CPU affinity core restriction.
  3. Yields for benchmark workload execution.
  4. Restores the original CPU affinity mask.
  5. Stops the background process monitor thread.

---

### 7. Portability & OS Support
- **Windows (Current Host)**: Uses `psutil.Process().cpu_affinity(...)` to configure core masks.
- **Linux/Azure (Eventual Deployment)**: The `psutil` affinity interface is fully portable across Windows and Linux, ensuring that the same code and tests will execute on Linux edge instances without modification.

---

### 8. Automated Tests
Tests in [`tests/test_cpu_control.py`](file:///d:/Full%20Stack/AdaptiveSR/tests/test_cpu_control.py) cover all functional requirements, verifying:
- CPU discovery, validation, and error boundaries.
- Thread configuration setting and verification in PyTorch and ONNX Runtime backends.
- Safety guarantees (affinity restoration on success and exceptions).
- ProcessMonitor background lifecycle execution, sample accumulation, and thread leak protection.
- Downstream model adapter smoke test compatibility.
- Context execution ordering constraints (affinity active before monitor starts).

---

### 9. Measurement-Control Caveats
- **A. CPU 0 Selection Policy**: Logical CPU ID selection is deterministic and starts from core 0 (e.g., `[0, 1, 2, ...]`). Because logical core 0 typically handles OS interrupts and background services, absolute latency measurements may experience host-dependent noise. Relative comparisons between models remain consistent as the same core policy is applied uniformly across configurations.
- **B. ProcessMonitor Contention**: The ProcessMonitor runs as a background thread inside the same benchmark process, meaning it shares the restricted CPU affinity mask. For small core configurations (especially 1 CPU), the monitor's CPU sampling overhead can introduce contention and influence latency measurements. This is a measurement-system limitation; future steps must use conservative sampling rates and document this impact when interpreting low-core results.
- **C. Context Ordering**: The CPU affinity mask is applied and verified *before* the ProcessMonitor is started. This ensures that the monitor's background thread executes entirely within the constrained hardware layout, providing consistent resource measurements from the first sample.
- **D. Methodological Limits**: These affinity and monitoring hooks represent experimental control mechanisms for benchmarking, not runtime resource allocation or placement schedulers.

---
---

## Step 5.4 — GPU Measurement

> [!IMPORTANT]
> **Step 5.4 provides GPU measurement infrastructure. It does NOT benchmark SR models.**
> No inference timing, FPS, throughput, latency statistics, warmup, PSNR, SSIM, or model comparison is implemented here. Those belong to Steps 5.5–5.6.

---

### 0. GPU Measurement Modes (Correction Pass)

Two distinct measurement modes are provided. They serve **different purposes** and must not be confused.

#### Mode A — Periodic Monitoring (`GPUMonitor`)

A background daemon thread calls `take_gpu_snapshot()` at a configurable interval.

**Purpose:** Sustained workload characterisation. Aggregate utilisation observation across longer-running or repeated inference workloads (Step 5.5+).

**Limitation:** Periodic sampling may **miss a single short-lived SR inference entirely** if the inference completes faster than the sampling interval. Therefore, GPUMonitor utilisation samples must NOT be treated as exact per-operation utilisation measurements.

#### Mode B — Synchronous Snapshots (`take_gpu_snapshot` / `GPUMeasurementBoundary`)

A single point-in-time GPU state capture via `take_gpu_snapshot()`, called explicitly before and after a bounded operation.

**What a before/after snapshot pair CAN reliably provide:**
- `process_gpu_memory_allocated_bytes` delta (process-level, PyTorch allocator)
- `process_gpu_memory_reserved_bytes` delta (process-level, PyTorch allocator)
- `gpu_memory_free_bytes` change (device-wide, NVML)
- `gpu_memory_total_bytes` (constant reference)

**What a before/after snapshot pair CANNOT reliably provide:**
- Peak GPU utilisation during a short inference. NVML utilisation is sampled instantaneously at the moment of the call. If the GPU kernel has already completed, the reported utilisation may be 0% even if the device was fully saturated during execution.

**Conclusion:**

| Mode | Purpose |
|---|---|
| Periodic (Mode A) | Sustained utilisation characterisation |
| Synchronous (Mode B) | Per-operation GPU memory/state boundaries |

> [!WARNING]
> **Neither mode is a substitute for explicit inference latency measurement.**
> Step 5.5 MUST use `torch.cuda.synchronize()` around explicit timing boundaries for accurate latency.

`GPUMeasurementBoundary` encapsulates a before/after snapshot pair and exposes memory delta helper properties (`memory_allocated_delta_bytes`, `memory_reserved_delta_bytes`, `free_memory_delta_bytes`). It contains **no timing or latency fields**.

---

### 0b. CPU / GPU Sampling Interval Policy (Correction Pass)

CPU telemetry (Step 5.3 `BenchmarkProcessMonitor`, default 0.05 s) and GPU telemetry (Step 5.4 `GPUMonitor`, default 0.5 s) use **independently configurable sampling intervals**.

**Rationale:** CPU and GPU telemetry have different collection mechanisms and overhead characteristics:
- CPU measurement (`psutil.Process.cpu_percent`) blocks for the interval duration.
- GPU measurement (NVML + PyTorch allocator) has different query latency and overhead.

**Policy consequences:**
- CPU and GPU utilisation curves must **NOT** be assumed to have identical temporal resolution merely because both are called "utilisation".
- The primary latency measurement in Step 5.5 MUST be performed using explicit timing boundaries around inference — NOT inferred from either monitor's sampling interval.

---

### 0c. nvidia-ml-py Migration (Correction Pass)

`requirements.txt` has been updated from the deprecated `pynvml==13.0.1` to the maintained `nvidia-ml-py` package.

Both packages expose an identical `pynvml` Python module API — all `import pynvml` statements in source code remain valid. The migration is a `requirements.txt`-only change.

**Why:** PyTorch's `torch/cuda/__init__.py` issues a `FutureWarning` flagging `pynvml` as deprecated and recommending `nvidia-ml-py`. Using `nvidia-ml-py` eliminates this warning and avoids changing GPU telemetry semantics mid-campaign.

---

### 1. Purpose & Objectives

Step 5.4 establishes the GPU-side observability layer needed to support future GPU-accelerated SR benchmarking. It is the direct GPU equivalent of Step 5.3's CPU control infrastructure:

| Step 5.3 | Step 5.4 |
|---|---|
| CPU affinity control | GPU device discovery |
| `CPUExecutionConfig` | `GPUDeviceInfo` |
| `BenchmarkProcessMonitor` | `GPUMonitor` |
| `benchmark_execution_context` | `gpu_measurement_context` |
| `ProcessResourceSnapshot` | `GPUSnapshot` |

---

### 2. Files Created

| File | Purpose |
|---|---|
| [`adaptive_sr/benchmarking/gpu_measurement.py`](file:///d:/Full%20Stack/AdaptiveSR/adaptive_sr/benchmarking/gpu_measurement.py) | Core GPU measurement infrastructure |
| [`tests/test_gpu_measurement.py`](file:///d:/Full%20Stack/AdaptiveSR/tests/test_gpu_measurement.py) | 14-area test suite |
| `adaptive_sr/shared/schemas.py` (modified) | `GPUSnapshot` Pydantic model appended |

---

### 3. CUDA Detection

`get_cuda_availability()` returns a dict with three fields:

```python
{
    "status": CUDAAvailability,   # UNAVAILABLE | NO_DEVICE | AVAILABLE
    "device_count": int,
    "device_ids": List[int],
}
```

The `CUDAAvailability` enum distinguishes four states:

| State | Meaning |
|---|---|
| `UNAVAILABLE` | `torch.cuda.is_available()` is False |
| `NO_DEVICE` | CUDA runtime present but zero devices visible |
| `AVAILABLE` | ≥1 CUDA device present and usable |

**Behaviour when CUDA is requested but unavailable:**  
`require_cuda(device_id)` raises `RuntimeError` with a clear, explicit message. There is no silent fall-back from CUDA to CPU.

---

### 4. GPU Discovery

`list_gpus() -> List[GPUDeviceInfo]` enumerates all visible CUDA devices.

- Returns `[]` on a CPU-only machine — this is **not an error**.
- Each device is independently queryable via `get_gpu_info(device_id)`.
- Multiple GPUs are identified independently as `cuda:0`, `cuda:1`, etc.

---

### 5. GPU Device Identity

`GPUDeviceInfo` is a frozen dataclass capturing the minimum stable identity required for benchmark records:

| Field | Required | Source |
|---|---|---|
| `device_id` | ✓ | `torch.cuda.device_count()` |
| `device_name` | ✓ | `torch.cuda.get_device_properties().name` |
| `total_memory_bytes` | ✓ | `torch.cuda.get_device_properties().total_memory` |
| `compute_capability` | optional | `props.major + '.' + props.minor` |
| `cuda_runtime_version` | optional | `torch.version.cuda` |
| `pytorch_cuda_version` | optional | `torch.version.cuda` |
| `driver_version` | optional | NVML `nvmlSystemGetDriverVersion` |

A benchmark record must **never** identify a GPU only as `"GPU"` because multiple devices may exist. `device_name` always carries the hardware name (e.g. `"NVIDIA GeForce RTX 4090"`).

---

### 6. GPU Memory Measurement

Three distinct memory quantities are tracked in `GPUSnapshot` and must **not** be conflated:

| Field | What it measures | Source |
|---|---|---|
| `gpu_memory_total_bytes` | Device-wide total VRAM | NVML / PyTorch props |
| `gpu_memory_free_bytes` | Device-wide free VRAM | NVML only |
| `process_gpu_memory_allocated_bytes` | Live tensor bytes in THIS process | `torch.cuda.memory_allocated()` |
| `process_gpu_memory_reserved_bytes` | Allocator slack for THIS process | `torch.cuda.memory_reserved()` |

**PyTorch allocator semantics:**
- `allocated` = memory currently held by live tensors (actual tensor usage)
- `reserved` = memory held by the caching allocator, including slack from freed tensors
- `reserved ≥ allocated` always

**Device-wide vs. process-level:**  
NVML memory figures are device-wide (all processes). PyTorch allocator figures are process-local. These are fundamentally different quantities and are stored in separate named fields with a `process_` prefix to make the distinction explicit.

---

### 7. GPU Utilization Measurement

GPU utilization is provided via NVML (`pynvml`) when available:

| Field | What it measures |
|---|---|
| `gpu_utilization_percent` | SM (compute) utilization across entire device — NOT process-specific |
| `memory_utilization_percent` | Memory-bus utilization across entire device |

> [!WARNING]
> **NVML utilization is device-wide, not process-specific.**  
> `gpu_utilization_percent = 80%` means 80% of the device's compute units are busy across **all processes**, not just the SR inference process. The future benchmark must clearly document this limitation when interpreting results.

---

### 8. NVML Availability

`pynvml==13.0.1` is listed in `requirements.txt`. However, NVML may fail to initialise even when `pynvml` is importable (e.g., missing NVIDIA drivers, VM without GPU passthrough).

**`NVMLContext`** handles this gracefully:
- Wraps `pynvml.nvmlInit()` and `nvmlShutdown()` in a context manager.
- On failure, sets `available=False` — callers branch on this flag.
- Never raises on a CPU-only host.

**Graceful degradation invariants (strict):**

| Field | When NVML unavailable |
|---|---|
| `gpu_utilization_percent` | `None` — **NOT 0** |
| `memory_utilization_percent` | `None` — **NOT 0** |
| `gpu_memory_free_bytes` | `None` |
| `nvml_available` | `False` |
| `utilization_source` | `"pytorch_allocator"` or `"unavailable"` |

Zero (0%) utilization is a **meaningful measurement** that must not be fabricated.

---

### 9. Sampling Lifecycle

`GPUMonitor` provides a start/stop background sampling lifecycle:

```python
monitor = GPUMonitor(device_id=0, sample_interval=0.5)
monitor.start()           # starts daemon thread
# ... workload runs ...
monitor.stop()            # joins thread — no zombie threads
samples = monitor.get_samples()  # List[GPUSnapshot]
```

Or via context manager (preferred — guaranteed cleanup):

```python
with GPUMonitor(device_id=0) as monitor:
    run_workload()
samples = monitor.get_samples()
```

Or via `@contextmanager` wrapper:

```python
with gpu_measurement_context(device_id=0) as monitor:
    run_workload()
samples = monitor.get_samples()
```

**Lifecycle guarantees:**
- `stop()` is called in a `finally` block — the thread stops even on exception.
- `stop()` is idempotent — calling it multiple times is safe.
- Background thread is a daemon — it will not prevent process exit.
- Thread is joined with a 5-second timeout to prevent indefinite blocking.

---

### 10. Sampling Interval

Default: **0.5 seconds** (documented).

- Conservative enough to avoid significant monitoring overhead.
- Configurable: `GPUMonitor(device_id=0, sample_interval=0.1)`.
- Step 5.5 may override this value.
- Do NOT hard-code a lower interval without measuring overhead impact.

> [!NOTE]
> GPU monitoring itself introduces non-zero overhead. This overhead is **not subtracted** from measurements. To maintain comparability, the same monitoring configuration must be used across all comparable benchmark runs.

---

### 11. GPUSnapshot Schema

Each observation in `List[GPUSnapshot]` (in `adaptive_sr/shared/schemas.py`) contains:

```python
GPUSnapshot(
    timestamp                          : str       # ISO-8601 UTC + 'Z'
    device_id                          : int       # CUDA device index
    gpu_name                           : str       # hardware name
    gpu_utilization_percent            : float|None  # NVML device-wide; None if NVML absent
    memory_utilization_percent         : float|None  # NVML device-wide; None if NVML absent
    gpu_memory_total_bytes             : int|None    # device-wide total VRAM
    gpu_memory_free_bytes              : int|None    # device-wide free (NVML only)
    process_gpu_memory_allocated_bytes : int|None    # process-level (torch allocator)
    process_gpu_memory_reserved_bytes  : int|None    # process-level (torch allocator)
    nvml_available                     : bool        # explicit NVML availability flag
    utilization_source                 : str         # 'nvml' | 'pytorch_allocator' | 'unavailable'
)
```

**None semantics:** `None` means genuinely unavailable — not 0, not unmeasured.

---

### 12. Process vs. Device GPU Utilization

NVML provides only **device-level** utilization. It does not identify which process is responsible for the observed utilization.

The benchmark must clearly distinguish:
- `"GPU utilization of device 0"` (device-wide, all processes)
- `"GPU utilization caused by this SR process"` (per-process — **not available** from NVML)

`utilization_source="nvml"` documents that device-wide data was used, ensuring this limitation is explicit in every snapshot.

---

### 13. Multi-GPU Support

- `list_gpus()` returns one `GPUDeviceInfo` per visible device.
- `GPUMonitor(device_id=N)` monitors a specific device (`cuda:0`, `cuda:1`, etc.).
- `validate_device_id()` rejects requests for non-existent device IDs with explicit messages.
- Multi-GPU *inference* and load balancing are NOT implemented.

---

### 14. Device Validation

`validate_device_id(device_id)` rejects:

| Input | Error |
|---|---|
| Negative ID | `ValueError: GPU device_id must be non-negative` |
| ID ≥ device count | `RuntimeError: Requested CUDA device N, but only …` |
| CUDA unavailable | `RuntimeError: CUDA is not available on this host` |

There is **no silent fall-back** to another GPU.

---

### 15. PyTorch Integration

- GPU device state is NOT changed globally.
- `get_gpu_info(device_id)` and `take_gpu_snapshot(device_id)` query a specific device without altering `torch.cuda.current_device()`.
- The Step 5.2 adapter remains responsible for model execution.
- Step 5.4 wraps measurement *around* that execution.

---

### 16. CUDA Asynchronous Execution Note

> [!IMPORTANT]
> CUDA operations are **asynchronous**: the CPU-side `model()` call may return before the GPU has finished executing.  
>
> Therefore, **Step 5.5 MUST call `torch.cuda.synchronize()` around timing boundaries** to ensure the GPU has completed work before recording latency.  
>
> Step 5.4 does **not** implement timing and therefore does **not** call `synchronize()`. This is a documented methodological requirement for Step 5.5 implementers.

---

### 17. GPU Warmup Note

> [!NOTE]
> Future benchmark execution in Step 5.5 MUST distinguish:
> - CUDA context creation overhead (first CUDA call in the process)
> - Model weight loading to VRAM
> - First-inference overhead (JIT compilation, cache misses)
> - Warmed steady-state inference
>
> Step 5.4 provides measurement infrastructure only. Warmup is NOT implemented.

---

### 18. No-GPU Behavior

On a CPU-only machine:
- `get_cuda_availability()` → `CUDAAvailability.UNAVAILABLE`
- `list_gpus()` → `[]` (not an error)
- `GPUMonitor(device_id=0)` → raises `RuntimeError` with a clear message
- All test suite infrastructure tests pass
- GPU-specific runtime tests are skipped with `pytest.mark.skipif` and an explicit reason

---

### 19. Limitations

- **NVML is device-wide only.** Per-process GPU utilization is not available via NVML; only per-process memory is available via the PyTorch caching allocator.
- **Monitoring overhead is not subtracted.** The NVML polling and PyTorch allocator calls introduce non-zero overhead. Run benchmarks with consistent monitoring configurations.
- **Asynchronous CUDA.** GPU work may still be executing after the CPU call returns. Step 5.5 must handle synchronization.
- **pynvml deprecation warning.** The currently installed `pynvml==13.0.1` is flagged by PyTorch as deprecated in favour of `nvidia-ml-py`. This warning does not affect functionality for Step 5.4. The dependency should be updated in a future maintenance step.

---

### 20. Automated Tests

Tests in [`tests/test_gpu_measurement.py`](file:///d:/Full%20Stack/AdaptiveSR/tests/test_gpu_measurement.py) cover all 14 required areas:

| # | Area | GPU Required? |
|---|---|---|
| 1 | CUDA availability detection | No (mocked) |
| 2 | GPU enumeration | No (mocked) |
| 3 | Device metadata retrieval | Yes (skipped if absent) |
| 4 | Invalid GPU ID rejection | No |
| 5 | GPU memory snapshot schema | No |
| 6 | GPU utilization availability handling | No (mocked) |
| 7 | Sampling lifecycle | Yes (skipped if absent) |
| 8 | Sampling interval configuration | Yes (skipped if absent) |
| 9 | Clean monitor shutdown | Yes (skipped if absent) |
| 10 | Exception-safe shutdown | Yes (skipped if absent) |
| 11 | Multi-GPU device selection | No (mocked) |
| 12 | No-GPU graceful behavior | No |
| 13 | Step 5.2 CUDA adapter compatibility | No / Yes |
| 14 | Existing Step 0–4 tests (regression) | No |

**Test design constraints (enforced):**
- No exact GPU utilization percentage assertions.
- No exact memory byte value assertions.
- Structural / range / invariant checks only.
- No GPU faking or global CUDA monkey-patching.

---
---

## Step 5.5 — Inference Benchmark Harness

### 1. Purpose & Core Execution Flow
Step 5.5 converts the execution-control and measurement infrastructure from Steps 5.2–5.4 into a reproducible inference benchmark harness. It isolates model inference from external overheads (such as frame loading and decoding) to accurately characterize model performance across CPU and GPU hardware configurations.

The harness executes the following sequence for each benchmark case:
1. **Validate Configuration:** Rejects invalid setups (e.g., negative warmup/measured counts, unsupported scales, invalid CPU execution configurations on CUDA).
2. **Prepare Input:** Retrieves the chunk from the Step 5.1 dataset. If the model is spatial, the first frame is loaded into memory as a single numpy array. If temporal, all frames are loaded into memory as a list of numpy arrays.
3. **Initialize Adapter:** Initializes the selected model adapter (`FSRCNNAdapter`, `FSRCNNInt8Adapter`, or `RealESRGANAdapter`) on the target device, allocating parameters and threads. This phase is excluded from the steady-state latency measurements.
4. **Configure Telemetry Context:** Restricts logical CPU affinity and triggers process-level CPU monitoring (on CPU) or initializes GPU monitoring (on CUDA).
5. **Warm up:** Performs the configured number of warmup executions (default: 3) to absorb JIT compilation, caching, and allocator overhead.
6. **Steady-State Trials timing:** Executes the trials (default: 20).
   - If CUDA: Synchronizes before starting the timer, performs inference, and synchronizes before stopping the timer using `torch.cuda.synchronize(device_id)`.
   - If CPU: Measures elapsed monotonic time directly using `time.perf_counter()`.
7. **Validate Outputs:** Confirms output shape matches the upscaling scale laws exactly outside the timed inference loop.
8. **Collect Telemetry:** Aggregates statistics, collects resource snapshots, and cleans up adapter sessions on completion.

---

### 2. Device Policy & Resource Controls
- **CPU Benchmarks:** Execute inside the `benchmark_execution_context` which binds logical CPU core affinity (`cpu_ids`) and launches the background `BenchmarkProcessMonitor` sampling process-level telemetry.
- **CUDA Benchmarks:** Execute on a specific GPU device ID (`cuda:N`), ensuring `torch.cuda.synchronize` boundaries are respected. Launch a background `GPUMonitor` sampling device utilization and record synchronous VRAM snapshots (`GPUMeasurementBoundary`) before/after trials.
- **Oversubscription:** The configuration preserves separate dimensions for logical core count and backend thread pool sizing (`num_threads`), allowing explicit oversubscription studies.
- **No silent fallback:** Requesting CUDA on a CPU-only environment or requesting invalid scales raises an explicit error at validation time.

---

### 3. Latency Statistics & Percentile Methodology
The harness collects raw float latencies (in seconds) for each trial and computes:
- Count
- Mean, Median, Min, and Max latency
- Standard deviation
- p95 latency (using NumPy's deterministic linear percentile interpolation method: `np.percentile(latencies, 95)`)

All statistics are kept at full floating-point precision and are only rounded during human-readable representation.

---

### 4. Throughput & FPS Definition
Model throughput is defined as:
$$\text{Throughput (FPS)} = \frac{1}{\text{mean\_latency\_seconds}}$$
It measures raw inference capacity under sequential execution (batch size = 1) and is mathematically isolated from:
- Video file decoding
- Host/Process resource monitoring sampling overhead
- Serialization and output writing
- Logging and database activities

---

### 5. Failure Handling & Trials Inspectability
- Individual trials that encounter runtime exceptions (e.g., CUDA out-of-memory or model failures) are captured gracefully.
- The harness logs successful vs failed trials and captures error traceback details.
- Every individual trial's performance latency and memory delta bounds remain inspectable in the resulting dictionary schema (`metadata["trials"]`).

---

### 6. Verification & Automated Tests
Automated tests in [`tests/test_benchmark_harness.py`](file:///d:/Full%20Stack/AdaptiveSR/tests/test_benchmark_harness.py) cover all functional constraints, verifying:
- Config schema checks and invalid parameters rejection.
- Mocked CPU paths with affinity bindings and `ProcessMonitor` lifecycle verification.
- Mocked GPU paths verifying `torch.cuda.synchronize` and `GPUMonitor` integration.
- Correct mathematical statistics calculations and percentile method documentation.
- Graceful skipping of GPU runs in environments without CUDA.
- Graceful error capturing for single-trial failure.
- Direct integration smoke test with the real lightweight FSRCNN model (`tinysr`) on CPU.

---

### 7. Pre-5.5 Hardening Pass Addendum

#### A. Synthetic Corpus Non-Keyframe GOP Boundary Note
> [!NOTE]
> The synthetic benchmark corpus uses controlled encoding and does not fully exercise production H.264 GOP/keyframe boundary behavior. Production video may contain chunk boundaries that do not align with keyframes. Therefore, Step 5.1 validates benchmark-input preparation and association, but does NOT claim to validate arbitrary production H.264 chunk boundary correctness.

#### B. Real-ESRGAN x2 Verification Method and Result
Verification method: Inspected the actual model construction and weight paths inside `realesrgan_backend.py`. The backend initializes a native `RRDBNet` generator architecture directly specifying `scale=scale` (which is `2` in our configuration), loading the dedicated `RealESRGAN_x2plus.pth` checkpoint, and passing it to the `RealESRGANer` upsampler wrapper. We verified that the model's pre-postprocessing/pre-crop output tensor dimensions are exactly double the input dimensions (e.g. `(720, 1280, 3)` for a `(360, 640, 3)` input), confirming no intermediate `x4` inference computation or subsequent downsampling to `x2` is executed. Therefore, the execution path is a native x2 model compute path.

#### C. Real-ESRGAN Crop Visibility
Real-ESRGAN output dimensions are strictly validated. If any crop operations occur, they are made visible in the output metadata under `crop_metadata` with:
- `crop_applied`: bool
- `pre_crop_width` & `pre_crop_height`
- `final_width` & `final_height`
- `crop_pixels_if_available`

#### D. CPU Selection Policy & ProcessMonitor Contention
- CPU exclusion allows sensitivity analysis by excluding CPU 0 (`exclude_cpu_ids=[0]`), reducing background noise from common system interrupts.
- Core-0 inclusion remains the default for backward compatibility.
- *ProcessMonitor Contention:* `ProcessMonitor` runs inside the benchmark process, so it is subject to the same affinity mask. At very small affinity masks (e.g. 1 logical core), monitor sampling may contend with inference. Interpret low-core results with this limitation. Do not mathematically subtract overhead.

#### E. GPU Sampling Limitations & Shared GPU Contamination
- *GPU Sampling Limits:* A 0.5-second periodic GPUMonitor is preserved. However, for short single-chunk inference operations, especially operations whose duration is substantially below 0.5 seconds, periodic NVML utilization samples may miss the GPU-busy interval entirely. Therefore, periodic GPU utilization MUST NOT be treated as a reliable per-inference GPU-utilization measurement. For the benchmark campaign, latency, GPU memory, and GPU availability/device identity remain usable measurements. Periodic GPU utilization should be interpreted primarily as sustained-load / aggregate utilization characterization. We do not generate or infer a fake per-inference utilization percentage when no valid sample overlaps the workload, representing unavailable/insufficient utilization data explicitly.
- *Shared GPU Contamination:* NVML utilization is device-wide. On shared machines (such as shared Azure GPU VMs), other processes contribute to observed utilization. NVML measurements do not represent exclusive AdaptiveSR workload.

#### F. P95 Confidence & Warmup Telemetry Separation
- *P95 confidence:* At `n=20`, p95 statistics are explicitly annotated as `"Exploratory p95; limited tail resolution at n=20."`.
- *Warmup Telemetry Separation:* Resource telemetry is collected separately during the warmup phase and measured trials. Results record `warmup_resource_summary` and `resource_summary` as distinct structures.

#### G. Session-to-Session Support
Harness supports repeated independent sessions under the same case mapping. Sessions preserve their raw values and session-level metadata (CPU/GPU info, timestamps, configurations) without pooling them, allowing within-session and between-session variance analysis.

#### H. Thermal & Throttling Limitations
Laptop thermal state, CPU thermal throttling, GPU thermal throttling, and background OS activity may significantly contribute to between-session variance. When reliable thermal telemetry is unavailable from the execution environment, `thermal_state` must be recorded explicitly as `"not_measured"`. The absence of thermal telemetry must be explicit, rather than silently assumed to mean "thermally stable".

#### I. Session Count & Decision-Quality Policy
We define clear session policies to govern benchmarking workflows:
- **Smoke test / Development benchmark:** Typically runs 1 session to verify code functionality quickly.
- **Decision-quality benchmark:** Requires a **minimum of 3 independent sessions** per configuration. The CPU-vs-GPU decision must be supported by multiple runs to analyze variance and prevent decisions based on a single outlier session. Sessions are preserved individually and must not be pooled.


