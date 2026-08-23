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
