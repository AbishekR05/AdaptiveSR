# Step 1 — Pre-Implementation Audit: Video & Content Profiling

This document performs a pre-implementation audit of the legacy AdaptiveSR codebase (`src/` and `benchmark/`) to identify reusable modules, extractable content features, chunking conventions, and compatibility constraints for Step 1.

---

## 1. Legacy Component Analysis

The following table audits every major class, function, and utility in the legacy codebase:

| Path / File | Component Name | Description | Classification | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| [`src/modules/video_loader.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/video_loader.py) | `VideoLoader` | Extracts metadata (resolution, frame rate, frame count, duration, has_audio, codec) via `ffprobe` and `OpenCV`. | **REUSE WITH MODIFICATION** | Excellent foundation for source metadata retrieval. Needs to support parsing specific segmented video chunks and sub-sequences. |
| [`src/modules/frame_extractor.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/frame_extractor.py) | `FrameExtractor` | Iterates and yields frames sequentially from OpenCV VideoCapture with millisecond timestamps. | **REUSE WITH MODIFICATION** | Solid frame generator. Needs to be modified to accept start/end frame boundaries to enable extraction of specific chunk intervals. |
| [`src/modules/scene_analyzer.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/scene_analyzer.py) | `analyze_frame` | Calculates normalized raw visual metrics (edge density via Canny, texture/blur via Laplacian variance, motion via absolute differences). | **REUSE WITH MODIFICATION** | Core content feature extractor. The motion estimation logic must be modified to scale dynamically based on input FPS to prevent temporal underestimation. |
| [`src/modules/complexity_estimator.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/complexity_estimator.py) | `estimate_complexity` | Computes a weighted visual complexity score from raw frame metrics. | **REUSE WITH MODIFICATION** | Clean logic, but needs to be modified to calculate chunk-level summaries (min, max, mean, variance) rather than single frame outputs. |
| [`src/modules/encoder.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/encoder.py) | `VideoEncoder` | Encodes raw frames to x264 via FFmpeg pipe and merges original audio tracks. | **NOT NEEDED** | This is used post-SR for stream rendering in client player emulation. Step 1 is restricted to pre-transmission source profiling. |
| [`src/modules/decision_engine.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/decision_engine.py) | `DecisionEngine` | Selects SR models dynamically based on device load and content metrics. | **NOT NEEDED** | Evaluated at runtime at the Edge Server. Not part of the pre-transmission profiling step. |
| [`src/modules/enhancement_engine.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/enhancement_engine.py) | `EnhancementEngine` | Runs FSRCNN or Real-ESRGAN super-resolution model inference on frame buffers. | **NOT NEEDED** | Upscaling is a runtime Edge processing step. Not part of pre-transmission profiling. |
| [`src/modules/device_monitor.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/device_monitor.py) | `DeviceMonitor` | Spawns background thread to monitor host GPU, CPU, RAM, and stream FPS. | **NOT NEEDED** | Host profiling occurs dynamically during active playback. Unrelated to source video profile indexing. |
| [`benchmark/compute_quality_metrics.py`](file:///d:/Full%20Stack/AdaptiveSR/benchmark/compute_quality_metrics.py) | `compute_metrics_for_videos` | Computes PSNR, SSIM, and LPIPS by comparing ground-truth and enhanced output videos. | **POTENTIAL DATA LEAKAGE** | These quality metrics require post-SR output comparisons. Recommending or relying on them during pre-transmission profiling creates severe information leakage. |
| [`benchmark/generate_dataset.py`](file:///d:/Full%20Stack/AdaptiveSR/benchmark/generate_dataset.py) | `generate_video_pair` | Creates synthetic testing footage (simple, complex, mixed categories) using OpenCV. | **NOT NEEDED** | Used to create synthetic mock assets. Step 1 operates on existing files. |

---

## 2. Feature Audit

The following table summarizes all content features calculated in the legacy code and evaluates their suitability for Step 1:

| Feature | Existing Implementation | Computation Cost | Video/Chunk Level | Source-Side Available? | Relevant to SR? | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Motion** | Absolute frame-to-frame pixel differences normalized to `[0,1]`. | Low | Frame (aggregatable to chunk) | Yes | Yes (High motion reduces SR visual benefit and increases temporal dependency). | **Carry Forward (with dynamic FPS scaling)** |
| **Texture Density** | Normalized variance of the Laplacian filter output. | Medium | Frame (aggregatable to chunk) | Yes | Yes (High texture density benefits significantly from SR reconstruction). | **Carry Forward** |
| **Edge Density** | Ratio of positive Canny edge pixels. | Medium | Frame (aggregatable to chunk) | Yes | Yes (Sharp edges represent optimal targets for SR sharpening). | **Carry Forward** |
| **Blur Clarity** | Laplacian variance scaled to blur thresholds. | Low (reused Laplacian) | Frame (aggregatable to chunk) | Yes | Yes (Highly blurred source frames render high-scale SR redundant). | **Carry Forward** |
| **Spatial Complexity** | Weighted sum of motion, edges, texture, and blur. | Extremely Low | Frame (aggregatable to chunk) | Yes | Yes (Overall metric mapping directly to ABR and SR scale decisions). | **Carry Forward** |

---

## 3. Chunking Audit

We inspected the legacy codebase for segment splitting mechanisms:
* **Current Status**: **None**. The legacy pipeline processes input videos as single continuous files. The segment files in Step 0 (e.g. `0000.mp4`, `0001.mp4`) were pre-divided by external scripts.
* **Deterministic Boundaries**: No deterministic frame-boundary or time-boundary chunking exists in the legacy Python modules.
* **Audio handling**: The legacy `VideoEncoder` merges audio tracks at the end of execution using FFmpeg, but does not segment audio stream packets into individual chunk containers.
* **Metadata persistence**: There is no schema or utility to persist chunk metadata indices.
* **Step 1 Requirements**: A new deterministic, source-side chunking utility is required to split the source video into chunks (e.g., 2.0s duration) and generate an index matching these segments.

---

## 4. FPS Audit (30 / 60 / 120 FPS Compatibility)

Running the legacy algorithms on higher frame rate video exposes several compatibility flaws:

### Temporal Underestimation (Motion Scale Factor)
The legacy motion calculation compares frame $t$ to frame $t-1$:
$$\text{raw\_motion} = \text{mean}(|I_{t} - I_{t-1}|)$$
Because the time gap between consecutive frames decreases as FPS rises ($\approx 33.3\text{ ms}$ at 30 FPS vs. $\approx 8.3\text{ ms}$ at 120 FPS), the raw motion delta is naturally smaller. Using the static `MOTION_SCALE_FACTOR = 4.0` causes the model to severely underestimate scene activity on 120 FPS source footage.
* **Mitigation**: Scale the motion factor dynamically:
  $$\text{adjusted\_motion\_scale} = 4.0 \times \frac{\text{video\_fps}}{30.0}$$
  Alternatively, sample frame diffs over a constant temporal window (e.g., comparing frame $t$ with frame $t - (\text{fps}/30)$).

### Real-Time Performance Constraints
At 120 FPS, each frame must be processed in under **8.3 ms** to maintain real-time streaming. Running heavy edge filters (Canny edge detection + Laplacian variance + difference calculations) sequentially on every frame at 1080p/4K resolution will bottleneck the pipeline.
* **Mitigation**: Introduce downsampling or spatial frame skipping during profiling (e.g., profiling every $N$-th frame for content indicators) to optimize performance.

---

## 5. Concise Recommendations

### A. Components to Reuse
* `VideoLoader` ([`video_loader.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/video_loader.py)): For basic metadata parsing.
* `FrameExtractor` ([`frame_extractor.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/frame_extractor.py)): To iterate video frames.
* `SceneDescriptor` ([`state_types.py`](file:///d:/Full%20Stack/AdaptiveSR/src/utils/state_types.py)): Reused for representing profile data.

### B. Components Requiring Modification
* `analyze_frame` ([`scene_analyzer.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/scene_analyzer.py)): Scale motion coefficient dynamically based on source FPS.
* `estimate_complexity` ([`complexity_estimator.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/complexity_estimator.py)): Aggregate metrics across chunk intervals (averaging, recording max spikes).

### C. Components to Discard
* `VideoEncoder` ([`encoder.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/encoder.py)): Unused during source profiling.
* `DecisionEngine`, `EnhancementEngine`, `DeviceMonitor`: Part of client/edge streaming runtime, not profiling.

### D. Features Worth Carrying into Step 1
* **Motion, Texture, Edges, Blur, and Overall Complexity**: Critical spatial/temporal profiles needed to map optimal SR models.

### E. Features that should NOT be used (Data Leakage)
* **PSNR, SSIM, and LPIPS**: These require comparison with final SR output frames which do not exist source-side before streaming starts. Using them creates severe information leakage.

### F. Missing Capabilities Step 1 Must Implement
* **Deterministic Segment Splitter**: Python/FFmpeg interface to slice a source video into chunks of specified duration (e.g. 2-second segments) with deterministic frame counts.
* **Profile Dataset Exporter**: Exporter to write chunk metrics (JSON format) detailing `chunk_id`, `start_frame`, `end_frame`, `motion`, `texture`, `edges`, `blur`, `complexity`, and file hashes.

### G. Potential Data-Leakage Risks
* Ensure the profiling metadata is strictly computed from raw source pixels. No network throughput indicators, edge server execution timers, or future runtime states can be included in the source profile index.

### H. 30/60/120 FPS Compatibility Risks
* Hardcoded frame indices (like assuming chunk 1 is frames 0-60) must be avoided. Frame offsets must be dynamically calculated based on the actual parsed FPS:
  $$\text{chunk\_frames} = \text{chunk\_duration} \times \text{parsed\_fps}$$
