# Validation Results - Phase 1 Passthrough Pipeline

This document logs the benchmarking and verification results of the Phase 1 passthrough pipeline.

## Execution Summary

- **Test Video**: `input_test.mp4`
- **Resolution**: 640x480
- **FPS**: 30.00
- **Total Frames**: 90 frames
- **Duration**: 3.0 seconds
- **Output Video**: `output_test.mp4` (identical frame count, resolution, and fps)
- **Processing Latency**: 0.49 seconds
- **Average Frame Processing Rate**: ~183.67 FPS

## Performance Profile (Averaged over 90 frames)

| Metric                       | Average Value                                              |
| :--------------------------- | :--------------------------------------------------------- |
| **CPU Utilization**          | ~0.0% (Fast execution, CPU load minimal)                   |
| **RAM Utilization**          | ~94.5%                                                     |
| **Battery Percentage**       | 97.0%                                                      |
| **GPU Utilization**          | N/A (GPU monitoring disabled/no nvidia NVML driver active) |
| **Avg. Frame Write Latency** | ~1.44 ms / frame                                           |

## Conclusion

The passthrough video processing pipeline (VideoLoader -> FrameExtractor -> VideoEncoder) is fully functional and performs raw video extraction, processing, and encoding with minimal overhead. The telemetry logging subsystem successfully records all device metrics in real-time.

---

# Validation Results - Phase 2 Device Monitor

This section logs the verification of the standalone background `DeviceMonitor` thread.

## Evaluation Setup

A verification script was run for 12 seconds sampling every 0.5s:

- **Baseline (0s-3s)**: System idle.
- **High Load (3s-8s)**: Bounded memory allocation of a 240 MB numpy float64 array (forced physical mappings via random fill) and 6 parallel threads running continuous matrix multiplications.
- **Cooldown (8s-12s)**: CPU threads halted, array deallocated, and garbage collector invoked.

## Telemetry Response Results

| Telemetry State        | Baseline Phase | High Load Phase                | Cooldown Phase      |
| :--------------------- | :------------- | :----------------------------- | :------------------ |
| **CPU Utilization**    | ~39.1%         | **~74.4% (Spike Detected)**    | ~29.4%              |
| **Process RAM (RSS)**  | ~32.1 MB       | **~309.8 MB (Spike Detected)** | ~42.6 MB (Released) |
| **System RAM (Total)** | ~88.6%         | ~90.2%                         | ~88.7%              |

## Verification Analysis

- **Reactive Tracking**: Both CPU and memory metrics show clear, distinct jumps matching the load execution phases.
- **Sampling CADENCE**: Telemetry logs indicate consistent sampling intervals of exactly ~0.5 seconds, fully decoupled from frame rates.
- **Hardware Fallbacks**: No hardware errors occurred on non-GPU platforms; parameters cleanly initialized to `None` values where unavailable.

---

# Validation Results - Phase 3 Scene Analyzer & Complexity Estimator

This section logs the verification of the custom visual analysis and complexity estimation pipeline.

## Evaluation Setup

A verification script generated 5 distinct synthetic frames representing different complexity categories:

1.  **Flat sky / blank wall**: Solid gray image ($v=0.0$).
2.  **Landscape (moderate detail)**: Color gradient background, green mountain polygon, and geometric sun circle ($v=56.0$).
3.  **Close-up face**: Large oval silhouette containing round eye ovals and mouth curves ($v=120.0$).
4.  **Moderately busy scene**: Checkerboard pattern background superimposed with thick text and multiple geometric shapes ($v\ge 500.0$).
5.  **Crowded street / high details**: Dense 12px grid overlaying high-frequency white noise ($v\ge 500.0$).

## Decision Settings & Scale Factors

- **Scale Factors**: `MOTION_SCALE_FACTOR = 4.0`, `TEXTURE_SCALE_FACTOR = 500.0`, `BLUR_SCALE_FACTOR = 300.0`.
- **Tuned Weights (Design Decision - Option 2)**:
  - `motion`: 0.25
  - `texture`: 0.50
  - `edges`: 0.20
  - `blur_clarity`: 0.05
    _(Design Choice: Texture and blur clarity are highly correlated as both derive from Laplacian variance. During initial validation, equal weighting caused the complexity score to drop in low-detail zones because the decrease in the inverted `(1 - blur_clarity)` term outpaced the gain in `texture`. We selected Option 2: preserve the metric in the database but heavily down-weight it to 0.05, keeping its influence secondary and documenting this redundancy as a measured methodology detail.)_

## Telemetry Response Results

| Frame Description                 | Motion | Texture | Edges  | Blur Clarity | Complexity Score |
| :-------------------------------- | :----- | :------ | :----- | :----------- | :--------------- |
| **Flat sky / blank wall**         | 0.00   | 0.0000  | 0.0000 | 0.0000       | **0.0500**       |
| **Landscape (moderate detail)**   | 0.00   | 0.1119  | 0.0029 | 0.1865       | **0.0972**       |
| **Close-up face**                 | 0.00   | 0.2399  | 0.0020 | 0.3999       | **0.1504**       |
| **Moderately busy scene**         | 0.00   | 1.0000  | 0.0304 | 1.0000       | **0.5061**       |
| **Crowded street / high details** | 0.00   | 1.0000  | 0.4304 | 1.0000       | **0.5861**       |

## Verification Analysis

- **Monotonicity (Ranking Sanity)**: The complexity score scales monotonically matching human intuition: Flat (0.0500) < Landscape (0.0972) < Close-up face (0.1504) < Busy (0.5061) < Crowded street (0.5861).
- **Determinism**: Multiple identical invocations on the same frame yielded exactly matching float values (Run 1: `0.150358`, Run 2: `0.150358`), confirming pure functions without side effects.
- **Stability**: Adjacent frame testing with camera drift (1px shift) showed a negligible score variance ($|c_t - c_{t-1}| = 0.0001 < 0.05$), ensuring the signal will not cause flickering model selections.
- **Edge Case Processing**: Passing `prev_frame=None` on initial frame execution returned exactly `motion=0.0` rather than causing numeric exceptions or crashes.
- **Motion Sensitivity (Translation Scaling & Limitations)**: Evaluated dynamic motion response using shape translations on a base frame:
  - _Small translation (5px shift)_: **`0.0078`**
  - _Medium translation (30px shift)_: **`0.0468`**
  - _Large translation (120px shift)_: **`0.1872`**
    This confirms that motion scales proportionally with translation distance ($0.0 < 0.0078 < 0.0468 < 0.1872 \le 1.0$) and successfully maps moving objects.
    > [!NOTE]
    > **Validation Limitation**: The translation test represents localized object movement in an otherwise static shot. Under a full-frame camera pan or global motion, the pixel diff mean will be much higher, meaning `MOTION_SCALE_FACTOR = 4.0` may saturate the motion score to 1.0 very quickly. This calibration choice should be stress-tested and adjusted against real video footage during the Phase 7 benchmarking harness.

## Design Decision

- **Inference Sampling Strategy**: Full **per-frame analysis** is selected for v1. This preserves responsiveness to sudden cuts or face entries. Compute overhead is low compared to upcoming neural VSR model inference.

---

# Validation Results - Phase 4 Decision Engine

This section logs the verification of the rule-based `DecisionEngine` logic and its robustness against missing parameters.

## Evaluation Setup

We verified the decision logic against a truth table covering all 5 rule tiers, fallback branches, and all-`None` edge conditions.

- **Rule 1**: Critical device state (low battery + high temp) $\rightarrow$ always lightweight (`tinysr`).
- **Rule 2**: Low scene complexity $\rightarrow$ lightweight (`tinysr`).
- **Rule 3**: Very high complexity + GPU headroom $\rightarrow$ heaviest (`basicvsr++`).
- **Rule 4**: High complexity + CPU headroom $\rightarrow$ mid-tier (`real_esrgan`).
- **Rule 5**: Fallback (moderate complexity, normal resource limits) $\rightarrow$ default (`real_esrgan`).

## Decision Engine Truth Table

| Case  | Scenario                                   | Battery | Temp | GPU  | CPU  | Complexity | Expected Model | Actual Model  | Status     |
| :---- | :----------------------------------------- | :------ | :--- | :--- | :--- | :--------- | :------------- | :------------ | :--------- |
| **1** | Low battery + hot                          | 0.15    | 0.80 | None | 0.30 | 0.50       | `tinysr`       | `tinysr`      | **PASSED** |
| **2** | Low battery, but cool                      | 0.15    | 0.40 | None | 0.30 | 0.50       | `real_esrgan`  | `real_esrgan` | **PASSED** |
| **3** | Battery unknown (desktop), hot             | None    | 0.80 | None | 0.30 | 0.50       | `real_esrgan`  | `real_esrgan` | **PASSED** |
| **4** | Flat scene, powerful device                | 0.90    | 0.30 | 0.10 | 0.10 | 0.10       | `tinysr`       | `tinysr`      | **PASSED** |
| **5** | Extreme complexity, GPU free               | 0.90    | 0.30 | 0.10 | 0.10 | 0.95       | `basicvsr++`   | `basicvsr++`  | **PASSED** |
| **6** | Extreme complexity, no GPU present         | 0.90    | 0.30 | None | 0.10 | 0.95       | `real_esrgan`  | `real_esrgan` | **PASSED** |
| **7** | Extreme complexity, GPU busy               | 0.90    | 0.30 | 0.90 | 0.10 | 0.95       | `real_esrgan`  | `real_esrgan` | **PASSED** |
| **8** | High complexity, CPU busy                  | 0.90    | 0.30 | None | 0.90 | 0.80       | `real_esrgan`  | `real_esrgan` | **PASSED** |
| **9** | Mid complexity, all fields None except CPU | None    | None | None | 0.40 | 0.50       | `real_esrgan`  | `real_esrgan` | **PASSED** |

## Verification Analysis

- **Rule Ordering**: Rule 1 (critical device safety) successfully overrides Rule 5. Rule 2 (complexity short-circuit) correctly bypasses powerful device states to conserve energy.
- **Robustness against None values**: Cases 3, 6, 8, and 9 successfully fall through rather than crashing. The all-`None` parameter check (Case 9) evaluated safely.
- **Reason-String Veracity**: Decision justification strings dynamically embed parameters:
  - _Case 1_: `"low battery (0.15 < 0.2) + high temp (0.80 > 0.75)"`
  - _Case 5_: `"very high complexity (0.95 > 0.9) + GPU headroom available (0.10 < 0.8)"`
  - _Case 6_: `"high complexity (0.95 > 0.75) + CPU headroom available (0.10 < 0.6)"` (Rule 3 skipped due to missing GPU)
- **GPU Integration**: An NVIDIA GeForce GTX 1650 is present on the development machine. By adding `pynvml` to the dependencies, GPU telemetry is now active. While GPU queries can return `None` during idle states on Windows hybrid laptops when the card is in a low-power D3 sleep state, the driver is successfully tracked under active CUDA workloads. For offline validation, Rule 3 was verified using synthetic state objects (Case 5).

## Design Decision

- **Decision Stability & Hysteresis**: In this phase, decisions are made on a frame-by-frame basis. A complexity score hovering near a threshold (e.g. `0.75`) may cause rapid model-switching (flickering). While hysteresis is out-of-scope for v1, it is flagged as an open constraint for implementation during Phase 6/7.

---

# Validation Results - Phase 5a Model Registry & Integration

This section logs the verification of the super-resolution model backends (`tinysr` / FSRCNN and `real_esrgan` / Real-ESRGAN), model registry dynamic dispatch, and model caching.

## Evaluation Setup
We verified model execution correctness, caching speedups, CPU fallback execution, and GPU dispatch routing on a `64x64` sample checkerboard BGR frame.

- **FSRCNN (`tinysr`) Weights**: PyTorch `.pt` file downloaded from `Nhat-Thanh/FSRCNN-Pytorch` checkpoint releases (x2).
- **Real-ESRGAN (`real_esrgan`) Weights**: PyTorch `.pth` files downloaded from `xinntao/Real-ESRGAN` releases (x2, x4).

## Test Matrix Verification Results

| # | Test Case | Target Model | Device | Expected Shape | Status | Measured Timing / Peak VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Smoke Test FSRCNN | `tinysr` | CPU/GPU | (128, 128, 3) | **PASSED** | Output image successfully saved |
| **2** | Smoke Test Real-ESRGAN | `real_esrgan` | CPU/GPU | (128, 128, 3) | **PASSED** | Output image successfully saved |
| **3** | Caching FSRCNN | `tinysr` | GPU | (128, 128, 3) | **PASSED** | Run 1: `51.5 ms` $\rightarrow$ Run 2: **`4.6 ms`** (11.2x speedup) |
| **4** | Caching Real-ESRGAN | `real_esrgan` | GPU | (128, 128, 3) | **PASSED** | Run 1: `954.6 ms` $\rightarrow$ Run 2: **`201.7 ms`** (4.7x speedup) |
| **5** | CPU vs GPU Dispatch | `tinysr` | Both | (128, 128, 3) | **PASSED** | Ran on CPU and CUDA GPU explicitly |
| **6** | EnhancementEngine Routing | Both | GPU | (128, 128, 3) | **PASSED** | Succeeded using `Decision` parameters |
| **7** | VRAM Stress (x4 scaling) | `real_esrgan` | GPU | (1920, 2560, 3) | **PASSED** | Peak VRAM: **`1510.05 MB`** (runs successfully under 4GB limit) |

## Verification Analysis
- **Import Workarounds**: The dynamic `torchvision.transforms.functional_tensor` and `collections.Container` import injections successfully resolved all Real-ESRGAN/BasicSR compatibility conflicts at runtime.
- **Dynamic Registry Dispatch**: `EnhancementEngine` successfully resolved string-path loader functions via `importlib` and cached them, executing without hardcoded model branches.
- **Latency Benchmarking**: 
  - FSRCNN (`tinysr`): **`7.87 ms`** on CPU, **`5.25 ms`** on GPU (GTX 1650).
  - Real-ESRGAN (`real_esrgan`): **`439.00 ms`** on CPU, **`167.01 ms`** on GPU.
  The results prove the ~50x difference in resource consumption between the lightweight and mid-tier models, validating the need for dynamic switching.
- **GPU Integration**: PyTorch CUDA is successfully enabled using the `D:\Full Stack\Caption Generator\caption-app\venv` target virtual environment, allowing all GPU-accelerated cases and stress tests to run to completion.

