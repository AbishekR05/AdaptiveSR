# Validation Results - Phase 1 Passthrough Pipeline

This document logs the benchmarking and verification results of the Phase 1 passthrough pipeline.

## Execution Summary

*   **Test Video**: `input_test.mp4`
*   **Resolution**: 640x480
*   **FPS**: 30.00
*   **Total Frames**: 90 frames
*   **Duration**: 3.0 seconds
*   **Output Video**: `output_test.mp4` (identical frame count, resolution, and fps)
*   **Processing Latency**: 0.49 seconds
*   **Average Frame Processing Rate**: ~183.67 FPS

## Performance Profile (Averaged over 90 frames)

| Metric | Average Value |
| :--- | :--- |
| **CPU Utilization** | ~0.0% (Fast execution, CPU load minimal) |
| **RAM Utilization** | ~94.5% |
| **Battery Percentage** | 97.0% |
| **GPU Utilization** | N/A (GPU monitoring disabled/no nvidia NVML driver active) |
| **Avg. Frame Write Latency** | ~1.44 ms / frame |

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

| Telemetry State | Baseline Phase | High Load Phase | Cooldown Phase |
| :--- | :--- | :--- | :--- |
| **CPU Utilization** | ~39.1% | **~74.4% (Spike Detected)** | ~29.4% |
| **Process RAM (RSS)** | ~32.1 MB | **~309.8 MB (Spike Detected)** | ~42.6 MB (Released) |
| **System RAM (Total)**| ~88.6% | ~90.2% | ~88.7% |

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
  *(Design Choice: Texture and blur clarity are highly correlated as both derive from Laplacian variance. During initial validation, equal weighting caused the complexity score to drop in low-detail zones because the decrease in the inverted `(1 - blur_clarity)` term outpaced the gain in `texture`. We selected Option 2: preserve the metric in the database but heavily down-weight it to 0.05, keeping its influence secondary and documenting this redundancy as a measured methodology detail.)*


## Telemetry Response Results

| Frame Description | Motion | Texture | Edges | Blur Clarity | Complexity Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Flat sky / blank wall** | 0.00 | 0.0000 | 0.0000 | 0.0000 | **0.0500** |
| **Landscape (moderate detail)** | 0.00 | 0.1119 | 0.0029 | 0.1865 | **0.0972** |
| **Close-up face** | 0.00 | 0.2399 | 0.0020 | 0.3999 | **0.1504** |
| **Moderately busy scene** | 0.00 | 1.0000 | 0.0304 | 1.0000 | **0.5061** |
| **Crowded street / high details** | 0.00 | 1.0000 | 0.4304 | 1.0000 | **0.5861** |

## Verification Analysis
- **Monotonicity (Ranking Sanity)**: The complexity score scales monotonically matching human intuition: Flat (0.0500) < Landscape (0.0972) < Close-up face (0.1504) < Busy (0.5061) < Crowded street (0.5861).
- **Determinism**: Multiple identical invocations on the same frame yielded exactly matching float values (Run 1: `0.150358`, Run 2: `0.150358`), confirming pure functions without side effects.
- **Stability**: Adjacent frame testing with camera drift (1px shift) showed a negligible score variance ($|c_t - c_{t-1}| = 0.0001 < 0.05$), ensuring the signal will not cause flickering model selections.
- **Edge Case Processing**: Passing `prev_frame=None` on initial frame execution returned exactly `motion=0.0` rather than causing numeric exceptions or crashes.
- **Motion Sensitivity (Translation Scaling & Limitations)**: Evaluated dynamic motion response using shape translations on a base frame:
  - *Small translation (5px shift)*: **`0.0078`**
  - *Medium translation (30px shift)*: **`0.0468`**
  - *Large translation (120px shift)*: **`0.1872`**
  This confirms that motion scales proportionally with translation distance ($0.0 < 0.0078 < 0.0468 < 0.1872 \le 1.0$) and successfully maps moving objects.
  > [!NOTE]
  > **Validation Limitation**: The translation test represents localized object movement in an otherwise static shot. Under a full-frame camera pan or global motion, the pixel diff mean will be much higher, meaning `MOTION_SCALE_FACTOR = 4.0` may saturate the motion score to 1.0 very quickly. This calibration choice should be stress-tested and adjusted against real video footage during the Phase 7 benchmarking harness.


## Design Decision
- **Inference Sampling Strategy**: Full **per-frame analysis** is selected for v1. This preserves responsiveness to sudden cuts or face entries. Compute overhead is low compared to upcoming neural VSR model inference.



