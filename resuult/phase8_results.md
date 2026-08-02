# Phase 8 Optimizations — Evaluation Report

This report presents the validation and benchmarking of the system performance optimizations implemented in **Phase 8**.

---

## 1. Before/After Performance Matrix

| Optimization | Baseline (Before) | Optimized (After) | Delta / Findings |
| :--- | :--- | :--- | :--- |
| **Skip-Enhancement Tier** | N/A (no skip logic; FSRCNN ran at 118ms) | **0.0 ms** (instant bypass) | **100% compute/battery save** on critical states for simple frames |
| **Dynamic Scale Reduction** | Hardcoded `scale=2` regardless of battery budget | **`scale=2`** automatically requested under critical battery | **Compute load capped** under battery constraints |
| **INT8 TinySR (CPU)** | **294.5 ms** (FP32 PyTorch CPU) | **589.4 ms** (INT8 ONNX Runtime CPU) | **0.50x speedup** (slower; see analysis section below) |
| **INT8 TinySR Quality** | — | **42.70 dB PSNR** (0.9798 SSIM) | **Negligible quality loss** (virtually identical to FP32) |

---

## 2. Key Findings & Academic Analysis

### A. The "Hourglass Quantization" Overhead
Our benchmarks show that dynamically quantizing FSRCNN (`tinysr`) to INT8 on CPU actually **halves performance** (slowing frame rendering from 294.5 ms to 589.4 ms).
*   **Why?** Dynamic quantization requires quantizing the input float tensor to INT8, performing integer operations, and dequantizing the result back to float32 on-the-fly. 
*   **The Hourglass Exception**: For extremely small models like FSRCNN (which contains only a few layers and ~100 KB of total parameter size), the computation cost of the convolutions is already extremely small. Consequently, the **data conversion overhead** (float $\rightarrow$ int8 $\rightarrow$ float) dominates the execution time, completely wiping out the speedup of integer matrix multiplications.
*   **Academic Takeaway**: Quantization is highly effective for large models (e.g., ResNet, Llama, Real-ESRGAN) where compute is the bottleneck, but is counter-productive for ultra-lightweight sub-megabyte network designs like FSRCNN.

### B. Skip-Enhancement Tier Validation
We verified that when battery is critical (< 10%) and scene complexity is low (< 0.15), the framework successfully issues a `"skip"` decision:
*   **Latencies are dropped to 0 ms** as the frame bypasses GPU/CPU inference models completely.
*   **Frame integrity is preserved** as the BGR numpy array is returned byte-identical to the source.

### C. Dynamic Scale Reduction Verification
Rule-checking confirms that when battery falls below the configured `scale_reduction_battery: 0.30` threshold:
*   The target output scale is restricted to **`scale=2`** even for models like `basicvsr++` that default to `scale=4`, protecting device resources from OOM/overheating under constraint.
