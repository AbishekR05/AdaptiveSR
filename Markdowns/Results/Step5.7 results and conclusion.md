# Step 5.7 — FPS / Real-Time Feasibility Analysis: Results & Conclusions

**Freeze Status:** PIPELINE VALIDATED. Real-time feasibility analysis is pipeline-validated over the available single-sample baseline configuration. Broader benchmark coverage is deferred pending further Step 5.5 multi-session execution runs.

> [!IMPORTANT]
> **SR Inference-Only Feasibility Disclosure:**
> These results measure SR model inference latency only. Decoding, preprocessing, encoding, and network transfer are NOT included. End-to-end streaming real-time feasibility cannot be established from Step 5.5/5.7 data alone.

## 1. Executive Summary

**Fastest Real-Time Eligible Configuration:** None found.

> **Step 5.5 benchmark coverage note:** Step 5.5 benchmark data currently contains only 1 record (tinysr/x2/cpu); broader coverage requires additional Step 5.5 runs, out of scope for Step 5.7.

## 2. Quantitative Benchmark Comparisons

| Model | Scale | Device | Latency (median, ms) | p95 Latency (ms) | p95 vs Budget (non-decisive) | Est. FPS | Source FPS | Budget (ms) | Ratio | Real-Time (measured) | Decision-Eligible | Session Count | p95 Conf | Interpretation | Caveats / Eligibility Warnings |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| tinysr | x2 | cpu | 644.47 | 660.97 | NO | 1.55 | 30.0 | 33.33 | 0.05 | NO | NO | 1 | exploratory | per_frame | Ineligibility reason: insufficient_sessions_count |

*Source FPS Provenance Note: The source_fps value was joined from the Step 5.1 corpus manifest (data/benchmarks/sr/manifests/benchmark_manifest.json → source_fps) using the Step 5.5 input identifier (input_id: synthetic_lowmotion_30fps) as a join key.*

## 3. Real-Time Feasibility Classifications

### 3.1 Measured + Decision-Eligible Configurations (Safe for Production)
- None.

### 3.2 Measured Feasible Only (Not Decision-Eligible due to high variance)
- None.

### 3.3 Failing Configurations (Unfeasible for Real-Time)
- `tinysr` x2 on `cpu` (Latency: **644.47 ms**)

## 4. Measured Scale Configuration

No scale-degradation trend can be established from the currently available benchmark data because only the x2 configuration has been measured.

### tinysr on cpu
- **Scale x2**: Latency **644.47 ms** (1.55 FPS), UNFEASIBLE.

## 5. CPU vs. GPU Performance Comparison

| Model | Scale | CPU Threads | CPU Latency (ms) | GPU Latency (ms) | Ratio (CPU/GPU) | Feasibility Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A |

*Note: CPU vs GPU comparison is unavailable for this configuration as no matching GPU run is present in the current Step 5.5 benchmark results dataset (gpu_benchmark_gap: true).*

## 6. Output Files

| File | Resolution / Mode | Description |
|---|---|---|
| `data/benchmarks/sr/results/fps_feasibility.json` | JSON Schema | Structured, per-configuration analyzed fps feasibility record |

## 7. Test Suite Results

Thirteen unit and integration tests were executed under `tests/test_fps_feasibility.py`:

```
tests/test_fps_feasibility.py::test_frame_budget_calculation                         PASSED
tests/test_fps_feasibility.py::test_estimated_fps_calculation                       PASSED
tests/test_fps_feasibility.py::test_real_time_ratio                                 PASSED
tests/test_fps_feasibility.py::test_real_time_classification_median_based           PASSED
tests/test_fps_feasibility.py::test_p95_never_drives_classification                 PASSED
tests/test_fps_feasibility.py::test_scale_comparison_omits_unsupported_combinations PASSED
tests/test_fps_feasibility.py::test_cpu_gpu_rows_separated                          PASSED
tests/test_fps_feasibility.py::test_missing_source_fps_sets_gap_flag                 PASSED
tests/test_fps_feasibility.py::test_invalid_latency_raises                          PASSED
tests/test_fps_feasibility.py::test_eligibility_warnings_preserved                  PASSED
tests/test_fps_feasibility.py::test_ineligible_session_still_reports_measured_feasibility PASSED
tests/test_fps_feasibility.py::test_output_schema_matches_spec                      PASSED
tests/test_fps_feasibility.py::test_integration_real_step5_5_fixture                PASSED
====================================== 13 passed ======================================
```

