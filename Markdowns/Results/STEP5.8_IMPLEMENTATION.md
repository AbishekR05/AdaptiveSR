# STEP 5.8 — MACHINE-READABLE BENCHMARK DATASET
## IMPLEMENTATION DOCUMENTATION

### 1. Objective & Scope
Step 5.8 establishes a unified, canonical, machine-readable dataset representation layer. It consolidates empirical evidence from:
- **Step 5.5**: Benchmark inference latencies, throughput, resource usage, and multi-session decision eligibility.
- **Step 5.6**: Visual quality evaluation metrics (Y-channel PSNR, SSIM, VMAF / `vmaf_unavailable`).
- **Step 5.7**: Real-time FPS feasibility analysis ($T_{\text{budget}}$, estimated FPS, real-time ratio, $P95$ exploratory feasibility).

> [!IMPORTANT]
> **Data Consolidation Layer Only:**
> Step 5.8 is strictly a data representation and consolidation layer. It does NOT implement adaptive model selection, bitrate adaptation, FPS adaptation, edge scheduling, online learning, or ML decision engine policies.
> Raw outputs from Steps 5.5, 5.6, and 5.7 remain completely preserved and un-overwritten.

---

### 2. Canonical Record Key & Join Mechanics
Records are joined using authoritative project identifiers:
$$\text{key} = \text{model\_id} + \text{"\_x"} + \text{scale} + \text{"\_"} + \text{device} + \text{"\_"} + \text{input\_id}$$
Example: `tinysr_x2_cpu_synthetic_lowmotion_30fps`

No new or incompatible join semantics are introduced.

---

### 3. Missing Data Policy
- If a measurement is absent from Step 5.5, 5.6, or 5.7 evidence, its value remains explicitly `null` (`None` in Python / `N/A` in CSV).
- **NEVER fabricate, mock, or estimate missing empirical data values.**
- Each section includes a `data_present` boolean flag (`step55_present`, `step56_present`, `step57_present`) for transparent data completeness tracking.

---

### 4. Canonical Machine-Readable Output Schema

#### Primary JSON Dataset (`data/benchmarks/sr/results/unified_benchmark_dataset.json`)
```json
[
    {
        "record_id": "tinysr_x2_cpu_synthetic_lowmotion_30fps",
        "model_id": "tinysr",
        "scale": 2,
        "device": "cpu",
        "input_id": "synthetic_lowmotion_30fps",
        "benchmark_inference": {
            "latency_ms": 10.0,
            "p95_latency_ms": 12.0,
            "throughput_fps": 100.0,
            "cpu_ids": [1],
            "num_threads": 2,
            "gpu_device_id": null,
            "resource_summary": {},
            "decision_eligible": false,
            "eligibility_reason": "insufficient_sessions_count",
            "session_count": 1,
            "data_present": true
        },
        "quality_evaluation": {
            "psnr_y": 32.5,
            "ssim_y": 0.91,
            "vmaf_mean": null,
            "vmaf_unavailable": true,
            "data_present": true
        },
        "realtime_feasibility": {
            "source_fps": 30.0,
            "frame_budget_ms": 33.333333333333336,
            "estimated_processing_fps": 100.0,
            "real_time_ratio": 3.3333333333333335,
            "real_time_feasible": true,
            "real_time_feasible_p95_exploratory": true,
            "p95_confidence": "exploratory",
            "caveats": ["Ineligibility reason: insufficient_sessions_count"],
            "data_present": true
        },
        "provenance": {
            "step55_source": "data/benchmarks/sr/results/step55_results.json",
            "step56_source": "data/benchmarks/sr/results/quality_clips.json",
            "step57_source": "data/benchmarks/sr/results/fps_feasibility.json",
            "dataset_version": "1.0",
            "generated_timestamp": "2026-09-08T22:17:00Z"
        }
    }
]
```

#### Flattened CSV Dataset (`data/benchmarks/sr/results/unified_benchmark_dataset.csv`)
Provides a flat tabular structure for easy data analysis in pandas, Excel, R, or SQL pipelines.

---

### 5. Provenance & Tracing Policy
Every consolidated record maintains a `provenance` metadata block detailing:
- `step55_source`: Source Step 5.5 file path
- `step56_source`: Source Step 5.6 file path
- `step57_source`: Source Step 5.7 file path
- `dataset_version`: `1.0`
- `generated_timestamp`: ISO 8601 UTC timestamp of dataset consolidation

---

### 6. Verification Results
Run automated test suite:
```bash
python -m pytest tests/test_benchmark_dataset.py -v
python -m pytest tests/ -v
```
All Step 5.8 unit and integration tests pass, and the full regression suite remains passing without breaking changes.
