# Step 7 — FPS Adaptation & Feasibility Implementation Documentation

## Overview

Step 7 implements the **FPS Adaptation & Feasibility Layer** (`adaptive_sr/adaptation/fps_adapter.py`). It compares video playback frame rate (FPS) requirements against measured Super-Resolution (SR) processing latency—derived from Step 5 benchmark datasets or Step 6 Edge runtime telemetry—and generates a machine-readable adaptation signal (`FPSAdaptationSignal`).

This layer is decision-oriented but **does NOT** implement a full decision engine, bitrate adaptation, edge node selection, or resource allocation (which are reserved for Steps 8+).

---

## 1. Core Mathematical Formulas

$$\text{frame\_budget\_ms} = \frac{1000.0}{\text{source\_fps}}$$

$$\text{estimated\_processing\_fps} = \frac{1000.0}{\text{measured\_latency\_ms}}$$

$$\text{real\_time\_ratio} = \frac{\text{frame\_budget\_ms}}{\text{measured\_latency\_ms}}$$

$$\text{realtime\_feasible} = \begin{cases} \text{True} & \text{if } \text{measured\_latency\_ms} \le \text{frame\_budget\_ms} \iff \text{real\_time\_ratio} \ge 1.0 \\ \text{False} & \text{otherwise} \end{cases}$$

---

## 2. Adaptation Classification Tiers

Deterministic classification based on `real_time_ratio`:

| Adaptation Tier | Condition | Headroom | Interpretation / Action |
|---|---|---|---|
| `realtime` | $\text{real\_time\_ratio} \ge 1.0$ | $\ge 0\%$ | Processing completes within frame budget. Safe for real-time playback. |
| `near_realtime` | $0.75 \le \text{real\_time\_ratio} < 1.0$ | $-25\%$ to $0\%$ | Within $25\%$ of budget; recoverable via modest client buffering or light frame-skipping. |
| `below_realtime` | $0.50 \le \text{real\_time\_ratio} < 0.75$ | $-50\%$ to $-25\%$ | Latency is $1.33\times$ to $2.0\times$ frame budget; playback stalls expected without scaling down. |
| `severely_below_realtime` | $\text{real\_time\_ratio} < 0.50$ | $< -50\%$ | Latency exceeds $2.0\times$ frame budget; unfeasible for real-time execution. |
| `invalid` | Non-positive or `NaN` | N/A | Non-positive or undefined FPS / latency inputs. |

---

## 3. Machine-Readable Signal Schema (`FPSAdaptationSignal`)

```json
{
  "source_fps": 30.0,
  "frame_budget_ms": 33.3333,
  "measured_latency_ms": 20.0,
  "estimated_processing_fps": 50.0,
  "real_time_ratio": 1.6667,
  "realtime_feasible": true,
  "adaptation_tier": "realtime",
  "model_id": "tinysr",
  "scale": 2,
  "device": "cpu",
  "base_representation_id": "360p",
  "measurement_provenance": "step6_edge_telemetry",
  "decision_eligible": true,
  "end_to_end_streaming_feasible": true,
  "warnings": []
}
```

---

## 4. Measurement Provenance & Integration

- **Step 5 Benchmark Data**: `FPSAdapter.evaluate_from_step5_record(record)` ingests offline benchmark records (`fps_feasibility.json` / `BenchmarkResult`) with provenance `"step5_benchmark"`.
- **Step 6 Edge Telemetry**: `FPSAdapter.evaluate_from_step6_telemetry(telemetry)` ingests live Edge response headers (`X-SR-Processing-Time`, `X-Edge-Cloud-RTT`) or JSON telemetry with provenance `"step6_edge_telemetry"`.
- **Direct Parameter Evaluation**: `FPSAdapter.evaluate(source_fps, measured_latency_ms, ...)` provides standalone evaluation for unit tests or upstream planners.

---

## 5. Verification & Test Results

```bash
pytest tests/test_fps_adaptation.py tests/test_remote_sr.py tests/test_foundation.py -v
```

### Test Summary (28/28 Passed):
- `test_30fps_latency_below_budget`: Passed
- `test_30fps_latency_above_budget`: Passed
- `test_exact_budget_boundary`: Passed
- `test_invalid_zero_fps`: Passed
- `test_invalid_zero_latency`: Passed
- `test_missing_measurements`: Passed
- `test_multiple_models_scales_devices`: Passed
- `test_decision_eligibility_distinction`: Passed
- `test_classification_thresholds`: Passed
- `test_real_local_step6_integration`: Passed (Live end-to-end Edge HTTP TestClient execution with `TinySR` forward pass)
- `tests/test_remote_sr.py` (5 tests): Passed
- `tests/test_foundation.py` (13 tests): Passed

---

## 6. Limitations & Frozen Boundaries

- **Steps 0–6 Frozen**: Steps 0 through 6 remain 100% frozen. No contracts or code in Steps 0–6 were modified.
- **Inference Pipeline**: Step 7 consumes existing timing data and does not re-run or duplicate SR model forward passes.
- **Out of Scope**: Model selection logic, bitrate adaptation, edge node selection, resource allocation, and final decision engine are deferred to Steps 8+.
