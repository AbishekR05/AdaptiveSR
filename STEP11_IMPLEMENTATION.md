# STEP 11 — END-TO-END ADAPTIVESR STREAMING RUNTIME

## 1. System Architecture

Step 11 builds the end-to-end closed-loop AdaptiveSR streaming runtime operating chunk-by-chunk:
`observe → adaptation signals (Steps 7–9) → fuzzy decision (Step 10) → execute selected configuration (Step 6) → update buffer/state → next chunk`.

```
                  +----------------------------------------------+
                  |         AdaptiveSRRuntime Orchestrator       |
                  +----------------------------------------------+
                                         |
               +-------------------------+-------------------------+
               |                         |                         |
               v                         v                         v
     +--------------------+    +--------------------+    +--------------------+
     | Step 7 FPS Adapter |    | Step 8 Bitrate Ad. |    | Step 9 Edge Eval.  |
     +--------------------+    +--------------------+    +--------------------+
               |                         |                         |
               +-------------------------+-------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     | Step 10 Fuzzy Adaptive Decision Engine |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |  Step 6 Remote SR Execution (Edge)    |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     | Client Playback Buffer Math & State   |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     | Machine-Readable Per-Chunk Telemetry  |
                     +---------------------------------------+
```

---

## 2. Hardened Runtime Semantics

### 2.1 Execution Failure Buffer Semantics
- For `delivery_mode = "execution_failed"`:
  - The failed chunk is **NOT** delivered (`chunk_delivered = false`, `delivered_chunk_duration = 0.0`).
  - `chunk_duration` is **NOT** added to the playback buffer.
  - Buffer becomes: `buffer_after = max(0.0, buffer_before - client_elapsed_seconds)`.
  - Stall duration: `stall_duration = max(0.0, client_elapsed_seconds - buffer_before)`.
  - If `client_elapsed_seconds > buffer_before`, stall count increments and `total_stall_duration` increments by `stall_duration`.
- For successful SR or native delivery:
  - `chunk_delivered = true` and `delivered_chunk_duration = chunk_duration`.
  - Replenishment formula: `buffer_after = max(0.0, buffer_before - client_elapsed_seconds) + chunk_duration`.

### 2.2 Final Decision Eligibility
- Preserves the selected candidate's `decision_eligible` explicitly in the final decision telemetry record when an SR candidate is selected (`decision.decision_eligible: bool`).
- When no SR candidate is selected (`no_suitable_candidate` / `no_feasible_candidates`), `decision.decision_eligible` is `null` (not `false`), because no evaluated SR candidate exists to which the candidate eligibility value applies.

### 2.3 Native Fallback Delivery Path
- Native execution is resolved independently of Step 10 SR edge selection via `NativeDeliveryRegistry`.
- Native fallback delivery maps the fallback representation to a native content delivery origin endpoint (`delivery_origin: "native_origin"`, `delivery_endpoint: ...`).
- Native execution does not invent an SR `edge_id` or require model ID, scale factor, or hardware device (`model_id: null`, `scale: null`, `device: null`).

### 2.4 Quality Evidence Coverage & Identity Bridge
- Live runtime quality is available **ONLY** when matching precomputed Step 5.6 / Step 8 evidence exists in `QualityEvidenceStore`.
- Provides an explicit identity bridge mapping runtime `(video_id, chunk_id)` to Step 5.6 benchmark dataset `input_id`.
- Evidence is matched by explicit identity/provenance: `(input_id, representation_id, model_id, scale, device)`.
- If no deterministic evidence mapping exists $\implies$ `quality_evaluable = false`, metrics = `null`, candidate remains non-selectable.
- Live streaming does NOT fabricate PSNR/SSIM/VMAF or imply reference high-resolution video data exists during live playback.

### 2.5 Clock Semantics
- `client_elapsed_seconds` is measured strictly using a monotonic clock (`time.monotonic()`).
- `client_elapsed_seconds` is the **ONLY** quantity used for client buffer depletion and stall calculation.
- Wall-clock timestamps are explicitly named `request_start_unix_timestamp` and `completion_unix_timestamp`.
- Server-reported diagnostic fields (`download_transfer_time`, `sr_processing_time`) are retained strictly for diagnostic breakdown telemetry and are **NOT** additive with `client_elapsed_seconds`.

### 2.6 Requested vs. Executed State
- Tracks both `requested_configuration` and `executed_configuration`.
- `delivery_mode`: `"sr"` | `"native"` | `"execution_failed"`.
- A requested SR configuration is **NOT** recorded as executed if Step 6 execution fails (`executed_configuration = null`).

### 2.7 Configuration Switch Tracking
- Counts changes between consecutive **EXECUTED** delivery states:
  - Native state: `("native", "native_origin", representation_id)`
  - SR state: `("sr", edge_id, representation_id, model_id, scale, device)`
- Explicitly tracks `SR → native` and `native → SR` transitions. Failed requests do not alter executed state.

### 2.8 Edge Routing & Native Delivery
- `EdgeRegistry` maps `edge_id` to its HTTP SR endpoint URL (`edge_id -> endpoint_url`).
- `NativeDeliveryRegistry` maps fallback representations to native content origin endpoints (`representation_id -> endpoint_url`).

### 2.9 Dynamic Test Conditions
- `DynamicConditionProfile` is test-only and active only when `enabled = true`.
- Telemetry explicitly records `is_injected_test_condition` and `injected_fields: List[str]`. Normal runtime never injects conditions.

### 2.10 Error Telemetry
- Execution failures produce structured `ErrorTelemetry`: `error_code`, `error_message`, `traceback_available`.

---

## 3. Telemetry Schema Specification

### 3.1 Successful Remote SR Execution Telemetry Example
```json
{
  "identity": {
    "video_id": "sample",
    "chunk_id": "0000",
    "request_id": "req_12345",
    "edge_id": "edge_01",
    "cluster_id": "cluster_01"
  },
  "delivery_mode": "sr",
  "requested_configuration": {
    "edge_id": "edge_01",
    "representation_id": "360p",
    "target_resolution": "720p",
    "model_id": "tinysr",
    "scale": 2,
    "device": "cpu"
  },
  "executed_configuration": {
    "edge_id": "edge_01",
    "representation_id": "360p",
    "target_resolution": "720p",
    "model_id": "tinysr",
    "scale": 2,
    "device": "cpu"
  },
  "decision": {
    "decision": "selected",
    "fuzzy_suitability": 65.42,
    "suitability_tier": "high",
    "rejection_reason": null,
    "fallback_reason": null,
    "min_suitability_threshold": 35.0,
    "decision_eligible": true
  },
  "timing": {
    "request_start_unix_timestamp": 1788920947.17,
    "completion_unix_timestamp": 1788920947.57,
    "client_elapsed_seconds": 0.40,
    "download_transfer_time": 0.25,
    "sr_processing_time": 0.09
  },
  "network": {
    "measured_bandwidth_mbps": 10.0,
    "rtt_seconds": 0.03,
    "bytes_received": 150000
  },
  "resource": {
    "cpu_utilization_pct": 25.0,
    "gpu_utilization_pct": 30.0,
    "gpu_memory_used_mb": 500.0
  },
  "buffer": {
    "buffer_before": 1.0,
    "buffer_after": 2.6,
    "stall_count": 0,
    "stall_duration": 0.0,
    "chunk_delivered": true,
    "delivered_chunk_duration": 2.0
  },
  "provenance": {
    "fps_signal_provenance": {"signal_count": 4, "decision_eligible": true},
    "bitrate_signal_provenance": {"signal_count": 4, "decision_eligible": true, "quality_evaluable": true},
    "resource_signal_provenance": {"signal_count": 4},
    "decision_signal_provenance": {
      "fps": "step7_fps_adaptation",
      "bitrate": "step8_bitrate_adaptation",
      "edge": "step9_edge_resource"
    }
  },
  "is_injected_test_condition": false,
  "injected_fields": [],
  "error": null
}
```
*Note: In `timing`, `client_elapsed_seconds` (0.40s) != `download_transfer_time` (0.25s) + `sr_processing_time` (0.09s), explicitly illustrating that diagnostic timing fields are NOT additive.*

### 3.2 Native Fallback Execution Telemetry Example
```json
{
  "delivery_mode": "native",
  "requested_configuration": null,
  "executed_configuration": {
    "delivery_origin": "native_origin",
    "delivery_endpoint": "http://localhost:8000/cloud/videos/sample/360p",
    "representation_id": "360p",
    "target_resolution": "360p",
    "model_id": null,
    "scale": null,
    "device": null
  },
  "decision": {
    "decision": "no_suitable_candidate",
    "fuzzy_suitability": null,
    "suitability_tier": null,
    "rejection_reason": "No candidate met feasibility or suitability threshold",
    "fallback_reason": "no_suitable_candidate",
    "min_suitability_threshold": 35.0,
    "decision_eligible": null
  },
  "buffer": {
    "buffer_before": 1.0,
    "buffer_after": 2.8,
    "stall_count": 0,
    "stall_duration": 0.0,
    "chunk_delivered": true,
    "delivered_chunk_duration": 2.0
  }
}
```

### 3.3 Execution Failure Telemetry Example
```json
{
  "delivery_mode": "execution_failed",
  "requested_configuration": {
    "edge_id": "edge_01",
    "representation_id": "360p",
    "target_resolution": "720p",
    "model_id": "tinysr",
    "scale": 2,
    "device": "cuda"
  },
  "executed_configuration": null,
  "decision": {
    "decision": "execution_failed",
    "fuzzy_suitability": 80.0,
    "suitability_tier": "high",
    "rejection_reason": "Edge GPU out of memory",
    "fallback_reason": null,
    "min_suitability_threshold": 35.0,
    "decision_eligible": true
  },
  "buffer": {
    "buffer_before": 5.0,
    "buffer_after": 4.95,
    "stall_count": 0,
    "stall_duration": 0.0,
    "chunk_delivered": false,
    "delivered_chunk_duration": 0.0
  },
  "error": {
    "error_code": "STEP6_EXECUTION_FAILED",
    "error_message": "Edge GPU out of memory",
    "traceback_available": true
  }
}
```

---

## 4. Test Execution & Terminal Results

### Actual Terminal Output Summary
```
====================== 108 passed, 1 warning in 6.24s ======================
```

- **Step 11 Integration Suite (`tests/test_end_to_end_runtime.py`)**: 31 / 31 Passed (Tests A–N + 17 Hardening Semantic Patch Tests).
- **Frozen Steps 0–10 Core Regression Suite**: 77 / 77 Passed.
- **Total Regression Suite**: 108 / 108 Passed (0 Failed, 0 Skipped).

---

## 5. Known Limitations

- **Browser/VLC Player Interaction**: Step 11 uses a mathematical client buffer simulation model based on monotonic elapsed time.
- **Hysteresis / Stability Control**: Step 11 measures raw decision behavior. Hysteresis and stability smoothing are deferred to Step 12.
- **Ablation & Final Baselines**: Experimental campaigns, baseline comparisons (e.g. heuristic ABR vs. Fuzzy AdaptiveSR), and QoE trade-off analysis belong to Step 12.
s belong to Step 12.
