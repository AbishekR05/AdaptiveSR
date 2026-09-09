# Step 9 — Edge Selection & Resource Allocation Implementation Documentation

## Overview

Step 9 implements the **Edge Resource Evaluation Layer** (`adaptive_sr/adaptation/edge_evaluator.py`). It determines whether available Edge nodes can support a requested Super-Resolution (SR) workload under their current compute and network conditions, producing machine-readable candidate/resource signals (`EdgeResourceSignal`) for Step 10.

This layer evaluates hardware capability, load state, network telemetry, and workload feasibility. It **does NOT** make final adaptive decisions, choose final representations/models, or synthesize arbitrary scalar utility functions (which are reserved for Step 10).

---

## 1. Resource State Schema: Capability vs. Availability

Step 9 strictly separates hardware capability from current resource utilization:

- **Hardware Capability (`hardware_capability`)**:
  - `cpu_cores_total`: Total logical CPU cores reported by the host.
  - `gpu_available`: Boolean flag indicating physical GPU presence.
  - `gpu_device_name`: Model name of the physical GPU (e.g., `NVIDIA RTX 5060`).
  - `gpu_memory_total_bytes`: Total physical VRAM in bytes.
  - `supported_devices`: List of supported runtime execution devices (e.g., `["cpu", "cuda"]`).
  - `supported_models`: List of registered SR model adapters (e.g., `["tinysr", "real_esrgan"]`).

- **Current Resource Availability / Load State (`resource_availability`)**:
  - `cpu_utilization_percent`: System-wide CPU % load.
  - `memory_utilization_percent`: System RAM % load.
  - `gpu_utilization_percent`: GPU core % load (a capable GPU under heavy load is NOT treated as idle).
  - `gpu_memory_free_bytes`: Currently available VRAM in bytes.
  - `active_requests`: In-flight request count.
  - `queue_depth`: Pending queue depth.

---

## 2. Feasibility Evaluation & Safety Thresholds

1. **Hardware & Model Compatibility**: Requested `device` and `model_id` must be explicitly supported by the candidate node.
2. **Configurable Operational Safety Thresholds**:
   - Resource overload thresholds (`max_cpu_utilization_percent` defaulting to 95.0%, `max_gpu_utilization_percent` defaulting to 98.0%) are documented and treated as **configurable engineering/operational safety limits**, NOT universal scientific constants.
   - `required_gpu_memory_bytes` is an explicit workload requirement derived from model runtime operational requirements rather than an invented static model constant.
3. **CUDA Safety & Memory Check**: If `device="cuda"` is requested:
   - If `gpu_available == False`: `resource_feasible = false` (`feasibility_status = "infeasible"`). **NO silent fallback to CPU occurs.**
   - If `gpu_memory_free_bytes < required_gpu_memory_bytes`: `resource_feasible = false`.
   - If `gpu_utilization_percent >= max_gpu_utilization_percent` (98.0%): `resource_feasible = false`.
4. **CPU Utilization Check**: If `cpu_utilization_percent >= max_cpu_utilization_percent` (95.0%): `resource_feasible = false`.
5. **Missing Telemetry Handling**: Missing load metrics remain `null` (`None`) in the signal. If GPU load telemetry is absent for a CUDA request, `feasibility_status` is marked as `"unknown"`.

---

## 3. Network Telemetry Semantics

- Network metrics (`cloud_edge_rtt_ms`, `client_edge_rtt_ms`, `measured_bandwidth_mbps`) are logged in `network_telemetry`.
- **RTT Non-Implication**: `cloud_edge_rtt_ms` is recorded as a separate network telemetry metric and is **never** conflated with complete end-to-end streaming latency or SR processing time.

---

## 4. Machine-Readable Signal Schema (`EdgeResourceSignal`)

```json
{
  "edge_id": "edge_01",
  "cluster_id": "cluster_01",
  "model_id": "tinysr",
  "scale": 2,
  "device": "cpu",
  "base_representation_id": "360p",
  "target_resolution": "1280x720",
  "resource_feasible": true,
  "network_feasible": true,
  "feasibility_status": "feasible",
  "hardware_capability": {
    "cpu_cores_total": 16,
    "gpu_available": false,
    "gpu_device_name": null,
    "gpu_memory_total_bytes": null,
    "supported_devices": ["cpu"],
    "supported_models": ["tinysr", "real_esrgan"]
  },
  "resource_availability": {
    "cpu_utilization_percent": 32.5,
    "memory_utilization_percent": 45.0,
    "gpu_utilization_percent": null,
    "gpu_memory_free_bytes": null,
    "active_requests": 1,
    "queue_depth": 0
  },
  "network_telemetry": {
    "cloud_edge_rtt_ms": 12.5,
    "client_edge_rtt_ms": null,
    "measured_bandwidth_mbps": null
  },
  "sr_telemetry": {
    "sr_processing_time_ms": 42.15,
    "estimated_processing_fps": 23.72
  },
  "measurement_provenance": "step6_edge_telemetry",
  "warnings": [
    "Cloud-to-Edge RTT measured; Cloud RTT does not represent end-to-end streaming latency."
  ]
}
```

---

## 5. Candidate Evaluation & Deterministic Non-Preferential Ordering

`EdgeResourceEvaluator.evaluate_candidates(candidates, ...)` evaluates a list of Edge nodes:
- Feasible and infeasible candidates are each identified explicitly with full attribute signals preserved.
- **Non-Preferential Ordering**: Returned candidates are sorted strictly by `edge_id` alphabetically. This provides deterministic output ordering for repeatable testing/logging without imposing preference ordering (such as CPU utilization ascending) or weighted utility scores prior to Step 10.
- **No Preference Policy or Utility Score**: Step 9 does NOT decide which feasible node is "best". Final multi-objective selection policy belongs exclusively to Step 10.

---

## 6. Verification & Test Results

```bash
pytest tests/test_edge_selection.py tests/test_bitrate_adaptation.py tests/test_fps_adaptation.py tests/test_remote_sr.py tests/test_foundation.py -v
```

### Test Summary (55/55 Passed):
- `test_gpu_capable_feasible_edge`: Passed
- `test_cpu_only_edge_cpu_request`: Passed
- `test_unavailable_requested_gpu`: Passed (Verified NO silent CPU fallback)
- `test_insufficient_gpu_memory`: Passed
- `test_overloaded_edge_node`: Passed
- `test_missing_resource_telemetry`: Passed (Preserved `null` values)
- `test_missing_network_telemetry`: Passed (Preserved `null` values)
- `test_multiple_edge_candidates`: Passed (Verified stable `edge_id` ordering, independent representations, and zero CPU preference bias)
- `test_feasible_vs_infeasible_candidates`: Passed
- `test_rtt_not_treated_as_end_to_end_latency`: Passed
- `test_no_silent_cpu_fallback`: Passed
- `test_configurable_resource_thresholds`: Passed (Verified custom engineering overload thresholds)
- `test_real_local_step6_integration`: Passed (Live end-to-end Edge HTTP `/health` & chunk telemetry ingestion)
- `tests/test_bitrate_adaptation.py` (12 tests): Passed
- `tests/test_fps_adaptation.py` (12 tests): Passed
- `tests/test_remote_sr.py` (5 tests): Passed
- `tests/test_foundation.py` (13 tests): Passed

---

## 7. Frozen Boundaries & Limitations

- **Steps 0–8 Frozen**: Steps 0 through 8 remain 100% frozen. No contracts or code in Steps 0–8 were modified.
- **No Final Adaptation Decision**: Step 9 evaluates candidate node feasibility and preserves measurable state; it does not select the final representation, model, or edge node.
- **Out of Scope**: Final Adaptive Decision Engine and global multi-objective optimization are deferred to Step 10.
