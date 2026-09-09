# Step 8 — Bitrate / Quality Adaptation Implementation Documentation

## Overview

Step 8 implements the **Bitrate & Quality Adaptation Layer** (`adaptive_sr/adaptation/bitrate_adapter.py`). It measures the bandwidth-vs-quality tradeoff between reference video representations (e.g. native 720p) and candidate lower-bitrate representations enhanced with Super-Resolution (e.g. 360p + $2\times$ SR $\rightarrow$ 720p output), generating a machine-readable `BitrateAdaptationSignal`.

This layer evaluates bandwidth savings and ingests Step 5.6 visual quality metrics for later decision-making (Steps 9/10). It **does NOT** make final representation, model, or edge node selections.

---

## 1. Core Mathematical Formulas

$$\text{bandwidth\_saving\_percent} = \frac{\text{reference\_bitrate\_bps} - \text{candidate\_bitrate\_bps}}{\text{reference\_bitrate\_bps}} \times 100.0$$

- If `reference_bitrate_bps <= 0` or missing: `bitrate_saving_percent` returns `null` (`None`) with an explicit warning.
- If `candidate_bitrate_bps > reference_bitrate_bps`: `bitrate_saving_percent` is negative, indicating a bandwidth penalty.

---

## 2. Machine-Readable Signal Schema (`BitrateAdaptationSignal`)

```json
{
  "reference_representation_id": "720p",
  "candidate_representation_id": "360p",
  "reference_bitrate_bps": 3000000.0,
  "candidate_bitrate_bps": 1000000.0,
  "bitrate_saving_percent": 66.6667,
  "base_resolution": "640x360",
  "target_resolution": "1280x720",
  "model_id": "tinysr",
  "scale": 2,
  "device": "cpu",
  "psnr_db": 33.2,
  "ssim": 0.92,
  "vmaf": null,
  "quality_evaluable": true,
  "quality_provenance": "model_inference",
  "measurement_provenance": "step5.6_quality_eval",
  "decision_eligible": true,
  "quality_equivalent_to_native": null,
  "quality_equivalence_status": "not_evaluated",
  "warnings": [
    "Quality equivalence to native representation not established; lower bitrate + SR does not automatically guarantee native quality parity.",
    "VMAF metric unavailable or not measured."
  ]
}
```

---

## 3. Native vs. SR Quality Parity Policy & Audit Rules

1. **Quality Evaluability vs. Decision Eligibility**:
   - `quality_evaluable = true` indicates that valid quality measurements (PSNR, SSIM, or VMAF) exist.
   - `decision_eligible` is NOT inferred from `quality_evaluable`; it strictly preserves Step 5's eligibility rules ($\text{session\_count} \ge 3$, $\text{CV} \le 15\%$). A candidate can be `quality_evaluable = true` while `decision_eligible = false`.
2. **Quality Equivalence Semantics**:
   - Lowering base bitrate and applying Super-Resolution does **NOT** automatically guarantee quality equivalent to a native higher-resolution representation.
   - Unless explicit comparative native quality evidence is provided, `quality_equivalent_to_native` returns `null` (`None`) with `quality_equivalence_status = "not_evaluated"`.
3. **Bitrate Semantics**:
   - Bandwidth savings are calculated strictly from declared/measured bitrates (`reference_bitrate_bps` vs `candidate_bitrate_bps`), without assuming resolution implies bitrate.

---

## 4. Relationship to Step 7 FPS Adaptation

| Feature | Step 7 (FPS Adaptation) | Step 8 (Bitrate / Quality Adaptation) |
|---|---|---|
| **Primary Metric** | Real-time feasibility ($\text{latency} \le \text{frame budget}$) | Bandwidth savings & visual quality ($\text{PSNR}$, $\text{SSIM}$, $\text{VMAF}$) |
| **Input Source** | Step 5 benchmarks / Step 6 Edge SR telemetry | Video manifests / Step 5.6 quality evaluation outputs |
| **Output Signal** | `FPSAdaptationSignal` | `BitrateAdaptationSignal` |
| **Adaptation Tier** | `realtime`, `near_realtime`, `below_realtime`, `severely_below_realtime` | Percentage saving + Quality evaluability flag |
| **Role in Pipeline** | Evaluates computational feasibility | Evaluates bandwidth efficiency & visual fidelity |

---

## 5. Verification & Test Results

```bash
pytest tests/test_bitrate_adaptation.py tests/test_fps_adaptation.py tests/test_remote_sr.py tests/test_foundation.py -v
```

### Test Summary (42/42 Passed):
- `test_bitrate_saving_calculation`: Passed
- `test_equal_bitrate`: Passed
- `test_candidate_bitrate_greater_than_reference`: Passed
- `test_zero_or_missing_bitrate`: Passed
- `test_available_quality_metrics`: Passed
- `test_unavailable_vmaf_metric`: Passed
- `test_model_inference_vs_bicubic_provenance`: Passed
- `test_base_target_representation_identity`: Passed
- `test_multiple_representations`: Passed
- `test_missing_quality_measurements`: Passed
- `test_decision_eligibility_distinction`: Passed
- `test_real_local_step6_integration`: Passed
- `tests/test_fps_adaptation.py` (12 tests): Passed
- `tests/test_remote_sr.py` (5 tests): Passed
- `tests/test_foundation.py` (13 tests): Passed

---

## 6. Frozen Boundaries & Limitations

- **Steps 0–7 Frozen**: Steps 0 through 7 remain 100% frozen. No contracts or code in Steps 0–7 were modified.
- **No Global Utility Score**: Step 8 outputs raw bandwidth savings and quality metrics; it does not synthesize a scalar utility score or perform decision selection.
- **Out of Scope**: Edge node selection, resource allocation, and final Adaptive Decision Engine are deferred to Steps 9/10.
