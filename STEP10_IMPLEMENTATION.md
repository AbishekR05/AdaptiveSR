# Step 10 — Fuzzy Adaptive Decision Engine Implementation Documentation

## Overview

Step 10 implements the **Fuzzy Adaptive Decision Engine** (`adaptive_sr/adaptation/fuzzy_engine.py`). It consumes validated signals from Steps 7–9:
- Step 7: `FPSAdaptationSignal`
- Step 8: `BitrateAdaptationSignal`
- Step 9: `EdgeResourceSignal`

It constructs candidate SR configurations, applies a **Hard Feasibility Gate**, evaluates a 5-variable **Mamdani Fuzzy Inference Engine**, and defuzzifies candidate suitability using **Centroid Defuzzification**. The engine selects the optimal feasible Super-Resolution (SR) configuration (`FuzzyDecisionSignal`) without fabricating measurements or using arbitrary weighted utility coefficients.

---

## 1. Candidate Construction & Matching

Candidate configurations are constructed by matching tuple keys across input signals:
$$\text{CandidateKey} = (\text{base\_representation\_id}, \text{target\_resolution}, \text{model\_id}, \text{scale}, \text{device}, \text{edge\_id})$$

Every candidate configuration preserves:
- `base_representation_id` (e.g. `360p`, `480p`)
- `target_resolution` (e.g. `1280x720`, `1920x1080`)
- `model_id` (e.g. `tinysr`, `real_esrgan`)
- `scale` (e.g. `2`, `4`)
- `device` (e.g. `cpu`, `cuda`)
- `edge_id` (e.g. `edge_01`)

---

## 2. Hard Feasibility Gate

Before fuzzy inference, candidates are filtered through a strict Hard Feasibility Gate. Hard infeasibility cannot be overridden by fuzzy scoring.

A candidate is rejected if:
1. `resource_feasible == False` or `feasibility_status == "infeasible"` in Step 9.
2. Device is `cuda` but physical GPU is unavailable (`gpu_available == False`).
3. Requested `model_id` is not in the candidate node's `supported_models`.
4. `real_time_ratio` is missing, undefined, or $\le 0.0$.
5. Required Step 7, Step 8, or Step 9 signal is missing.

All hard-rejected candidates are logged in `rejected_candidates` with explicit `rejection_reasons`.

---

## 3. Fuzzy Variables & Membership Functions

The fuzzy inference engine uses 5 continuous input variables and 1 continuous output variable.

### Input 1: `real_time_ratio` ($r \ge 0$)
Ratio of frame budget to measured SR latency ($r = \frac{\text{frame\_budget\_ms}}{\text{latency\_ms}}$).
- `poor`: Trapezoid $[0.0, 0.0, 0.5, 0.8]$ — $r \le 0.5$ is fully poor; $r > 0.8$ is $0$.
- `moderate`: Triangle $[0.6, 0.85, 1.1]$ — Centered around $0.85$.
- `good`: Trapezoid $[0.95, 1.2, 3.0, 3.0]$ — $r \ge 1.2$ is fully good.
*Justification*: $r = 1.0$ represents exact real-time playback boundary. Ratios $> 1.2$ offer compute headroom.

### Input 2: `bandwidth_saving_percent` ($b \in [-50\%, 100\%]$)
Percentage bandwidth saving relative to native reference stream.
- `low`: Trapezoid $[-50.0, -50.0, 0.0, 25.0]$ — Savings $\le 0\%$ are fully low.
- `medium`: Triangle $[15.0, 35.0, 55.0]$ — Centered around $35\%$ saving.
- `high`: Trapezoid $[45.0, 65.0, 100.0, 100.0]$ — Savings $\ge 65\%$ are fully high.
*Justification*: Negative/zero saving offers no network gain; $>50\%$ saving provides substantial network relief.

### Input 3: `quality_suitability` ($q \in [0.0, 1.0]$)
Normalized visual quality score derived transparently from valid available metrics:
$$q_{\text{VMAF}} = \frac{\text{VMAF}}{100}, \quad q_{\text{PSNR}} = \min\left(1.0, \max\left(0.0, \frac{\text{PSNR} - 25}{15}\right)\right), \quad q_{\text{SSIM}} = \text{SSIM}$$
Average of valid normalized scores is used. If no valid quality metrics are present, defaults to neutral ($0.5$) with a warning.
- `poor`: Trapezoid $[0.0, 0.0, 0.3, 0.5]$
- `acceptable`: Triangle $[0.4, 0.6, 0.8]$
- `good`: Trapezoid $[0.7, 0.85, 1.0, 1.0]$

### Input 4: `edge_resource_condition` ($c \in [0.0, 1.0]$)
Available compute headroom derived from $c = 1.0 - \frac{\text{cpu\_utilization}}{100}$.
- `constrained`: Trapezoid $[0.0, 0.0, 0.15, 0.35]$ (CPU load $> 85\%$)
- `moderate`: Triangle $[0.25, 0.50, 0.75]$
- `available`: Trapezoid $[0.65, 0.85, 1.0, 1.0]$ (CPU load $< 35\%$)

### Input 5: `network_condition` ($n \in [0.0, 1.0]$)
Normalized network quality derived from RTT or bandwidth telemetry.
- `poor`: Trapezoid $[0.0, 0.0, 0.25, 0.50]$
- `moderate`: Triangle $[0.35, 0.60, 0.80]$
- `good`: Trapezoid $[0.70, 0.85, 1.0, 1.0]$

---

## 4. Complete Rule Base & Mamdani Inference

The rule base uses AND = $\min()$ and OR = $\max()$ antecedent operators:

| Rule | Antecedents | Output Level | Rationale |
|---|---|---|---|
| **R1** | `real_time_ratio` IS good **AND** `bandwidth_saving` IS high **AND** `quality` IS good **AND** `resource_condition` IS available **AND** `network_condition` IS good | `very_high` | Optimal streaming configuration |
| **R2** | `real_time_ratio` IS good **AND** `bandwidth_saving` IS high **AND** `quality` IS acceptable **AND** `resource_condition` IS moderate | `high` | Strong bandwidth saving with acceptable quality and compute |
| **R3** | `real_time_ratio` IS good **AND** `quality` IS good **AND** `bandwidth_saving` IS medium **AND** `resource_condition` IS available | `high` | High visual quality with compute headroom |
| **R4** | `real_time_ratio` IS moderate **AND** `bandwidth_saving` IS medium **AND** `quality` IS acceptable | `medium` | Balanced trade-off candidate |
| **R5** | `bandwidth_saving` IS low **AND** `quality` IS poor | `low` | Minimal bandwidth gain for low visual quality |
| **R6** | `real_time_ratio` IS poor **OR** `resource_condition` IS constrained | `very_low` | Severe computational or load bottleneck |
| **R7** | `real_time_ratio` IS poor | `very_low` | Cannot sustain real-time playback budget |
| **R8** | `network_condition` IS poor | `low` | Adverse network path impairs delivery reliability |

---

## 5. Output Variable & Centroid Defuzzification

Output variable `adaptation_suitability` spans $[0, 100]$:
- `very_low`: Trapezoid $[0, 0, 10, 25]$
- `low`: Triangle $[15, 30, 45]$
- `medium`: Triangle $[35, 50, 65]$
- `high`: Triangle $[55, 70, 85]$
- `very_high`: Trapezoid $[75, 90, 100, 100]$

Defuzzification uses continuous Centroid (Center-of-Area) integration over $y \in [0, 100]$ with step $\Delta y = 0.5$:
$$y^* = \frac{\int_0^{100} y \cdot \mu_{\text{agg}}(y) \, dy}{\int_0^{100} \mu_{\text{agg}}(y) \, dy}$$
where $\mu_{\text{agg}}(y) = \max_{k} \left(\min\left(w_k, \mu_{T_k}(y)\right)\right)$.

---

## 6. Selection, Minimum Threshold, & Tie-Breaking

1. **Minimum Suitability Threshold**: Configurable parameter `min_suitability_threshold` (default: $35.0$).
2. **No Suitable Candidate**: If no candidate reaches $35.0$, returns `decision = "no_suitable_candidate"`.
3. **Deterministic Tie-Breaking**: Candidates meeting the threshold are sorted by:
   - Primary: `defuzzified_suitability` (descending)
   - Secondary: `real_time_ratio` (descending)
   - Tertiary: `bitrate_saving_percent` (descending)
   - Quaternary: `candidate_id` (alphabetical ascending)

---

## 7. Machine-Readable Signal Schema (`FuzzyDecisionSignal`)

```json
{
  "decision": "selected",
  "selected_candidate": {
    "candidate_id": "edge_01:360p:1280x720:tinysr:x2:cpu",
    "edge_id": "edge_01",
    "base_representation_id": "360p",
    "target_resolution": "1280x720",
    "model_id": "tinysr",
    "scale": 2,
    "device": "cpu",
    "defuzzified_suitability": 84.15,
    "suitability_label": "very_high"
  },
  "selected_edge_id": "edge_01",
  "selected_representation_id": "360p",
  "target_resolution": "1280x720",
  "model_id": "tinysr",
  "scale": 2,
  "device": "cpu",
  "fuzzy_suitability": 84.15,
  "suitability_tier": "very_high",
  "min_suitability_threshold": 35.0,
  "candidate_evaluations": [...],
  "rejected_candidates": [...],
  "input_signal_provenance": {
    "fps": "step7_fps_adaptation",
    "bitrate": "step8_bitrate_adaptation",
    "edge": "step9_edge_resource"
  },
  "rule_inference_metadata": {
    "rule_count": 8,
    "inference_engine": "Mamdani_Centroid",
    "defuzzification_domain": "[0, 100]"
  },
  "warnings": [],
  "baseline_comparison_ready": true
}
```

---

## 8. Verification & Test Summary

Command executed:
```powershell
D:\Abishek\venv\Scripts\python.exe -m pytest tests/test_fuzzy_decision.py tests/test_edge_selection.py tests/test_bitrate_adaptation.py tests/test_fps_adaptation.py tests/test_remote_sr.py tests/test_foundation.py -v
```

### Test Results (68/68 Passed):
- `test_single_feasible_candidate`: Passed
- `test_multiple_feasible_candidates_highest_suitability_selected`: Passed
- `test_hard_infeasible_candidate_rejected`: Passed
- `test_cuda_requested_gpu_unavailable_hard_rejected`: Passed
- `test_tie_breaking_determinism`: Passed
- `test_minimum_suitability_threshold`: Passed
- `test_missing_quality_metrics_handling`: Passed
- `test_missing_network_telemetry_handling`: Passed
- `test_missing_resource_telemetry_handling`: Passed
- `test_realtime_feasible_vs_decision_eligible_distinction`: Passed
- `test_provenance_preservation`: Passed
- `test_deterministic_repeated_decisions`: Passed
- `test_integration_real_step7_8_9_signals`: Passed
- Previous test suites (`test_edge_selection.py`, `test_bitrate_adaptation.py`, `test_fps_adaptation.py`, `test_remote_sr.py`, `test_foundation.py` - 55 tests): Passed

---

## 9. Frozen Boundaries & Limitations

- **Steps 0–9 Frozen**: Steps 0 through 9 remain 100% frozen. No prior contracts or interfaces were altered.
- **Scope Boundary**: Step 10 selects the optimal feasible SR configuration (`representation`, `model`, `scale`, `device`, `edge_id`). It does NOT handle client video playback, end-to-end streaming loops, or final evaluation campaigns (deferred to Step 12).
- **Baseline Readiness**: Output structure includes `baseline_comparison_ready: true` to support direct comparative evaluation against non-fuzzy heuristic baselines in Step 12.
