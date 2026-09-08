# STEP 5.7 — FPS / REAL-TIME FEASIBILITY ANALYSIS
## IMPLEMENTATION DOCUMENTATION

### 1. Objective & Scope
Step 5.7 implements a clean, read-only analysis layer over empirical benchmark results generated in Step 5.5. Its objective is to determine whether measured Super-Resolution (SR) inference latencies fit within the available per-frame time budget dictated by the source video frame rate.

> [!IMPORTANT]
> **SR Inference-Only Scope Disclosure:**
> Step 5.7 evaluates **SR inference-only real-time feasibility**. It does NOT measure or account for video decoding, frame preprocessing, video encoding, network latency, network bandwidth, HTTP/socket overhead, or client playback buffering.
> Therefore, Step 5.7 does NOT establish end-to-end streaming real-time feasibility.

---

### 2. Inputs & Dependencies
- **Primary Input**: Empirical benchmark result records (`BenchmarkResult` or `MultiSessionResult`) produced by Step 5.5 (`adaptive_sr.benchmarking.harness`).
- **Source FPS Metadata**: Joined from Step 5.1 corpus manifests (`data/benchmarks/sr/manifests/*.json`) using the Step 5.5 `input_id` (e.g., `synthetic_lowmotion_30fps`).

---

### 3. Source FPS Provenance & Mathematical Formulas

#### Source FPS Provenance
The source FPS is retrieved from the authoritative Step 5.1 benchmark manifest:
$$\text{input\_id} \longrightarrow \text{benchmark\_manifest.json} \longrightarrow \text{source\_fps}$$
Supported frame rates include 30 FPS, 60 FPS, 120 FPS, and any arbitrary positive numeric FPS value.

#### Mathematical Formulas
1. **Frame Budget ($T_{\text{budget}}$)**:
   $$T_{\text{budget}} = \frac{1000}{\text{source\_fps}} \quad (\text{ms})$$
   - 30 FPS $\rightarrow 33.3333\dots$ ms
   - 60 FPS $\rightarrow 16.6666\dots$ ms
   - 120 FPS $\rightarrow 8.3333\dots$ ms

2. **Estimated Processing FPS ($\text{FPS}_{\text{est}}$)**:
   - For spatial per-frame models (`tinysr`, `tinysr_int8`, `real_esrgan`):
     $$\text{FPS}_{\text{est}} = \frac{1000}{L_{\text{median}}} \quad (L_{\text{median}} \text{ in ms})$$
   - For temporal sequence models:
     $$\text{FPS}_{\text{est}} = \frac{N_{\text{frames}}}{L_{\text{seq, median}}} \quad (L_{\text{seq, median}} \text{ in seconds})$$

3. **Real-Time Ratio ($R_{\text{realtime}}$)**:
   $$R_{\text{realtime}} = \frac{T_{\text{budget}}}{L_{\text{median}}}$$

4. **Measured Feasibility Condition**:
   $$\text{realtime\_measured} = (L_{\text{median}} \le T_{\text{budget}})$$

---

### 4. Semantics & Classification Logic

#### P95 Semantics
- P95 latency is evaluated for exploratory purposes (`real_time_feasible_p95_exploratory = L_{\text{p95}} \le T_{\text{budget}}`).
- P95 confidence (`exploratory` vs `decisive`) is carried forward from Step 5.5.
- **Exploratory P95 is never presented as a decisive headline production feasibility claim.**

#### Decision Eligibility Semantics
- **Measured Feasibility** ($L_{\text{median}} \le T_{\text{budget}}$) and **Decision Eligibility** (`decision_eligible`) are strictly separated.
- A configuration may be measured feasible yet ineligible for adaptive decision making if multi-session variance or session count criteria are unsatisfied (e.g., Step 5.5 requires $\ge 3$ sessions and $CV \le 15\%$).
- **Distinct Concepts Note**: `p95_confidence` (statistically exploratory vs decisive) and `decision_eligible` (multi-session eligibility gate) are distinct metrics. Decision eligibility does NOT imply that P95 itself is decisive.

#### Required 4-Tier Classification Categories
1. **MEASURED + DECISION-ELIGIBLE**: Feasible ($L_{\text{median}} \le T_{\text{budget}}$) AND `decision_eligible == True`.
2. **MEASURED FEASIBLE BUT NOT DECISION-ELIGIBLE**: Feasible ($L_{\text{median}} \le T_{\text{budget}}$) BUT `decision_eligible == False`.
3. **MEASURED UNFEASIBLE**: Infeasible ($L_{\text{median}} > T_{\text{budget}}$).
4. **NOT MEASURED**: Benchmark data unavailable for combination.

---

### 5. Schema & Reporting Formats

#### Machine-Readable Output Schema (`fps_feasibility.json`)
```json
{
    "benchmark_video_id": "synthetic_lowmotion_30fps",
    "model_id": "tinysr",
    "scale": 2,
    "device": "cpu",
    "cpu_ids": [1],
    "num_threads": 2,
    "gpu_device_id": null,
    "source_fps": 30.0,
    "frame_budget_ms": 33.333333333333336,
    "latency_ms": 10.0,
    "p95_latency_ms": 12.0,
    "p95_exploratory": true,
    "latency_interpretation": "per_frame",
    "estimated_processing_fps": 100.0,
    "real_time_ratio": 3.3333333333333335,
    "real_time_feasible": true,
    "real_time_feasible_p95_exploratory": true,
    "budget_utilization_percent": 30.0,
    "decision_eligible": false,
    "session_count": 1,
    "eligibility_reason": "insufficient_sessions_count",
    "p95_confidence": "exploratory",
    "caveats": ["Ineligibility reason: insufficient_sessions_count"],
    "source_fps_gap": false
}
```

---

### 6. Limitations & Known Coverage Gaps
1. **Inference-Only Scope**: Does not measure video decoding, network transport, or client display rendering.
2. **Coverage Limits**: Analyzes only available empirical records from Step 5.5. Absent configurations (e.g. missing GPU runs) report `gpu_benchmark_gap = True`.
3. **Scale Degradation Trends**: Established only when multiple scales for the exact same model are present in Step 5.5 evidence.

---

### 7. Non-Goals
Step 5.7 explicitly does NOT perform:
- Adaptive bitrate / FPS adjustment (ABR)
- Model selection or edge scheduling
- Online learning or ML decision policy optimization
- Real-time video streaming or network emulation
- Modifications to frozen Step 0–5.6 implementations

---

### 8. Verification Instructions & Results Status
To independently execute and verify the Step 5.7 test suite:
```bash
python -m pytest tests/test_fps_feasibility.py -v
python -m pytest tests/ -v
```
> [!NOTE]
> **Empirical Verification Note:**
> Methodological completeness and contract compliance are established in this specification. Independent scientific validation of test results requires running the live test suite on host hardware and verifying raw test runner log outputs. No methodological blockers requiring implementation iterations remain in the specification.
