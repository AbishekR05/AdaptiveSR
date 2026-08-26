# STEP5_7_IMPLEMENTATION.md

## FPS / Real-Time Feasibility Analysis — Implementation Specification

**Status:** Specification only. Not implemented. Antigravity implements from this document.
**Depends on:** Step 5.5 (frozen) benchmark output. Step 5.6 (frozen, pipeline-only) is not a dependency.
**Note on this spec:** No live repository was available for direct inspection in this review session — only `STEP5_IMPLEMENTATION.md` (Steps 5.1–5.5 docs) and the Step 5.6 report chain. Fields below are marked either **[CONFIRMED]** (stated explicitly in frozen docs) or **[VERIFY IN REPO]** (required by this spec but not confirmed in available docs — Antigravity must check the actual Step 5.5 code/output before use, and treat as a documented gap if absent, per §2 rule below).

---

## 1. Scope

Step 5.7 is a **read-only analysis layer** over existing Step 5.5 benchmark output.

MUST NOT:

- Re-run or modify the Step 5.5 harness.
- Perform scheduling, allocation, or cluster/CPU/GPU selection.
- Claim end-to-end streaming feasibility (see §5).
- Fabricate any field not present in Step 5.5 output.

---

## 2. Step 5.5 Schema — Confirmed Fields vs. Gaps

**[CONFIRMED]** present in Step 5.5 output (`STEP5_IMPLEMENTATION.md` §5.5):

- `model_id` — selected adapter ID (§5.5.1 item 3).
- `scale` — validated against adapter capability (§5.5.1 item 1, item 7).
- `device` — `"cpu"` or `"cuda:N"` execution path (§5.5.2).
- Per-trial latency array, from which count/mean/median/min/max/stddev/p95 are computed (§5.5.3). p95 uses `np.percentile(latencies, 95)`, documented as the **exploratory-only** statistic per project hardening history — n=20 is not sufficient for p95 to be a strong decision metric.
- `warmup` count (default 3) excluded from steady-state stats (§5.5.1 item 5, §5.5 §6).
- Trial count (default 20) (§5.5.1 item 6).
- CPU: `cpu_ids` (affinity) and `num_threads` are separate configuration dimensions (Step 5.3 `CPUExecutionConfig`, reused by 5.5 §2).
- GPU: `device_id` for `cuda:N` targeting; `torch.cuda.synchronize()` boundaries used for timing (§5.5.1 item 6, §5.5.2).
- Per-trial failure capture: `metadata["trials"]` records success/failure with traceback (§5.5.5).
- Multi-session CV≤15% variance eligibility gate, scoped per (model, device, cpu-config), with a documented `p95_confidence` structural enum and decision-eligibility criterion (per Step 5.5 hardening addendum record in project memory — §I2/§F2 correction pass). **[VERIFY IN REPO]** exact field name for this eligibility flag and the `p95_confidence` enum values.
- CPU-0 exclusion mandated for ≤2-core decision-quality runs (§D2 correction, per project memory) — **[VERIFY IN REPO]** how this is surfaced in output (a boolean flag vs. an absence of core 0 in `cpu_ids`).

**[VERIFY IN REPO]** — not confirmed present, do not assume:

- `benchmark_video_id` / `input_id` — per Step 5.6's own investigation (frozen), `input_id` was confirmed present in Step 5.5 config, `benchmark_video_id` and `chunk_id` were confirmed **absent** (Step-5.6-only additions). Step 5.7 must re-verify this directly rather than reuse Step 5.6's finding as authoritative, since Step 5.7 is a sibling consumer, not a downstream dependent of 5.6.
- `chunk_id` — confirmed **absent** from Step 5.5 per Step 5.6 investigation; Step 5.5 benchmarks operate on a single loaded chunk's first frame (spatial) or full frame list (temporal), not multiple chunks per trial (§5.5.1 item 2). This directly affects §9 below.
- `source_fps` — not mentioned anywhere in Step 5.5 docs. **This is the single most important gap to verify**, since Step 5.7's entire purpose depends on it. If absent from Step 5.5 output, Step 5.7 must obtain it by joining against the Step 5.1 benchmark manifest (`benchmark_video_id` → `source_fps`, confirmed present in Step 5.1 manifest per §5.1 §8) — NOT by hardcoding or assuming 30/60/120.
- GPU session identity / multi-GPU disambiguation beyond `device_id` — **[VERIFY IN REPO]**.

**Gap-handling rule:** Where a **[VERIFY IN REPO]** field turns out to be absent, Step 5.7 output must include an explicit `"<field>_gap": true` / documented-absence marker rather than omitting it silently or defaulting it.

---

## 3. Real-Time Feasibility Definition

T_budget_ms = 1000.0 / source_fps

**Primary decision statistic: median latency**, not mean, not p95.

Rationale (must be preserved in implementation comments/report):

- Mean is sensitive to tail outliers from cold-cache/contention trials.
- p95 at n=20 is explicitly exploratory per Step 5.5 hardening history — must not be silently promoted to the primary real-time decision metric.
- Median is the most robust already-supported statistic for a binary feasibility classification.

**Secondary/reporting statistic:** p95 latency is still reported alongside median, explicitly labeled `p95_exploratory: true` in every record, and MUST NOT be used to compute `real_time_feasible`.

real_time_feasible = (median_latency_ms <= T_budget_ms)

Also report a p95-based feasibility flag separately for transparency, always labeled exploratory:

real_time_feasible_p95_exploratory = (p95_latency_ms <= T_budget_ms)

---

## 4. Metrics

Per benchmark case, compute and expose exactly:

| Field                                | Formula / Source                                                  |
| ------------------------------------ | ----------------------------------------------------------------- |
| `latency_ms`                         | `median_latency_seconds * 1000` (primary statistic)               |
| `p95_latency_ms`                     | `p95_latency_seconds * 1000` (exploratory, reported not decisive) |
| `estimated_processing_fps`           | `1000.0 / latency_ms`                                             |
| `source_fps`                         | from Step 5.1 manifest join (see §2 gap)                          |
| `frame_budget_ms`                    | `1000.0 / source_fps`                                             |
| `real_time_ratio`                    | `frame_budget_ms / latency_ms`                                    |
| `real_time_feasible`                 | `latency_ms <= frame_budget_ms` (median-based)                    |
| `real_time_feasible_p95_exploratory` | `p95_latency_ms <= frame_budget_ms`                               |
| `budget_utilization_percent`         | `(latency_ms / frame_budget_ms) * 100`                            |

No additional derived metrics. Do not add a "confidence score" or composite index — out of scope.

---

## 5. Pipeline Distinction — SR Inference-Only, Not End-to-End

**Determination:** Step 5.5 measures **SR inference latency only** (§5.5.1: adapter `.process()` call timing, explicitly isolated from "video file decoding... network transfer... encoding" per §5.5 §4).

Step 5.7 MUST:

- Label every result and every report section **"SR inference real-time feasibility"**, never "streaming feasibility" or "end-to-end feasibility."
- Include a fixed disclosure block (top of report and top of machine-readable output) stating: _"These results measure SR model inference latency only. Decoding, preprocessing, encoding, and network transfer are NOT included. End-to-end streaming real-time feasibility cannot be established from Step 5.5/5.7 data alone."_
- Never compute or imply a combined "streaming FPS" number.

---

## 6. Scale Analysis

- Group results by `(model_id, device)`, list all benchmarked `scale` values from actual records only — do not assume x2/x3/x4 exist for every model (per Step 5.2, `real_esrgan` supports `[2,4]` only, `tinysr_int8` supports `[2]` only, `tinysr` supports `[2,3,4]`).
- Per group, report latency/estimated FPS/real_time_feasible trend across available scales — do not interpolate or estimate untested scales.
- If a model+scale combination has no Step 5.5 record, omit it — do not synthesize.

---

## 7. CPU vs GPU Analysis

For each `(model_id, scale)`, present CPU and GPU rows side by side where both exist, preserving:

- `cpu_ids`, `num_threads` (CPU rows)
- `device_id` (GPU rows, `cuda:N`)
- Eligibility/session fields (§8)

Step 5.7 exposes this comparison as **evidence only** — no CPU-vs-GPU recommendation, selection, or allocation decision is made here (explicit non-goal, confirmed in prompt §7/§14).

---

## 8. Session / Variance Handling

Every Step 5.7 record MUST carry, unmodified from Step 5.5:

- `decision_eligible: bool` — **[VERIFY IN REPO exact field name]**, derived from Step 5.5's CV≤15% multi-session gate.
- `session_count` used for the eligibility determination.
- `p95_confidence` enum value (structural, not free text, per Step 5.5 hardening).
- Any core-0 contention or GPU-contamination warning strings already attached to the Step 5.5 record.

**Rule:** If `decision_eligible == false`, Step 5.7 still computes and reports `real_time_feasible` (measured feasibility) but the record must carry `decision_eligible: false` alongside it, and the human-readable report must visually distinguish these rows (e.g., separate table section or a caveat column) — never merge ineligible results into headline "X is real-time" claims.

Two distinct concepts, both must appear in output:

- **`real_time_feasible`** — measured, from this specific benchmark run's median latency.
- **`decision_eligible AND real_time_feasible`** — the only combination safe to use for any later scheduling/allocation decision in Step 5.8+.

---

## 9. Chunk-Level Interpretation

**Determination (per §2 gap findings):** Step 5.5 does not benchmark multiple chunks per trial — it loads a single chunk, extracts either the first frame (spatial models) or the full frame list (temporal models) per §5.5.1 item 2. Therefore:

- For **spatial models** (`tinysr`, `tinysr_int8`, `real_esrgan`): the measured `latency_ms` is **per-frame SR processing latency**. `estimated_processing_fps` is a valid per-frame throughput estimate under sequential (batch=1) execution.
- For **temporal models** (`basicvsr++`, if benchmarked): the measured latency is **per-sequence (multi-frame window) latency**, not per-frame. `estimated_processing_fps` MUST be computed as `sequence_frame_count / latency_seconds`, not `1/latency_seconds`, and the record must carry `latency_interpretation: "per_frame" | "per_sequence"` explicitly.
- Step 5.7 MUST NOT assume a single-chunk latency measurement extrapolates linearly to full-chunk-duration throughput (e.g., a 2-second chunk at 30fps = 60 frames) without stating this as an assumption. Since Step 5.5 measures single-frame (or single-window) latency, not full-chunk latency, any full-chunk-duration estimate is a **derived extrapolation**, not a direct measurement — label any such number `chunk_duration_estimate_derived: true` if computed, and do not present it as measured.

**Default behavior:** Step 5.7's primary output is per-frame (or per-sequence, for temporal) latency-based feasibility, NOT a full-chunk-duration projection. Full-chunk projection is optional/derived (see §14).

---

## 10. Output Schema

```json
{
  "benchmark_video_id": "string | null (gap-flagged if unavailable)",
  "model_id": "string",
  "scale": "int",
  "device": "string",
  "cpu_ids": "[int] | null",
  "num_threads": "int | null",
  "gpu_device_id": "int | null",
  "source_fps": "float | null (gap-flagged if unavailable)",
  "frame_budget_ms": "float | null",
  "latency_ms": "float",
  "p95_latency_ms": "float",
  "p95_exploratory": true,
  "latency_interpretation": "per_frame | per_sequence",
  "estimated_processing_fps": "float",
  "real_time_ratio": "float | null",
  "real_time_feasible": "bool | null",
  "real_time_feasible_p95_exploratory": "bool | null",
  "budget_utilization_percent": "float | null",
  "decision_eligible": "bool",
  "session_count": "int",
  "p95_confidence": "enum (from Step 5.5)",
  "caveats": ["array of preserved warning strings from Step 5.5"],
  "chunk_duration_estimate_derived": "bool (only present if a derived full-chunk projection is included)",
  "source_fps_gap": "bool (true if source_fps could not be resolved)"
}
```

Field naming reuses Step 5.5 conventions exactly (`model_id`, `scale`, `device`, `cpu_ids`, `num_threads`) — no renaming.

---

## 11. Human-Readable Report

Table:

| Model | Scale | Device | Latency (median, ms) | Est. FPS | Source FPS | Budget (ms) | Ratio | Real-Time (measured) | Decision-Eligible |
| ----- | ----: | ------ | -------------------: | -------: | ---------: | ----------: | ----: | -------------------- | ----------------- |

Include only columns backed by §10 schema. Omit any column whose source field is a documented gap for that row (show `N/A`, never blank/silent).

Report sections:

1. Disclosure block (§5).
2. Fastest configuration overall (by median latency, decision-eligible only).
3. Configurations meeting real-time target, split into "measured feasible" vs "measured + decision-eligible."
4. Configurations failing real-time target.
5. Scale-degradation summary (§6).
6. CPU vs GPU comparison table (§7) — evidence only, no verdict.

No quality claims (PSNR/SSIM/VMAF) — that's Step 5.6's domain, do not reintroduce it here.

---

## 12. Edge Cases (No Silent Fallback)

| Case                                                                  | Required behavior                                                                                                                                                                           |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Zero/negative latency                                                 | Raise explicit error — invalid Step 5.5 record, do not compute ratio.                                                                                                                       |
| Missing `source_fps`                                                  | Set `source_fps_gap: true`, `real_time_feasible: null`, `frame_budget_ms: null`. Do not default to 30.                                                                                      |
| Unsupported model+scale                                               | Omit from output; do not fabricate a record.                                                                                                                                                |
| Missing benchmark record for expected combination                     | Log as gap in report summary; do not silently skip without mention.                                                                                                                         |
| Incomplete session data                                               | `decision_eligible: false`, carry through actual `session_count`.                                                                                                                           |
| Ineligible multi-session result                                       | Still compute `real_time_feasible`; flag `decision_eligible: false` per §8.                                                                                                                 |
| Exploratory p95                                                       | Always labeled `p95_exploratory: true`; never sole basis for `real_time_feasible`.                                                                                                          |
| CPU-only machine                                                      | GPU rows simply absent; no error.                                                                                                                                                           |
| GPU unavailable                                                       | Same as above; do not fabricate GPU rows.                                                                                                                                                   |
| Missing device metadata                                               | Raise explicit error — cannot classify CPU vs GPU without it.                                                                                                                               |
| Multiple GPUs                                                         | Each `device_id` treated as independent row, no aggregation.                                                                                                                                |
| Extremely high latency                                                | Computed normally; `real_time_ratio` will be «1, `real_time_feasible: false`. No special-casing/clamping.                                                                                   |
| FPS target mismatch (e.g., user wants 60fps eval but source is 30fps) | Not in scope — Step 5.7 uses `source_fps` from the corpus only, does not evaluate against an arbitrary target FPS. If a target-FPS comparison is desired later, that's a Step 5.8+ concern. |
| Chunk-level data without frame-level latency                          | Apply §9 `latency_interpretation` rule; never silently treat sequence latency as per-frame.                                                                                                 |

---

## 13. Tests (Minimum Set)

1. `test_frame_budget_calculation` — `T_budget_ms` correctness for 30/60/120 fps.
2. `test_estimated_fps_calculation` — inverse-latency correctness, including temporal per-sequence case (§9).
3. `test_real_time_ratio` — ratio correctness, including boundary case `ratio == 1.0`.
4. `test_real_time_classification_median_based` — confirms classification uses median, NOT mean or p95.
5. `test_p95_never_drives_classification` — explicit regression test asserting `real_time_feasible` is unaffected by p95 value alone.
6. `test_scale_comparison_omits_unsupported_combinations` — model+scale not in adapter capability list is excluded, not fabricated.
7. `test_cpu_gpu_rows_separated` — no accidental merging of CPU/GPU records into one row.
8. `test_missing_source_fps_sets_gap_flag` — no silent 30fps default.
9. `test_invalid_latency_raises` — zero/negative latency raises, does not compute silently.
10. `test_eligibility_warnings_preserved` — Step 5.5 caveats/warnings pass through unmodified into Step 5.7 output.
11. `test_ineligible_session_still_reports_measured_feasibility` — `decision_eligible=false` rows still get `real_time_feasible` computed, correctly flagged.
12. `test_output_schema_matches_spec` — validates §10 schema shape.
13. `test_integration_real_step5_5_fixture` — **required**, uses an actual Step 5.5 result record (not a synthetic mock) to run the full Step 5.7 pipeline end-to-end.

---

## 14. Explicit Non-Goals

- No modification to Step 5.5.
- No scheduling, cluster selection, or CPU/GPU allocation.
- No network-path optimization.
- No end-to-end streaming latency claims beyond what's measured (§5).
- No model retraining or adapter changes.
- No FLOPS-based theoretical estimates — measured latency only.
- Full-chunk-duration projection (§9) is optional and, if implemented, must be clearly derived/labeled — not a required Step 5.7 deliverable.

---

## 15. Freeze Gate

Step 5.7 freezes only when:

1. Step 5.7 consumes actual Step 5.5 output records (not mocks) in at least one integration test.
2. `source_fps` resolution is either confirmed present in Step 5.5 output or correctly joined from the Step 5.1 manifest — no hardcoded/default FPS anywhere.
3. `frame_budget_ms`, `estimated_processing_fps`, `real_time_ratio` formulas verified correct by unit tests.
4. `real_time_feasible` classification is deterministically median-based; p95 is present but provably non-decisive (test §13.5).
5. CPU and GPU records remain distinguishable in both schema and report, with no merged/aggregated rows.
6. `decision_eligible` and all Step 5.5 caveats/warnings are preserved verbatim in every output record — verified by test.
7. §9 chunk/frame interpretation is explicit per record (`latency_interpretation` field populated, no implicit assumption).
8. Report contains the SR-inference-only disclosure (§5) verbatim, and no end-to-end streaming claim appears anywhere in output or report.
9. All §13 tests pass, including the mandatory Step 5.5-fixture integration test.
10. Every **[VERIFY IN REPO]** item in §2 has been resolved during implementation — either confirmed present (field wired through) or documented as an explicit gap (`*_gap: true` pattern) in the final report. This spec cannot be marked "faithfully implemented" while any §2 item remains unverified.
