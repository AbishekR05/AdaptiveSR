# Step 5.5 — Hardening Pass 2 (Addendum to Pre-5.5 Hardening Pass)

This addendum closes four remaining gaps identified in review of the Pre-5.5 Hardening Pass
Addendum (§C, §D, §F, §I). It does not reopen any item already resolved (§A, §B, §E, §G, §H).

---

### C2. Crop Anomaly Threshold

`crop_metadata` (§C) is extended with an explicit sanity bound instead of being pure
observability.

```
crop_metadata:
    crop_applied: bool
    pre_crop_width, pre_crop_height: int
    final_width, final_height: int
    crop_pixels_width, crop_pixels_height: int      # replaces crop_pixels_if_available
    crop_within_tolerance: bool
    crop_tolerance_px: int   # default: 8 (covers standard tile-padding remainders)
```

Rule:

- `crop_pixels_width`/`crop_pixels_height` are always populated (0 when no crop).
  The old `_if_available` naming is dropped — the field is either 0 or a positive count,
  never absent.
- `crop_within_tolerance = (crop_pixels_width <= crop_tolerance_px) and
(crop_pixels_height <= crop_tolerance_px)`.
- If `crop_within_tolerance is False`, the trial is marked `flagged: "anomalous_crop"` in
  `metadata["trials"]` and surfaced in the run summary. It is NOT auto-failed — large crops
  may be legitimate for unusual input resolutions — but it can no longer pass silently into
  aggregate statistics unflagged.

---

### D2. CPU-0 Exclusion — Default for Decision-Quality Runs

`exclude_cpu_ids` (§D) remains available but is no longer opt-in for the runs that matter.

- **Smoke test / development benchmark:** `exclude_cpu_ids` defaults to `[]` (core 0
  included), unchanged. Fast iteration, not used for decisions.
- **Decision-quality benchmark (§I):** `exclude_cpu_ids=[0]` is now the **required default**
  for every CPU configuration at or below 2 logical cores. Both variants are run and both are
  reported:
  - `cpu_config_baseline` — core 0 included (comparable to any historical runs).
  - `cpu_config_isolated` — core 0 excluded.
  - `core0_contention_delta` — `mean_latency(baseline) - mean_latency(isolated)`, recorded
    per case, not discarded after computation.

  If `core0_contention_delta` exceeds 10% of `mean_latency(isolated)`, the case is marked
  `flagged: "core0_noise_significant"`. This flag is informational for the CPU-vs-GPU decision,
  not a blocker.

- Above 2 logical cores, `exclude_cpu_ids=[0]` is optional (contention from OS interrupts on a
  single reserved core becomes proportionally smaller as core count grows), but the baseline/
  isolated split described above remains available on request.

---

### F2. P95 as a Structural Field, Not Just a String Note

The `"Exploratory p95; limited tail resolution at n=20."` annotation (§F) is promoted from a
free-text note to a structural field so downstream consumers (schedulers, plotting scripts,
future ML feature pipelines) cannot silently drop it.

```
latency_stats:
    count, mean, median, min, max, stdev: float
    p95: float
    p95_confidence: "exploratory" | "reliable"     # NEW — enum, not prose
    p95_min_recommended_n: int                       # NEW — always 100 for "reliable"
```

Rule:

- `p95_confidence = "exploratory"` whenever `count < 100`. At the current default
  (`count=20`), this is always `"exploratory"`.
- Any consumer reading `p95` MUST branch on `p95_confidence`. The benchmarking library's own
  summary/report generator refuses to print `p95` in a headline comparison table when
  `p95_confidence == "exploratory"` — it falls back to `median`/`max` for headline comparisons
  and moves `p95` to a footnote column. This makes the previous failure mode (a string
  annotation nobody reads) structurally harder to reproduce.

---

### I2. Variance Gate + Per-Config Session Requirement

§I's 3-session minimum is a data-collection requirement; it did not previously gate the
decision itself, and did not specify which CPU-affinity configurations require the minimum.

**Per-config scope**: The 3-session minimum applies **per (model, device, CPU-affinity-config)
tuple**, not once per model/device pair. Concretely, for any CPU decision-quality benchmark,
sessions are collected separately for each of `cpu_config_isolated` at 1, 2, 4, and
(if applicable) full-core configurations — not extrapolated from a single core count.

**Variance gate**: Collecting 3 sessions is necessary but not sufficient. A case is only
eligible to support a CPU-vs-GPU decision if it additionally passes:

```
coefficient_of_variation (between-session, on mean_latency)
    = stdev(session_means) / mean(session_means)
    <= 0.15   (15%)
```

- If `CV > 0.15` for a given (model, device, cpu-config) tuple, the case is marked
  `decision_eligible: False`, with reason `"between_session_variance_exceeds_threshold"`.
  A 4th+ session may be added to attempt to reduce CV rather than accepting the existing 3.
- The CPU-vs-GPU decision document (post-Step-5 deliverable) MUST cite `decision_eligible: True`
  cases only, or explicitly note that a decision is being made on `decision_eligible: False`
  data as a documented risk — silent use of high-variance data is not permitted.
- `thermal_state` (§H) is included alongside `session_means` in the eligibility record, so that
  if CV is high and `thermal_state` was `"not_measured"` across sessions, that correlation is
  visible rather than left to guesswork.

---

### Summary of What Changed vs. Pre-5.5 Hardening Pass

| Item                | Before                                 | After                                                         |
| ------------------- | -------------------------------------- | ------------------------------------------------------------- |
| Crop visibility (C) | Logged, no threshold                   | Logged + tolerance check + `flagged`                          |
| CPU-0 exclusion (D) | Opt-in, core-0 default everywhere      | Required baseline+isolated split for ≤2-core decision runs    |
| P95 confidence (F)  | Free-text annotation                   | Structural enum field + report generator suppression          |
| Session count (I)   | 3 sessions, ungated, scope unspecified | 3 sessions per (model, device, cpu-config), gated by CV ≤ 15% |

No changes to §A, §B, §E, §G, §H — those remain as accepted in the prior pass.
