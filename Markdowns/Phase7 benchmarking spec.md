# Phase 7 Spec — Benchmarking & Evaluation Harness

Status: Phases 1–6 complete. Full adaptive pipeline runs end-to-end, logs per-frame
telemetry, and switches between `tinysr`/`real_esrgan` correctly. This document specifies
Phase 7: the experiments that actually prove your thesis claim — that adaptive
orchestration beats static execution.

**This is the phase your entire results chapter is built from. Get the harness right
before running big experiments — a bug here means re-running everything.**

---

## 0. Runtime Budget Reality Check — Read First

Real-ESRGAN is ~6s/frame on your GPU (Phase 6 measured). A "always real_esrgan" baseline
over a 300-frame (10s) clip is **30 minutes**, and you need **4 configs** (TinySR-only,
Real-ESRGAN-only, Adaptive, and whatever ablation variants) per test video, across
**multiple test categories** (landscape, faces, busy street, etc. per your design doc).

Do the arithmetic before committing to a dataset size: 5 categories × 4 configs × 10s
clips ≈ 5 × 4 × ~30min (dominated by the real_esrgan-only baseline) ≈ **10 hours of
unattended compute**, not counting reruns for bugs. This is fine if planned for, brutal if
discovered halfway through.

**Recommendation**: use **short clips (5-8 seconds, ~150-240 frames) per category**, run
them **overnight or unattended** rather than interactively, and build the harness so a
crash partway through doesn't lose completed runs (see section 2). Scale up clip length
only for your single best "hero" demo video used in visual comparisons, not across the
whole evaluation matrix.

---

## 1. Test Video Set

Per your design doc's categories, you don't need all of them — pick 4-5 that best
demonstrate contrast:

- **Flat/simple** (sky, static wall, slow pan) — expect heavy `tinysr` selection
- **Complex/static** (crowded street photo held still, dense texture)
- **Human face close-up** — expect heavy `real_esrgan` selection
- **Mixed** (cuts between simple and complex shots) — expect visible switching, this is
  your headline "adaptive beats static" video

Real-world source (downloaded/user-recorded), downscaled to 480p, matching Phase 1's
prep convention. Keep each clip 5-8s per the budget note above.

### Reference (ground-truth) strategy for quality metrics

You won't have true HR references for downloaded video. Standard workaround, and it
matches Base Paper 3's real-world-degradation framing already in your lit review:

1. Source a genuinely higher-resolution clip (e.g. 1080p or better)
2. Downscale it to 480p — this becomes your pipeline's LR input
3. The original higher-res clip is your ground truth for PSNR/SSIM/LPIPS

Do this for at least your "mixed" category video — you don't need ground truth for every
clip (the model-selection-distribution and system-metrics experiments don't need it), but
your visual-quality numbers do.

---

## 2. Benchmark Runner — `benchmark/run_baselines.py`

```python
CONFIGS = {
    "baseline_tinysr":     {"force_model": "tinysr"},
    "baseline_real_esrgan": {"force_model": "real_esrgan"},
    "adaptive":            {"force_model": None},  # normal DecisionEngine behavior
}

def run_experiment(video_path, config_name, config, output_dir):
    result_path = output_dir / f"{video_path.stem}__{config_name}.mp4"
    log_path = output_dir / f"{video_path.stem}__{config_name}.csv"
    if result_path.exists() and log_path.exists():
        print(f"SKIP (already done): {config_name} on {video_path.name}")
        return  # resumability — critical given section 0's runtime budget
    run_pipeline(video_path, result_path, log_path,
                 force_model=config.get("force_model"))

for video in TEST_VIDEOS:
    for config_name, config in CONFIGS.items():
        run_experiment(video, config_name, config, OUTPUT_DIR)
```

**The `if result_path.exists(): skip` check is not optional.** Given multi-hour unattended
runs, your process _will_ get interrupted at some point (laptop sleep, crash, you closing
the lid). Without resumability, one interruption costs you everything completed so far.

To support `force_model`, add a small parameter to `main.py`'s pipeline runner: if set,
skip the `DecisionEngine.decide()` call entirely and always return a fixed `Decision` for
that model — don't hack this by editing `decision_config.yaml` thresholds to force one
model, that's fragile and easy to forget to revert.

---

## 3. Quality Metrics — `benchmark/compute_quality_metrics.py`

```python
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

lpips_model = lpips.LPIPS(net='alex')  # 'alex' is faster and standard for this use

def compute_metrics(enhanced_frame, reference_frame):
    # Ensure same dimensions — resize reference to match enhanced output if needed
    p = psnr(reference_frame, enhanced_frame, data_range=255)
    s = ssim(reference_frame, enhanced_frame, channel_axis=2, data_range=255)
    l = lpips_model(to_tensor(enhanced_frame), to_tensor(reference_frame)).item()
    return {"psnr": p, "ssim": s, "lpips": l}
```

Run this per-frame against your ground-truth video (section 1), for each of the 3
baseline outputs, average across the clip. This produces the core comparison table:

| Config               | Avg PSNR | Avg SSIM | Avg LPIPS |
| -------------------- | -------- | -------- | --------- |
| baseline_tinysr      |          |          |           |
| baseline_real_esrgan |          |          |           |
| adaptive             |          |          |           |

**Expected honest result, state this going in**: `adaptive` will likely land _between_
the two static baselines on pure reconstruction quality (PSNR/SSIM/LPIPS) — that's not a
failure, that's the correct tradeoff your thesis argues for. The story isn't "adaptive has
the best quality," it's "adaptive gets quality close to the expensive baseline at a
fraction of the compute cost" — which is why the system metrics table (next section)
matters just as much as this one.

---

## 4. System Metrics — from your existing per-frame CSV logs

No new instrumentation needed — Phase 6's `PipelineLogger` already captures everything.
Aggregate across each run's CSV:

```python
import pandas as pd

def summarize_run(csv_path):
    df = pd.read_csv(csv_path)
    return {
        "total_time_s": df["inference_time_ms"].sum() / 1000,
        "avg_cpu": df["cpu"].mean(),
        "avg_gpu": df["gpu"].mean(),
        "battery_delta": df["battery"].iloc[0] - df["battery"].iloc[-1] if df["battery"].notna().any() else None,
        "avg_temp": df["temperature"].mean() if df["temperature"].notna().any() else None,
        "model_distribution": df["selected_model"].value_counts(normalize=True).to_dict(),
    }
```

Produces your second core table:

| Config               | Total Time | Avg CPU | Avg GPU | Battery Δ | Model Distribution         |
| -------------------- | ---------- | ------- | ------- | --------- | -------------------------- |
| baseline_tinysr      |            |         |         |           | 100% tinysr                |
| baseline_real_esrgan |            |         |         |           | 100% real_esrgan           |
| adaptive             |            |         |         |           | X% tinysr / Y% real_esrgan |

This table is where your thesis's actual headline number comes from — e.g. "adaptive
achieved N% of real_esrgan's PSNR at M% of its compute time."

---

## 5. Decision Stability Metric

You flagged this as an open question back in Phase 4. Now you have real per-frame logs —
answer it:

```python
def decision_stability(csv_path):
    df = pd.read_csv(csv_path)
    switches = (df["selected_model"] != df["selected_model"].shift()).sum() - 1  # -1 for first row
    return switches / len(df)  # switch rate: fraction of frames where model changed vs prev frame
}
```

Report this for your `adaptive` runs. If it's high (frequent flip-flopping frame to
frame, especially on your "mixed" category clip), that's a legitimate finding — it tells
you hysteresis (deferred back in Phase 4) would be a valuable Phase 8 addition, and you
can say so with actual evidence instead of a hypothetical. If it's low/stable, that's
equally good evidence the rule-based engine already behaves reasonably. Either outcome is
a real result — don't tune thresholds after the fact just to make this number look better.

---

## 6. Ablation Study (optional, do if time allows — not blocking)

Per your design doc: compare adaptive-with-full-signals against adaptive-with-one-signal-
disabled, to show each module's individual contribution:

- **Ablation A**: Decision Engine ignores Device Monitor (always assumes full budget)
- **Ablation B**: Decision Engine ignores Complexity Estimator (always assumes fixed
  mid-complexity)
- **Full**: both active (your normal adaptive run)

Cheapest way to implement: add `ignore_device` / `ignore_scene` flags to
`DecisionEngine.decide()` that substitute a fixed dummy value for the ignored input
before applying rules — don't build separate engine classes for this.

This is genuinely valuable for the thesis (isolates each module's contribution) but real
compute cost on top of section 0's budget — treat as time-permitting, not required for
Phase 7 to be "done."

---

## 7. Validation / Milestone

### Test 1 — Resumability

Kill the benchmark runner mid-execution (Ctrl+C after 1-2 experiments complete), restart
it, confirm it skips completed runs and resumes rather than re-running or crashing.

### Test 2 — Full comparison table produced

All 3 configs × your chosen test videos complete, quality metrics computed against
ground truth for at least the "mixed" category, system metrics aggregated for all.

### Test 3 — Sanity-check the expected tradeoff

Confirm `adaptive`'s quality metrics land between the two static baselines (not below
both — if adaptive is worse than _both_ baselines on quality, something's wrong with
either the Decision Engine logic or the metrics computation, investigate before writing
this up as a result).

### Report format

This report doubles as your thesis's Results chapter draft — write it accordingly: the 2
core tables (quality, system metrics), the decision stability number, model distribution
per config, and 2-3 before/after visual examples per category (reuse Phase 6's approach).

---

## 8. Explicitly Out of Scope for Phase 7

- No GUI (Phase 9)
- No new model integration
- No threshold retuning based on results — record what the current config produces;
  tuning based on this data is legitimate _future work_ to mention, not something to do
  now and then re-report as if it were the original config

---

## 9. Definition of Done

- [ ] `run_baselines.py` implemented with resumability (Test 1 passes)
- [ ] Ground-truth reference video prepared for at least one category
- [ ] Quality metrics table produced (PSNR/SSIM/LPIPS, all 3 configs)
- [ ] System metrics table produced (time/CPU/GPU/battery/model distribution, all 3
      configs)
- [ ] Decision stability computed and reported for adaptive runs
- [ ] Sanity check (Test 3) confirms adaptive lands between the two baselines on quality
- [ ] Report written in Results-chapter-ready format
- [ ] Ablation study attempted if time allows (optional, not blocking)
