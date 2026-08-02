# Phase 6 Spec — Complete Pipeline Integration

Status of prior phases: Phases 1–4 complete. Phase 5 complete: `tinysr` and `real_esrgan`
working and benchmarked; `basicvsr++` deliberately deferred, registry entry kept with
`"available": False`. This document specifies Phase 6: wiring everything into one
end-to-end run — real video in, adaptively enhanced video out.

---

## 0. Read This First — A Real Bug This Phase Will Surface

**Your Decision Engine's Rule 3 can select `basicvsr++`, and `basicvsr++` is unavailable.**

Rule 3 (from Phase 4): `complexity > very_high_complexity AND gpu < gpu_headroom → basicvsr++`.
You now have a real GPU with real utilization readings — this rule *can and will* fire on
genuinely complex real-world footage. When it does, `EnhancementEngine` will try to load
`basicvsr_backend.py`, which raises `NotImplementedError` by design (per Phase 5b). Without
a fix, your very first full-video run will crash the moment it hits one sufficiently
complex frame with the GPU idle enough to trigger Rule 3.

**This must be fixed before any end-to-end run, not discovered by it crashing.** Two ways
to fix it — pick one, both are legitimate:

**Option A (recommended): Decision Engine checks availability.** Your original design doc
already lists "Model Registry" as one of the Decision Engine's inputs (see
`docs/decision_engine_design`, Inputs section) — this phase is where that finally gets
wired in for real. Modify `DecisionEngine.__init__` to also load `MODEL_REGISTRY`, and
have Rule 3 check `MODEL_REGISTRY["basicvsr++"]["available"]` before selecting it:

```python
from src.modules.model_registry import MODEL_REGISTRY

# inside decide():
if scene.complexity > t["very_high_complexity"] and \
   device.gpu is not None and device.gpu < t["gpu_headroom"] and \
   MODEL_REGISTRY["basicvsr++"].get("available", True):
    return Decision(model="basicvsr++", ...)
```
This way the engine never even *decides* on an unavailable model — cleaner, and it's the
architecturally "correct" fix since your design doc always intended the registry to
inform decisions.

**Option B: EnhancementEngine catches it and falls back.** Reuse the exact same
`_fallback_single_frame` mechanism built in Phase 5b for insufficient sequence context —
wrap the dispatch call in a try/except for `NotImplementedError` and route to
`real_esrgan`, logging the substitution the same way.

**Pick Option A.** It's more correct (the engine shouldn't recommend something that can't
execute) and it's less code (no new exception handling path, reuses existing `is not None`
guard patterns from Phase 4). Whichever you pick, this must be validated explicitly in
this phase's test suite — see Test 2 below — not assumed to work because it wasn't hit
during a short test run.

---

## 1. Pipeline Orchestration — `main.py`

### Processing model: sequential, single-threaded for v1

Do **not** attempt to parallelize frame processing in this phase (e.g. processing multiple
frames concurrently, or running Device Monitor's existing background thread alongside a
multi-threaded enhancement stage). Two independent reasons:

1. Your original roadmap explicitly places "Parallel frame processing / async execution"
   under **Phase 8 (Optimization)**, not Phase 6. Pulling it forward now adds concurrency
   bugs to a phase whose job is proving the *logic* is correct end-to-end.
2. `real_esrgan` at ~9s/frame on your GPU (per Phase 5a's measured numbers) means even a
   short clip will take real wall-clock time regardless of threading — optimizing that is
   explicitly deferred, not a Phase 6 concern.

`DeviceMonitor` keeps running on its own background thread exactly as built in Phase 2 —
that doesn't change. What's sequential is the *frame processing loop itself*: extract →
analyze → decide → enhance → buffer → write, one frame fully completing before the next
starts.

### High-level flow

```python
def run_pipeline(input_path, output_path, config_path="configs/decision_config.yaml"):
    loader = VideoLoader(input_path)
    metadata = loader.load()

    device_monitor = DeviceMonitor(poll_interval=0.5)
    device_monitor.start()

    decision_engine = DecisionEngine(config_path)
    enhancement_engine = EnhancementEngine(device=get_inference_device())
    frame_buffer = FrameBuffer()
    logger = PipelineLogger(log_path="logs/run_<timestamp>.csv")

    prev_frame = None
    for frame_no, frame in enumerate(FrameExtractor(loader).frames()):
        t_start = time.time()

        scene_metrics = analyze_frame(frame, prev_frame)
        complexity = estimate_complexity(scene_metrics)
        scene_descriptor = SceneDescriptor(**scene_metrics, complexity=complexity)

        device_state = device_monitor.get_state()

        decision = decision_engine.decide(device_state, scene_descriptor)

        t_infer_start = time.time()
        enhanced_frame = enhancement_engine.enhance(frame, decision)
        inference_ms = (time.time() - t_infer_start) * 1000

        frame_buffer.put(frame_no, enhanced_frame)

        logger.log_row(frame_no=frame_no, timestamp=t_start, decision=decision,
                        scene=scene_descriptor, device=device_state,
                        inference_ms=inference_ms)

        prev_frame = frame

    device_monitor.stop()

    encoder = VideoEncoder(output_path, metadata)
    encoder.write_frames(frame_buffer.ordered_frames())
    encoder.finalize()
```

This is a skeleton, not literal final code — adapt to your actual Phase 1–5 module
interfaces, but preserve the **ordering of operations** and the **explicit logging of every
value per frame**, since that log is your Phase 7 evaluation data source.

---

## 2. `frame_buffer.py` — kept intentionally simple for v1

Per the design doc, the Frame Buffer's job is: maintain ordering, prevent frame loss,
synchronize timestamps, prepare for reconstruction. Since Phase 6 is strictly sequential
(no concurrent frame processing), you do **not** need real synchronization primitives yet —
a buffer that appends in order and hands back an ordered list is sufficient and correct:

```python
class FrameBuffer:
    def __init__(self):
        self._frames = {}  # frame_no -> enhanced frame

    def put(self, frame_no: int, frame):
        self._frames[frame_no] = frame

    def ordered_frames(self):
        for i in sorted(self._frames.keys()):
            yield self._frames[i]
```

Don't over-build this now. The reason it's a dict keyed by frame number (not just a list
you append to) rather than trusting insertion order: it makes the module already correct
if Phase 8 later introduces out-of-order concurrent completion (frame 5 finishing before
frame 3 because `real_esrgan` and `tinysr` have wildly different latencies) — you get that
correctness property for free now without adding threading complexity now.

---

## 3. `PipelineLogger` — wiring up Module 12 from your design doc

This is the first phase where the "Logging System" module from your original design
actually gets built for real (Phases 1–5's validation reports used ad hoc logging for
their own tests; this is the production per-frame logger the whole framework depends on
for Phase 7).

Required columns, matching your design doc's specified log schema exactly:

```
frame_no, timestamp, selected_model, complexity_score, cpu, gpu, ram, battery,
temperature, inference_time_ms, decision_reason
```

```python
import csv

class PipelineLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self._file = open(log_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["frame_no", "timestamp", "selected_model",
                                "complexity_score", "cpu", "gpu", "ram", "battery",
                                "temperature", "inference_time_ms", "decision_reason"])

    def log_row(self, frame_no, timestamp, decision, scene, device, inference_ms):
        self._writer.writerow([
            frame_no, timestamp, decision.model, scene.complexity,
            device.cpu, device.gpu, device.ram, device.battery, device.temperature,
            inference_ms, decision.reason,
        ])
        self._file.flush()  # flush per row — you want partial logs if a long run crashes mid-video

    def close(self):
        self._file.close()
```

`None` values (battery/gpu/temperature on some readings) will just write as empty CSV
cells — that's fine and expected, don't special-case them here. Flushing every row costs
negligible time relative to `real_esrgan`'s ~9s/frame, and means a crash partway through a
long video doesn't lose all your logging data — worth keeping even though it's not the
most "efficient" way to write a CSV.

---

## 4. Encoder integration — don't drop audio

Your Phase 1 validation didn't mention audio explicitly — worth confirming now, since a
"complete pipeline" that silently drops the audio track is a real, easy-to-miss regression
a reviewer would notice immediately when playing back your demo video. If your `encoder.py`
from Phase 1 doesn't already re-mux the original audio track, add it now via FFmpeg's
`-map 0:a` (copy audio stream from input, don't re-encode it) alongside the new
video stream. Confirm explicitly in this phase's validation that the output video's
duration and audio presence match the input.

---

## 5. Practical Runtime Budgeting for Your Test Video

Real-ESRGAN measured at ~9s/frame on your GPU (Phase 5a). A 90-frame video at 30fps
(your Phase 1 test video) would take **~13+ minutes** if every frame routed to
`real_esrgan`, and worse if any hit the Rule-3-fallback path. For Phase 6's validation,
**use a short clip** — 3–5 seconds (90–150 frames) is fine for Phase 1–5, but for this
phase's first full run, consider trimming further (1–2 seconds, 30–60 frames) purely to
keep iteration cycles fast while you're still debugging pipeline wiring. Scale back up to
longer clips once the short run is clean — don't debug pipeline logic against a
13-minute run where a bug 200 frames in costs you a long wait to rediscover.

---

## 6. Validation / Milestone

### Test 1 — End-to-end run, no crash
Full short test video in → enhanced video out. No exceptions. Output video opens and
plays. Frame count, fps, and duration match input (same check as Phase 1, now with real
enhancement in the loop instead of passthrough).

### Test 2 — Unavailable-model guard (critical, per section 0)
Construct a synthetic scenario (mocked `DeviceState`/`SceneDescriptor`, same style as
Phase 4's tests) that would have triggered Rule 3 pre-fix — very high complexity, low GPU
utilization. Confirm the Decision Engine now returns `real_esrgan` (or your chosen
fallback), **not** `basicvsr++`, and that this is visible in the log's
`decision_reason` column. This is a required regression test, not optional — write it as
an actual pytest case, not just an informal check.

### Test 3 — Model switching evidence
Run against a test video with genuinely mixed content (some flat/simple frames, some
complex ones — doesn't need to be your final evaluation dataset, just enough to exercise
both `tinysr` and `real_esrgan`). Open the resulting log CSV and confirm **both** models
appear in the `selected_model` column across the run — if every frame picked the same
model, either your test video lacks complexity variation, or something's wrong with the
Decision Engine wiring. Report the actual distribution (e.g. "62% tinysr, 38%
real_esrgan") — this is your first real instance of the "Model Selection Distribution"
metric your evaluation chapter already specifies.

### Test 4 — Frame ordering integrity
Confirm the output video's frames are in the correct order despite frames going through
different models with very different processing latencies. Simplest check: if your test
video has a visible frame counter or timestamp burned into each frame (worth adding to a
synthetic test clip specifically for this), verify it's monotonically increasing in the
output with no gaps or repeats.

### Test 5 — Log completeness
Open the CSV after a run and confirm: one row per frame (row count == frame count), no
missing columns, `None` fields render as empty rather than crashing the writer, and
`inference_time_ms` values are broadly consistent with your Phase 5a benchmarks (sanity
check, not exact match — e.g. `real_esrgan` rows should be in the seconds range, `tinysr`
rows in the tens-of-milliseconds range, not the reverse).

### Report format
Same style as prior phases. This report is a good candidate to include a couple of
before/after frame image comparisons (flat frame enhanced by tinysr, complex frame
enhanced by real_esrgan) — first time in the project you'll have visual evidence of the
adaptive framework actually doing its job end-to-end, worth capturing for the thesis
regardless of what the eval chapter needs later.

---

## 7. Explicitly Out of Scope for Phase 6

- No parallel/async frame processing (Phase 8)
- No PSNR/SSIM/LPIPS quality metrics (Phase 7)
- No comparison against static-model baselines (Phase 7)
- No decision hysteresis/stability smoothing (flagged as open question back in Phase 4 —
  this phase's Test 3/log data is exactly what will tell you whether it's actually needed)
- No GUI (Phase 9)

---

## 8. Definition of Done

- [ ] Section 0's unavailable-model bug fixed (Option A) and covered by Test 2
- [ ] `main.py` wires Loader → Extractor → Scene Analyzer → Device Monitor → Decision
      Engine → Enhancement Engine → Frame Buffer → Encoder, sequentially
- [ ] `frame_buffer.py` implemented (simple ordered-dict version, per section 2)
- [ ] `PipelineLogger` implemented with the exact column schema in section 3
- [ ] Audio passthrough confirmed working in the encoder
- [ ] All 5 validation tests run and reported, including actual model-selection
      distribution numbers
- [ ] At least one before/after visual comparison captured for the report