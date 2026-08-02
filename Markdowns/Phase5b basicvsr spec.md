# Phase 5b Spec — BasicVSR++ (Temporal / Sequence-Based) Integration

Status of prior phases: Phases 1–4 complete. Phase 5a complete — `tinysr` (FSRCNN) and
`real_esrgan` are working, registry-driven, cached, with measured latency/VRAM numbers on
your GTX 1650 (4GB). This document specifies Phase 5b: adding `basicvsr++` as the third
registered model.

**This is the highest-risk single item in the whole project**, flagged as such since the
original roadmap. Read this whole spec before starting — the risk isn't "will the code
run," it's "will the dependency stack even install cleanly, and will 4GB VRAM be enough."
Budget accordingly, and treat the fallback plan in section 6 as a real option, not a last resort.

---

## 1. Why This Is Different From 5a

TinySR and Real-ESRGAN both take **one frame in, one frame out** — that's why
`requires_sequence: False` was set for both in the registry. BasicVSR++ is a genuinely
different kind of model: it takes a **sequence of consecutive frames** as input and uses
temporal propagation (information flowing forward/backward across the sequence) to
reconstruct each frame with better consistency than any single-frame model could achieve.
This has three concrete consequences for your architecture:

1. **`enhancement_engine.py` needs a second code path.** A model with
   `requires_sequence: True` can't be called with a single `frame_bgr` argument — it needs
   a window of frames (e.g. the current frame plus N before/after). This is why the
   registry schema already reserved this field in 5a.
2. **The Frame Buffer (Phase 6 territory, but touched here) has to provide that window.**
   For 5b's *isolated* validation, you don't need the real buffer yet — just a small
   sequence of test frames you manually assemble — but the interface contract needs to be
   decided now so Phase 6 doesn't require reworking this module.
3. **Memory scales with sequence length, not just resolution.** This is the actual VRAM
   risk. A 5-frame window at 480p→1080p output is a materially bigger memory footprint
   than one Real-ESRGAN frame at the same output resolution.

---

## 2. Dependency Reality Check — Do This Before Writing Any Integration Code

BasicVSR++'s reference weights are distributed via OpenMMLab's `mmagic` (formerly
`mmediting`), which brings in `mmcv`, `mmengine`, and its own config system. This is a much
heavier, more opinionated dependency stack than `realesrgan`'s standalone pip package, and
version compatibility with modern PyTorch/CUDA is a known pain point community-wide (you
already hit two unrelated compatibility shims in 5a with a comparatively lighter package —
expect more here, not less).

**Before writing `basicvsr_backend.py`, do a standalone install test in an isolated venv**
(not your main project env — you don't want a failed mmcv build corrupting an environment
that already has 5a working):

```bash
python -m venv test_mmagic_env
# activate it
pip install openmim
mim install mmcv
mim install mmagic
```

Run this and record: did it complete without error? What versions of `mmcv`/`torch`/`cuda`
did it resolve to, and are they compatible with your existing `torch` install from 5a?
`mmcv` in particular ships CUDA-version-specific prebuilt wheels — if the wheel it wants
doesn't match your installed CUDA toolkit version, you're looking at a source build, which
on Windows is its own multi-hour rabbit hole.

**If this standalone test fails or drags on past ~2 hours of dependency wrangling, stop
and move to the fallback plan in section 6 instead of sinking further time in.** This is a
final-year project with a deadline, not an open-ended engineering exercise — a documented,
honest "we attempted BasicVSR++ integration, hit dependency conflict X, and made this
informed decision instead" is a perfectly legitimate thing to write in a thesis, and is a
much better use of your remaining time than fighting a CUDA wheel mismatch for a week.

---

## 3. Registry Update

```python
MODEL_REGISTRY = {
    # ... tinysr, real_esrgan unchanged from 5a ...
    "basicvsr++": {
        "display_name": "BasicVSR++",
        "loader": "src.modules.backends.basicvsr_backend.load_model",
        "infer_fn": "src.modules.backends.basicvsr_backend.infer_sequence",
        "expected_memory_mb": None,   # fill in after Test 1 below — do not guess
        "expected_latency_ms_cpu": None,
        "expected_latency_ms_gpu": None,
        "supported_scales": [4],       # BasicVSR++ is typically trained for a fixed scale — confirm against the checkpoint you use
        "quality_rating": "very_high",
        "requires_sequence": True,
        "sequence_window": 5,          # frames before+after current frame; tune down if VRAM-constrained
    },
}
```

Note the changed function name: `infer_sequence`, not `infer` — this is deliberate, not a
typo, to make the signature difference visible at the call site rather than hidden behind
an identical name with a different argument shape.

---

## 4. `enhancement_engine.py` — sequence dispatch path

```python
def enhance(self, frame_bgr, decision: Decision, frame_window: list | None = None):
    entry = MODEL_REGISTRY[decision.model]
    infer_fn = self._resolve(decision.model)
    if entry["requires_sequence"]:
        if frame_window is None or len(frame_window) < entry["sequence_window"]:
            # Not enough context frames available yet (e.g. near start/end of video).
            # Do NOT silently crash or silently process garbage — fall back explicitly.
            return self._fallback_single_frame(frame_bgr, decision)
        return infer_fn(frame_window, self.device, scale=decision.scale)
    else:
        return infer_fn(frame_bgr, self.device, scale=decision.scale)

def _fallback_single_frame(self, frame_bgr, decision):
    # Explicit, logged fallback — e.g. route to real_esrgan for this one frame.
    # This must be visible in your Phase 6/7 decision logs, not a silent substitution.
    fallback_decision = Decision(model="real_esrgan", scale=decision.scale,
                                  reason=f"insufficient sequence context for {decision.model}, fell back")
    return self.enhance(frame_bgr, fallback_decision)
```

This fallback matters more than it looks: **the first and last few frames of every video
will never have a full 5-frame window** (no frames exist before frame 0, or after the last
frame). Without an explicit fallback, either the pipeline crashes at the start of every
video, or someone "fixes" it by padding with duplicate/black frames, which quietly
degrades BasicVSR++'s actual output quality at exactly the frames you'd notice most. Log
this fallback explicitly — it's a real, expected behavior, not a bug, and your evaluation
chapter should be able to report how often it triggered.

---

## 5. `src/modules/backends/basicvsr_backend.py`

```python
_model_cache = None

def load_model(device: str):
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    # Load via mmagic's inference API or a lighter standalone BasicVSR++ implementation
    # (search for standalone PyTorch reimplementations if mmagic proves too heavy —
    # several exist with pretrained weights and no mmcv dependency; worth checking
    # BEFORE committing to the full mmagic stack, given section 2's warning)
    return _model_cache

def infer_sequence(frame_window: list, device: str, scale: int = 4):
    """
    frame_window: list of BGR frames, ordered, length == registry's sequence_window
    Returns: single enhanced BGR frame corresponding to the CENTER frame of the window
    """
    model = load_model(device)
    # stack frames into (1, T, C, H, W) tensor per BasicVSR++'s expected input shape
    # with torch.no_grad(): output_sequence = model(stacked_tensor)
    # extract and return only the center frame's output — that's the frame this call
    # was invoked for; the rest of the sequence was context, not additional output
    return enhanced_center_frame_bgr
```

**Important scoping note buried in the docstring above, make it explicit in your own
notes too**: `infer_sequence` takes a window and returns **one** enhanced frame (the center
of that window), not five. Don't build this as "returns 5 enhanced frames" — that would
require re-architecting how the Frame Buffer hands off output in Phase 6, and it's not how
BasicVSR++ is normally deployed for this use case anyway (sliding window, one output per
call, window slides forward one frame at a time).

**Alternative if `mmagic` proves too painful**: search for a standalone BasicVSR++
PyTorch reimplementation with pretrained weights before fully committing (several exist on
GitHub without the full OpenMMLab dependency chain). This is worth 30 minutes of searching
*before* starting the mmagic install test in section 2 — not after you've already spent
hours on it.

---

## 6. Fallback Plan — If Integration Doesn't Work Out

If section 2's dependency check fails, or VRAM proves insufficient even at a reduced
`sequence_window` (try 3 before giving up entirely — a 3-frame window costs meaningfully
less than 5), the legitimate fallback is:

**Ship the framework with 2 models (TinySR + Real-ESRGAN) as the working implementation**,
and present BasicVSR++ as a **documented, partially-attempted extension** in your thesis:
what you tried, what broke, why, and what a hypothetical Phase 5b-complete system would
look like architecturally (your registry schema and `requires_sequence`/sequence-dispatch
code already exist and are real evidence of designing for it, even if the specific model
never got wired in). This is a completely normal, defensible outcome for an
undergraduate project attempting to integrate a heavy third-party research codebase —
examiners generally respond far better to "we identified this risk early, attempted it,
hit X, and made an informed call" than to a project that silently never mentions it tried.

Do not treat this fallback as failure. Decide on it explicitly and document the decision
the same way you've documented every other design choice in this project.

---

## 7. Validation / Milestone

### Test 1 — Standalone dependency install (section 2)
Report pass/fail and however far you got. This alone is worth documenting even if it fails.

### Test 2 — Single sequence smoke test
Feed a hand-assembled 5-frame window (can be 5 consecutive frames pulled manually from any
test video) through `infer_sequence`. Confirm output shape is correct and the image isn't
garbage — same visual sanity check as 5a's Test 1.

### Test 3 — VRAM stress test at reduced window
Run at `sequence_window=5` first; if OOM, retry at 3. Report peak VRAM either way — this
is the number that determines whether 5b ships at all. Compare directly against 5a's
Real-ESRGAN VRAM number (2653.55 MB, no tiling) for context in your report.

### Test 4 — Boundary/fallback behavior
Call `EnhancementEngine.enhance()` with `frame_window=None` and with a window shorter than
`sequence_window`. Confirm it falls back to `real_esrgan` (per section 4) rather than
crashing, and confirm the fallback is logged with a clear reason string.

### Report format
Same style as prior phases — but if section 2 or 3 fails, the report's job shifts from
"here are the passing test results" to "here is the documented, reasoned decision to defer
BasicVSR++," per section 6. Either outcome is a complete, valid Phase 5b report.

---

## 8. Definition of Done (either path)

**If integration succeeds:**
- [ ] Dependency install path documented (mmagic or standalone alt), pinned in `requirements.txt`
- [ ] `basicvsr_backend.py` implemented with caching, sequence input, center-frame output
- [ ] Registry updated with real (not placeholder) latency/VRAM numbers
- [ ] `enhancement_engine.py`'s sequence dispatch + explicit fallback implemented and tested
- [ ] All 4 validation tests reported

**If integration is deferred (fallback plan):**
- [ ] Section 2's dependency attempt documented with specific failure point
- [ ] VRAM ceiling documented if that was the blocker, with the 3-frame retry attempted first
- [ ] Registry entry kept in place (commented or `available: False`) as evidence of designed-for extensibility
- [ ] One paragraph written for the thesis explaining the decision, matching the framing in section 6