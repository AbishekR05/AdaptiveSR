# Phase 5a Spec — Model Registry + Lightweight & Mid-Tier Model Integration

Status of prior phases: Phase 1 (video passthrough), Phase 2 (`DeviceMonitor`), Phase 3
(`scene_analyzer` + `complexity_estimator`), and Phase 4 (`DecisionEngine`) are complete
and validated. This document specifies Phase 5a.

**Scope note**: Phase 5 is split into 5a (this doc) and 5b (BasicVSR++, separate spec).
Reason: your dev GPU (GTX 1650, 4GB VRAM) comfortably handles single-frame models but is
tight for BasicVSR++'s multi-frame temporal processing. Shipping 5a first gets you a full
working end-to-end adaptive pipeline (TinySR ↔ Real-ESRGAN switching) without a VRAM
problem blocking that milestone. BasicVSR++ is layered in afterward as 5b.

---

## Objective

Build:

1. `model_registry.py` — central metadata store for available SR models (no inference logic)
2. `enhancement_engine.py` — loads a model and runs inference on a frame, dispatched
   entirely by the `Decision` object from Phase 4 (no decision logic here)
3. Two working model backends: **FSRCNN** (lightweight, maps to `tinysr` in your Decision
   Engine's vocabulary) and **Real-ESRGAN** (mid-tier)

After this phase: an LR frame goes in, the Decision Engine's chosen model runs, an HR
frame comes out — for two of your three registered models.

---

## 1. Device dispatch — CPU/GPU selection

Now that you have a real GPU, `enhancement_engine.py` needs an explicit, testable rule for
where inference runs. Don't hardcode `.cuda()` — detect and fall back cleanly:

```python
import torch

def get_inference_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
```

Important: **this is independent of your `DeviceMonitor`'s `gpu` reading.** `DeviceMonitor.gpu`
(via `pynvml`) reports _utilization_, used by the Decision Engine to decide _which model_
to pick. `get_inference_device()` here answers a different question — _where does the
chosen model actually execute_. Don't conflate them. A `None` GPU utilization reading
(D3 sleep state, per your Phase 4 note) does not mean CUDA is unavailable for inference —
it just means telemetry was momentarily unreadable. Keep these two checks fully separate
in code and in your validation report.

---

## 2. `model_registry.py`

Pure metadata, no PyTorch imports, no model loading. This mirrors your Decision Engine's
job/data separation principle — the registry describes what's available, it doesn't do
anything.

```python
MODEL_REGISTRY = {
    "tinysr": {
        "display_name": "FSRCNN (lightweight)",
        "loader": "src.modules.backends.fsrcnn_backend.load_model",
        "infer_fn": "src.modules.backends.fsrcnn_backend.infer",
        "expected_memory_mb": 50,
        "expected_latency_ms_cpu": 15,
        "expected_latency_ms_gpu": 3,
        "supported_scales": [2, 3, 4],
        "quality_rating": "medium",
        "requires_sequence": False,
    },
    "real_esrgan": {
        "display_name": "Real-ESRGAN",
        "loader": "src.modules.backends.realesrgan_backend.load_model",
        "infer_fn": "src.modules.backends.realesrgan_backend.infer",
        "expected_memory_mb": 800,
        "expected_latency_ms_cpu": 900,
        "expected_latency_ms_gpu": 60,
        "supported_scales": [2, 4],
        "quality_rating": "high",
        "requires_sequence": False,
    },
    # "basicvsr++": added in Phase 5b
}
```

Notes:

- `requires_sequence: False` on both — flag this field now even though it's unused until
  5b. It's how `enhancement_engine.py` will know later whether to hand a single frame or
  a buffered window to a model's `infer()` — building the field into the registry now
  means 5b doesn't require touching this file's schema again.
- Latency numbers are **placeholders** — replace with your own measured values once you've
  run the Phase 5a benchmark below. Don't leave placeholder numbers in the file
  un-flagged; either measure them or comment them as unverified.
- String-path loaders (not direct function references) keep this file import-light and
  match a common "lazy registry" pattern — avoids loading PyTorch/model weights just to
  inspect metadata.

---

## 3. Backend modules

Create `src/modules/backends/` with one file per model. Each exposes exactly two functions
with a consistent signature, so `enhancement_engine.py` can call any of them identically.

### `src/modules/backends/fsrcnn_backend.py`

```python
import torch

_model_cache = None

def load_model(device: str):
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    # Load FSRCNN architecture + pretrained weights here
    # _model_cache = FSRCNN(...).to(device)
    # _model_cache.load_state_dict(torch.load("models/tinysr/fsrcnn_x2.pth", map_location=device))
    # _model_cache.eval()
    return _model_cache

def infer(frame_bgr, device: str, scale: int = 2):
    model = load_model(device)
    # preprocess: BGR->RGB, HWC->CHW, normalize to [0,1], add batch dim, .to(device)
    # with torch.no_grad(): output = model(input_tensor)
    # postprocess: back to HWC uint8 BGR
    return enhanced_frame_bgr
```

### `src/modules/backends/realesrgan_backend.py`

Same two-function shape. Use the official `realesrgan` pip package's `RealESRGANer`
wrapper rather than hand-rolling the architecture — it already handles tiling for large
frames, which matters on a 4GB card (see section 5).

**Model caching is required, not optional**: both `load_model()` functions must cache the
loaded model in a module-level variable and return the cached instance on repeat calls.
Reloading weights from disk per-frame would dominate your latency numbers and invalidate
every benchmark you run in Phase 7. This is the single most common mistake at this stage —
check for it explicitly in validation.

---

## 4. `enhancement_engine.py`

```python
import importlib
from src.modules.model_registry import MODEL_REGISTRY
from src.utils.state_types import Decision

class EnhancementEngine:
    def __init__(self, device: str):
        self.device = device
        self._resolved_cache = {}

    def _resolve(self, model_name: str):
        if model_name in self._resolved_cache:
            return self._resolved_cache[model_name]
        entry = MODEL_REGISTRY[model_name]
        module_path, fn_name = entry["infer_fn"].rsplit(".", 1)
        infer_fn = getattr(importlib.import_module(module_path), fn_name)
        self._resolved_cache[model_name] = infer_fn
        return infer_fn

    def enhance(self, frame_bgr, decision: Decision):
        infer_fn = self._resolve(decision.model)
        return infer_fn(frame_bgr, self.device, scale=decision.scale)
```

No `if model == "tinysr": ...` branching here — dispatch is purely registry-driven. This
is what makes "add a new model without changing the Decision Engine" (your original
design principle) actually true rather than aspirational.

---

## 5. VRAM safety on the 4GB card

Real-ESRGAN's official wrapper supports **tiling** (splits large frames into tiles,
processes sequentially, stitches back together) specifically for exactly this situation.
Enable it rather than risking OOM on full-frame 1080p output:

```python
from realesrgan import RealESRGANer
upsampler = RealESRGANer(scale=4, model_path=..., tile=400, tile_pad=10, half=True)
```

`half=True` (FP16) roughly halves memory further on CUDA — use it for GPU inference, but
be aware FP16 isn't supported identically on CPU fallback, so branch `half` on `device`.
Document whatever tile size you land on in the registry's metadata comments — this is a
hardware-specific tuning value, not a universal constant.

---

## 6. Validation / Milestone

This phase's validation has two parts: **correctness** (does it run, does output look
right) and **performance** (does it match your registry's latency assumptions, does
caching actually work).

### Test 1 — Independent backend smoke test

For each of the two models: LR frame in → HR frame out, no errors, output shape matches
`(H*scale, W*scale, 3)`. Save before/after images and visually confirm the output isn't
garbage (all-black, all-noise, wrong color channels — BGR/RGB mixups are the classic bug
here).

### Test 2 — Caching verification

Call `infer()` twice in a row on the same model and time each call separately. Second call
should be dramatically faster than the first (no weight-loading overhead). Report both
timings explicitly — this is the check that catches the "reloading weights per frame"
mistake mentioned above.

### Test 3 — CPU vs GPU dispatch

Run the same frame through `infer()` with `device="cpu"` and `device="cuda"` explicitly.
Confirm both succeed and report the latency difference. This also validates your
`get_inference_device()` fallback path independent of whether CUDA happens to be
available at test time.

### Test 4 — Full dispatch via EnhancementEngine + real Decision objects

Take 2–3 `Decision` objects from your Phase 4 test matrix (e.g. Case 2 → `real_esrgan`,
Case 4 → `tinysr`) and confirm `EnhancementEngine.enhance()` correctly routes to the right
backend and returns a valid enhanced frame for each — this is the first test that
exercises Phase 4 and Phase 5 wired together, even though full pipeline integration is
Phase 6.

### Test 5 — VRAM stress check (GPU only)

Run Real-ESRGAN on your largest expected input size (480p frame, scale 4 → ~1920px wide
output) with tiling enabled. Confirm no OOM. Report peak VRAM usage if you can capture it
(`torch.cuda.max_memory_allocated()`).

### Report format

Same style as Phases 1–4: markdown doc, tables for latency (measured, not placeholder,
values per model per device), pass/fail per test, and a short note on what changed for the
`configs/models.yaml` entries.

---

## 7. Explicitly Out of Scope for Phase 5a

- BasicVSR++ — Phase 5b, separate spec, once 5a is confirmed working
- No batch inference across multiple frames — single-frame calls only for now (matches
  `requires_sequence: False`)
- No pipeline integration (frame buffer, full video run) — that's Phase 6
- No quality metrics (PSNR/SSIM/LPIPS) yet — that's Phase 7's evaluation harness, this
  phase is "does it run correctly and fast," not "how good does it look numerically"

---

## 8. Definition of Done

- [ ] `model_registry.py` created with `tinysr` and `real_esrgan` entries (metadata only)
- [ ] `src/modules/backends/fsrcnn_backend.py` implemented with caching
- [ ] `src/modules/backends/realesrgan_backend.py` implemented with caching + tiling
- [ ] `enhancement_engine.py` implemented, fully registry-driven dispatch (no hardcoded
      model branching)
- [ ] `get_inference_device()` implemented, tested independently of `DeviceMonitor.gpu`
- [ ] All 5 validation tests run with real measured numbers, written into a report
- [ ] Registry's placeholder latency/memory values replaced with measured ones (or
      explicitly flagged as still-placeholder if not yet measured)
