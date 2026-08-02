# Phase 8 Spec — Optimization

Status: Phases 1–7 complete. Core adaptive framework works and is benchmarked. This phase
is explicitly **time-permitting** — every item here is prioritized by effort/payoff, and
none of them are required for the project to be "done." Do as many as time allows, in the
order given, and stop whenever you need to move to documentation/writeup.

---

## 0. What Phase 7 Already Told You — Don't Re-litigate Settled Questions

Your Phase 7 report showed **low decision-switch rates** (3.3% Mixed, 1.7% Futbol) with
no flip-flopping observed. That means **hysteresis (smoothing to prevent rapid model
switching), flagged as an open question back in Phase 4, is not currently a demonstrated
problem** — you have actual evidence it isn't needed at this test scale. Don't spend Phase
8 time building hysteresis speculatively; if you want to strengthen this finding, that's a
Phase 10 (longer-clip evaluation) task, not an optimization task. This section exists so
you don't re-open a question your own data already answered.

---

## 1. Skip-Enhancement Tier (do this first — cheapest, ~30 min)

Already speced in the addendum. One more Decision Engine rule tier, one more
`EnhancementEngine` branch that returns the original frame unchanged:

```yaml
# decision_config.yaml
skip_enhancement:
  battery: 0.10
  temperature: 0.85
  complexity: 0.15
```

```python
if device.battery is not None and device.battery < t["skip_enhancement"]["battery"] and \
   scene.complexity < t["skip_enhancement"]["complexity"]:
    return Decision(model="skip", scale=1, reason="critical battery + trivial frame, passthrough")
```

`EnhancementEngine.enhance()` needs a trivial early return for `decision.model == "skip"`:
just return the input frame as-is, no inference call, no model load. Add `"skip"` to your
registry too (metadata: near-zero latency/memory) so logging and distribution reporting
treat it consistently with the other models rather than as a special case scattered
through the code.

**Validation**: construct a synthetic `DeviceState`/`SceneDescriptor` pair that satisfies
the skip thresholds (same style as Phase 4's test matrix), confirm `Decision.model ==
"skip"` and confirm the output frame is byte-identical to the input (skip must not
silently alter the frame).

---

## 2. Dynamic Scale Reduction (~1-2 hours)

Currently every `Decision` hardcodes `scale=2` regardless of device budget. Let
constrained states request a lower scale even from models that support higher ones —
directly cuts compute for models like Real-ESRGAN where scale affects output resolution
(and therefore tile count/inference cost) directly.

Add a budget check before setting scale in each rule branch:
```python
def _select_scale(device: DeviceState, max_scale: int, t: dict) -> int:
    if device.battery is not None and device.battery < t["scale_reduction_battery"]:
        return min(2, max_scale)
    return max_scale
```
Call this in each Decision Engine branch instead of hardcoding `scale=2`. Keep the logic
in one helper function, don't duplicate the threshold check across every rule — this is
exactly the kind of thing that drifts out of sync if copy-pasted five times.

**Validation**: same synthetic-state pattern — low battery + a model that supports scale 4
should still return `scale=2`; healthy battery should return the model's max supported
scale. Also re-run one Phase 7 category (Complex is fine) with this active and confirm
latency drops without a quality collapse — quick before/after PSNR check, not a full
re-run of the whole benchmark matrix.

---

## 3. INT8 Quantization for TinySR (~half a day, biggest real payoff)

This is the one actually worth the effort if you have time for only one "real"
optimization item — it's also the strongest answer to "but does this actually help
low-end devices," which is a fair question your project should be able to answer
concretely, not just in theory.

```python
# Export
import torch
model = load_fsrcnn_model()
model.eval()
torch.onnx.export(model, dummy_input, "models/tinysr/fsrcnn_x2.onnx",
                   opset_version=13, input_names=["input"], output_names=["output"])
```
```python
# Quantize (dynamic quantization is simplest for CPU inference)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("models/tinysr/fsrcnn_x2.onnx",
                  "models/tinysr/fsrcnn_x2_int8.onnx",
                  weight_type=QuantType.QInt8)
```
New backend variant, `fsrcnn_backend_int8.py`, using `onnxruntime.InferenceSession`
instead of PyTorch for inference — register as a distinct model in `MODEL_REGISTRY`
(e.g. `"tinysr_int8"`) rather than silently replacing `tinysr`, so you can benchmark old
vs new side by side instead of losing your Phase 5/7 baseline comparison point.

**Validation**: 
- Re-run Phase 5a's Test 1 (smoke test) and Test 2 (caching) equivalents against the
  quantized model on CPU specifically — the whole point is CPU speedup, so benchmark CPU,
  not GPU, here.
- Report: latency before/after quantization (expect 2-4x on CPU per the addendum), and a
  quality delta (PSNR/SSIM against the FP32 TinySR output, not against ground truth — you
  want to know how much quantization itself cost you, isolated from the model's inherent
  quality ceiling).
- If quality drop is negligible (~subjective + a small PSNR delta, no visible artifacts),
  this is a strong result: meaningfully faster on CPU-only hardware for basically free.

---

## 4. Adaptive Tiling (~1 hour, only if time remains)

You already have Real-ESRGAN tiling working (Phase 5a, `tile=400`, then found `tile=0`
faster on your specific card in the follow-up fix). Make tile size a device-budget-aware
parameter instead of one fixed value, so a more constrained state uses smaller tiles
(lower peak memory, more overhead) and a healthier state can use `tile=0` (fastest, per
your own measured finding) or a larger tile size:

```python
def _select_tile_size(device: DeviceState, t: dict) -> int:
    if device.gpu is None:
        return 400  # conservative default, no live GPU telemetry to base a decision on
    if device.gpu < t["tile_size_healthy_gpu_threshold"]:
        return 0  # your Phase 6 finding: no tiling was fastest at low GPU load
    return 400  # more conservative under higher GPU load
```

Pass this through to `realesrgan_backend.py`'s `RealESRGANer` init instead of the fixed
value from 5a. Low priority relative to items 1-3 — only do this if you have spare time,
since your Phase 6 finding already suggests `tile=0` may just be better across the board
on this specific card, in which case this rule may rarely diverge from a fixed choice.

**Validation**: confirm no VRAM regression (re-run Phase 5a's VRAM stress test at
whichever tile size the rule selects under a simulated high-GPU-load state) — this is the
one item where getting it wrong has a real failure mode (OOM), so don't skip validating it
even though the feature itself is low priority.

---

## 5. What NOT to Do in Phase 8

- Don't build hysteresis (section 0)
- Don't parallelize frame processing — still explicitly out of scope, this was never
  actually promoted into Phase 8's plan, it was only ever mentioned as a *possible* future
  item in your original roadmap. Skip it; it's real engineering effort for a benefit
  you haven't demonstrated you need (Phase 7's total latency numbers weren't
  parallelization-bottlenecked, they were inference-cost-bottlenecked)
- Don't quantize Real-ESRGAN — the addendum already flagged this as lower priority due to
  expected quality degradation on a GAN-based model; not worth the effort given items 1-3
  exist

---

## 6. Validation / Milestone Summary

Each item above has its own inline validation. No new full benchmark matrix re-run is
required — Phase 8 is about targeted, isolated improvements with before/after evidence per
item, not a repeat of Phase 7's full experimental sweep. If you do items 1-3, a short
final report should show:

| Optimization | Before | After | Delta |
|---|---|---|---|
| Skip tier (synthetic critical-state test) | N/A (no skip logic) | frame passthrough confirmed | — |
| Dynamic scale (Complex category re-run) | latency X, PSNR Y | latency X', PSNR Y' | |
| INT8 TinySR (CPU) | latency X ms | latency X' ms | N% faster |
| INT8 TinySR quality | — | PSNR delta vs FP32 | |

---

## 7. Definition of Done

- [ ] Skip-enhancement tier implemented and validated (item 1)
- [ ] Dynamic scale reduction implemented and validated (item 2) — optional if time-constrained
- [ ] INT8 TinySR quantization implemented, benchmarked on CPU, quality delta reported
      (item 3) — do this one if you only have time for one item beyond skip-tier
- [ ] Adaptive tiling (item 4) — optional, lowest priority
- [ ] Short before/after report per completed item (section 6 table)
- [ ] No regression in existing Phase 5-7 tests caused by these changes (quick smoke re-run,
      not full re-benchmark)