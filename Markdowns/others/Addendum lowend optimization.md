# Addendum — Low-End Device Optimization (Phase 8 preview)

Context: Phase 6 confirmed the adaptive loop works, but also confirmed the real gap —
TinySR (~150-200ms) vs Real-ESRGAN (~6s) is a huge cliff, and FP16 isn't a free win on
this GPU. On genuinely weak hardware, the engine just leans harder on TinySR, which is
correct behavior but a thin story on its own. These are concrete ways to make the
"low-end device" case more interesting, roughly ordered by effort/payoff.

---

## 1. Skip-Enhancement Tier (cheapest, do this first)

Your own design doc already specifies this as a valid Decision Engine output. Add one
more rule tier:

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

`EnhancementEngine` just returns the original frame unchanged for `model="skip"`. ~30 min
of work, and it directly demonstrates "the framework can choose to do nothing" — a strong,
easy thesis point.

## 2. Dynamic Scale Reduction

Instead of always 2x/4x, let low-budget states request scale=2 even for models that
support 4x. Cuts compute meaningfully. Requires the Decision Engine to also set `scale`
based on device budget, not just model choice — small change to existing rules.

## 3. INT8 Quantization (biggest real payoff, more effort)

Export TinySR to ONNX, quantize to INT8 via `onnxruntime` or `torch.quantization`. Typical
CPU speedup: 2-4x over FP32, with minimal quality loss on a model this small. This is the
one actually worth doing if you want a genuine "runs well on a weak CPU" claim, not just
"runs slower." Real-ESRGAN can technically be quantized too but expect more quality
degradation — lower priority.

## 4. More aggressive tiling on constrained devices

You already have tiling working for VRAM safety (Phase 5a). Make tile size configurable
per device-budget tier (smaller tiles = less peak memory, more overhead) rather than one
fixed value — cheap to wire in since the mechanism already exists.

---

## Recommendation

Do #1 now if you have 30 minutes — it's nearly free and strengthens Phase 6's story
immediately. Treat #2-4 as explicit Phase 8 scope, not Phase 6/7 blockers.
