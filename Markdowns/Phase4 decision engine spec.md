# Phase 4 Spec — Decision Engine

Status of prior phases: Phase 1 (video passthrough pipeline), Phase 2 (`DeviceMonitor`),
and Phase 3 (`scene_analyzer` + `complexity_estimator`) are complete and validated. This
document specifies Phase 4.

---

## Objective

Build `decision_engine.py`: the module that consumes `DeviceState` (Phase 2) and
`SceneDescriptor` (Phase 3) and outputs a `Decision` — which model to run, at what scale,
and why. This is a **rule-based** engine for v1 (per your original design decision —
interpretable, debuggable, easy to evaluate; ML-based decision policies are explicitly
future work, not in scope here).

At this stage, no actual Super-Resolution model exists yet (that's Phase 5). The engine
outputs a **model name string** — validation is entirely about whether the _decision logic_
is correct, not about running inference.

---

## 1. `state_types.py` — confirm/add this dataclass

```python
from dataclasses import dataclass

@dataclass
class Decision:
    model: str            # e.g. "tinysr", "real_esrgan", "basicvsr++"
    scale: int             # upscale factor, e.g. 2
    reason: str             # human-readable justification, for logging
    priority: str = "medium"  # "low" | "medium" | "high" — reserved for future scheduling use
```

---

## 2. Handling `None` fields — the most important part of this phase

`DeviceState` fields (`gpu`, `battery`, `charging`, `temperature`) can legitimately be
`None` (desktop with no battery, no NVML driver, no thermal sensor — all confirmed real
conditions from your Phase 2 validation). **Every rule that references one of these fields
must explicitly skip itself, not silently treat `None` as 0 or as "condition met."**

This is the single most likely place for a subtle bug: in Python, `None < 0.20` raises a
`TypeError`, but if someone "fixes" that with `(battery or 0) < 0.20`, that silently means
"no battery info" gets treated as "critically low battery" — which is wrong and would
skew every desktop-run experiment toward always picking the lightweight model. Guard
explicitly:

```python
if device.battery is not None and device.battery < threshold:
    ...
```

not

```python
if (device.battery or 0) < threshold:  # WRONG — None becomes 0, not "unknown"
```

---

## 3. `decision_engine.py`

```python
import yaml
from src.utils.state_types import DeviceState, SceneDescriptor, Decision

class DecisionEngine:
    def __init__(self, config_path="configs/decision_config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.thresholds = self.cfg["thresholds"]

    def decide(self, device: DeviceState, scene: SceneDescriptor) -> Decision:
        t = self.thresholds

        # Rule 1: critical device state -> always lightweight, regardless of scene
        if device.battery is not None and device.battery < t["low_battery"] and \
           device.temperature is not None and device.temperature > t["high_temp"]:
            return Decision(model="tinysr", scale=2,
                             reason=f"low battery ({device.battery:.2f}) + high temp ({device.temperature:.2f})",
                             priority="high")

        # Rule 2: low complexity -> lightweight regardless of device state
        if scene.complexity < t["low_complexity"]:
            return Decision(model="tinysr", scale=2,
                             reason=f"low complexity ({scene.complexity:.2f})")

        # Rule 3: very high complexity + GPU headroom -> heaviest model
        if scene.complexity > t["very_high_complexity"] and \
           device.gpu is not None and device.gpu < t["gpu_headroom"]:
            return Decision(model="basicvsr++", scale=2,
                             reason=f"very high complexity ({scene.complexity:.2f}) + GPU available ({device.gpu:.2f})",
                             priority="high")

        # Rule 4: high complexity + CPU headroom -> mid-tier model
        if scene.complexity > t["high_complexity"] and device.cpu < t["cpu_headroom"]:
            return Decision(model="real_esrgan", scale=2,
                             reason=f"high complexity ({scene.complexity:.2f}) + CPU headroom ({device.cpu:.2f})")

        # Rule 5: fallback — moderate complexity, no strong constraint triggered
        return Decision(model="real_esrgan", scale=2,
                         reason="default: moderate complexity, no constraint triggered")
```

### Design notes

- **Rule order matters and is intentional**: critical-device-state checks come first
  (safety/thermal concerns override quality preferences), then low-complexity short-circuit
  (don't waste compute on easy frames even if the device is idle and cool), then the
  complexity/resource tiers. Do not reorder without re-running the validation matrix below.
- **Rule 3 requires `device.gpu is not None`** — on a machine with no GPU (your current
  dev setup, per Phase 2's `GPU: N/A`), this rule can never fire, and the engine will fall
  through to Rule 4/5 instead. This is correct behavior, not a bug — but it means you
  **cannot fully validate Rule 3 on your current hardware** using real `DeviceState`
  readings. Validate it with synthetic/mocked `DeviceState` objects instead (see section 4).
- Every branch sets `reason` to something that explains _why_, referencing the actual
  values, not just a static string — this is what your Phase 7 logs will use for the
  "Decision Reason" column, and what you'll quote directly in the thesis when explaining
  specific model-selection examples.

---

## 4. `configs/decision_config.yaml`

```yaml
thresholds:
  low_battery: 0.20
  high_temp: 0.75
  low_complexity: 0.25
  high_complexity: 0.75
  very_high_complexity: 0.90
  cpu_headroom: 0.60
  gpu_headroom: 0.80
```

These are the same starting values used in the original plan — do not hand-tune yet.
Tuning requires real footage + real device load data you don't have until Phase 6/7.

---

## 5. Validation / Milestone

Unlike Phases 2–3, this phase's validation is **entirely synthetic and deterministic** —
you're testing whether the logic matches a truth table, not whether a sensor responds to
load. Write this as a `pytest` suite (`tests/test_decision_engine.py`), and summarize
results in the same markdown report format as Phases 1–3.

### Required test matrix

Construct `DeviceState`/`SceneDescriptor` pairs covering every rule branch, including the
`None`-field edge cases. At minimum:

| #   | Scenario                                   | battery | temp | gpu  | cpu  | complexity | Expected model | Expected rule fired                                                              |
| --- | ------------------------------------------ | ------- | ---- | ---- | ---- | ---------- | -------------- | -------------------------------------------------------------------------------- |
| 1   | Low battery + hot                          | 0.15    | 0.80 | None | 0.30 | 0.50       | tinysr         | Rule 1                                                                           |
| 2   | Low battery, but cool                      | 0.15    | 0.40 | None | 0.30 | 0.50       | real_esrgan    | Rule 5 (Rule 1 must NOT fire — temp not None but below threshold)                |
| 3   | Battery unknown (desktop), hot             | None    | 0.80 | None | 0.30 | 0.50       | real_esrgan    | Rule 5 (Rule 1 must NOT fire — battery is None, not "low")                       |
| 4   | Flat scene, powerful device                | 0.90    | 0.30 | 0.10 | 0.10 | 0.10       | tinysr         | Rule 2 (complexity short-circuits regardless of device power)                    |
| 5   | Extreme complexity, GPU free               | 0.90    | 0.30 | 0.10 | 0.10 | 0.95       | basicvsr++     | Rule 3                                                                           |
| 6   | Extreme complexity, no GPU present         | 0.90    | 0.30 | None | 0.10 | 0.95       | real_esrgan    | Rule 4 (Rule 3 must NOT fire — gpu is None, falls through)                       |
| 7   | Extreme complexity, GPU present but busy   | 0.90    | 0.30 | 0.90 | 0.10 | 0.95       | real_esrgan    | Rule 4 (Rule 3 skipped — gpu.90 exceeds gpu_headroom.80)                         |
| 8   | High complexity, CPU busy                  | 0.90    | 0.30 | None | 0.90 | 0.80       | real_esrgan    | Rule 5 (Rule 4 must NOT fire — cpu.90 exceeds cpu_headroom.60; falls to default) |
| 9   | Mid complexity, all fields None except cpu | None    | None | None | 0.40 | 0.50       | real_esrgan    | Rule 5 (confirms engine never crashes when every optional field is None)         |

Case 9 is the most important robustness check: confirm the engine runs to completion
with **every** optional `DeviceState` field set to `None` simultaneously — this is the
realistic condition for your current no-GPU, and potentially no-thermal-sensor, dev
machine, and it must never raise an exception.

### Also required: exact reason-string sanity check

For at least 2–3 of the above cases, assert that `decision.reason` contains the values
you'd expect (e.g. case 1's reason string should mention both the battery and temp
values). This isn't about exact string matching everywhere — it's about confirming the
reason field is actually populated with real numbers, not a placeholder, since this is
what your evaluation chapter's "Decision Reason" log column depends on.

### Report format

Same style as Phases 1–3: a markdown doc with the test matrix above filled in with
actual pass/fail results, plus a short "Verification Analysis" section covering:

- Confirm all 9 cases (or however many you implement) pass
- Explicitly state which rules could _not_ be exercised with real Phase 2 hardware
  readings on your current machine (almost certainly Rule 3, since you have no GPU) and
  confirm they were instead validated via synthetic/mocked `DeviceState` objects
- Confirm the all-`None` robustness case (case 9) does not raise

---

## 6. Explicitly Out of Scope for Phase 4

- No actual model loading or inference — `Decision.model` is just a string until Phase 5
- No ROI-only enhancement, frame skipping, or dynamic upscale factor logic — these are
  documented future extensions, not v1 requirements
- No threshold auto-tuning — thresholds are fixed config values for this phase
- No decision _history_ or stability tracking across frames (e.g. hysteresis to prevent
  rapid model-switching) — this is worth flagging as a known gap for Phase 6/7, since
  once this engine runs frame-by-frame on real video, you may see the model selection
  flicker between calls if complexity hovers near a threshold. Don't build hysteresis now;
  just note it as a likely follow-up once you see real per-frame decision logs.

---

## 7. Definition of Done

- [ ] `Decision` dataclass added to `state_types.py`
- [ ] `DecisionEngine.decide()` implemented exactly per the rule order above, with
      explicit `is not None` guards on every optional field
- [ ] `configs/decision_config.yaml` created with the threshold values above
- [ ] `tests/test_decision_engine.py` covering all 9 matrix cases (or a documented
      equivalent set), all passing
- [ ] Validation report written, including which rules were validated via real hardware
      vs. synthetic/mocked state, and confirming the all-`None` robustness case
- [ ] Note on decision-stability/hysteresis flagged as a known open question for Phase 6/7,
      not solved here
