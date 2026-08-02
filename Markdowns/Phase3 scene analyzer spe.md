# Phase 3 Spec — Scene Analyzer + Complexity Estimator

Status of prior phases: Phase 1 (video passthrough pipeline) and Phase 2 (standalone
`DeviceMonitor` thread) are complete and validated. This document specifies Phase 3.

---

## Objective

Build two modules:

1. `scene_analyzer.py` — computes raw visual metrics per frame (motion, texture, edges, blur/noise)
2. `complexity_estimator.py` — combines those metrics into a single 0–1 complexity score

These, together with `DeviceState` from Phase 2, are the two inputs the Decision Engine
(Phase 4) will consume. Getting the _signal quality_ right here matters more than in prior
phases — this is the first module whose output quality (not just plumbing correctness)
affects the thesis's core claim.

---

## 1. `state_types.py` — add/confirm this dataclass

```python
from dataclasses import dataclass

@dataclass
class SceneDescriptor:
    motion: float       # 0-1
    texture: float      # 0-1
    edges: float         # 0-1
    blur_clarity: float  # 0-1, higher = sharper/less blurry
    complexity: float    # 0-1, final combined score
```

Note: `blur_clarity` is stored as "clarity" (higher = sharper), not "blurriness", so all
four fields follow the same convention: higher value = more visual complexity/detail
contribution. Complexity Estimator will use `(1 - blur_clarity)` where blur signal should
increase complexity, or use `blur_clarity` directly where it should decrease it — see
formula in section 3.

---

## 2. `scene_analyzer.py`

### Function signature

```python
def analyze_frame(frame: np.ndarray, prev_frame: np.ndarray | None) -> dict:
    """
    frame: current BGR frame (as read by OpenCV)
    prev_frame: previous BGR frame, or None for the first frame in a video
    Returns dict with keys: motion, texture, edges, blur_clarity (all floats, 0-1)
    """
```

### Metric implementations

**Motion** (requires `prev_frame`; if `prev_frame is None`, return `0.0`):

```python
def compute_motion(frame, prev_frame) -> float:
    gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_curr, gray_prev)
    raw_score = diff.mean() / 255.0
    return min(raw_score * MOTION_SCALE_FACTOR, 1.0)  # MOTION_SCALE_FACTOR: tune empirically, start at ~4.0
```

Start with frame differencing (cheap, sufficient for v1). Do NOT implement optical flow
unless frame-diff demonstrably fails to discriminate static vs. moving scenes in the
validation step below — this is explicitly deferred complexity, not a v1 requirement.

**Edge density**:

```python
def compute_edges(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    return (edges > 0).sum() / edges.size
```

This is already naturally in [0, 1] — no extra scaling needed.

**Texture**:

```python
def compute_texture(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return min(laplacian_var / TEXTURE_SCALE_FACTOR, 1.0)  # TEXTURE_SCALE_FACTOR: tune empirically, start at ~500.0
```

**Blur clarity** (reuses the Laplacian variance computation — do not compute it twice,
pass it in or cache it within the same `analyze_frame` call):

```python
def compute_blur_clarity(laplacian_var: float) -> float:
    return min(laplacian_var / BLUR_SCALE_FACTOR, 1.0)  # BLUR_SCALE_FACTOR: tune empirically, start at ~300.0
```

Note this is intentionally similar to texture (both derive from Laplacian variance) — flag
this redundancy in your validation writeup rather than silently duplicating a metric. It's
fine to keep both as separate fields for now (matches the design doc's schema), but note
in your logs that they're correlated, and don't be surprised if the complexity formula
ends up effectively double-weighting "flat vs. detailed" as a result.

### Important implementation notes

- All `*_SCALE_FACTOR` constants are placeholders — they exist to map raw metric ranges
  into roughly [0, 1] before clamping. Do not spend time hand-tuning these before running
  the validation step in section 4; pick reasonable starting values, clamp with `min(x, 1.0)`,
  and revisit only if the validation frames don't rank correctly.
- `analyze_frame` must be a **pure function** with no hidden state — no class attributes
  that persist between calls, no global caches. The caller (pipeline loop) is responsible
  for passing `prev_frame` in. This matters for the "run the same frame twice → same score"
  stability test below.
- Do not implement Face/Object/Text detection in this phase — those are explicitly marked
  "Future Features" in the design doc. Scope for Phase 3 is exactly: motion, texture,
  edges, blur_clarity.

---

## 3. `complexity_estimator.py`

```python
def estimate_complexity(scene_metrics: dict, weights: dict | None = None) -> float:
    """
    scene_metrics: dict with keys motion, texture, edges, blur_clarity
    weights: optional override; defaults to equal weighting (0.25 each)
    """
    w = weights or {"motion": 0.25, "texture": 0.25, "edges": 0.25, "blur_clarity": 0.25}
    complexity = (
        w["motion"] * scene_metrics["motion"]
        + w["texture"] * scene_metrics["texture"]
        + w["edges"] * scene_metrics["edges"]
        + w["blur_clarity"] * (1 - scene_metrics["blur_clarity"])  # low clarity (blurry) -> add to complexity
    )
    return min(max(complexity, 0.0), 1.0)
```

Weights should live in `configs/decision_config.yaml` under a new `complexity_weights` key
so they're tunable without code changes — same principle as the decision thresholds:

```yaml
complexity_weights:
  motion: 0.25
  texture: 0.25
  edges: 0.25
  blur_clarity: 0.25
```

Do not hand-tune these weights yet. Ship v1 with equal weights and only revisit after the
validation step produces a ranking that disagrees with intuition.

---

## 4. Validation / Milestone (required before calling Phase 3 done)

This phase's validation is different in kind from Phases 1–2 — those verified plumbing
("does data flow correctly"). This one verifies **signal quality** ("is the number
meaningful"). Write the results into a markdown report the same style as the Phase 1/2
validation docs already produced, with a table like:

| Frame Description           | Motion | Texture | Edges | Blur Clarity | Complexity |
| --------------------------- | ------ | ------- | ----- | ------------ | ---------- |
| Flat sky / blank wall       |        |         |       |              |            |
| Landscape (moderate detail) |        |         |       |              |            |
| Moderately busy scene       |        |         |       |              |            |
| Close-up face               |        |         |       |              |            |
| Crowded street              |        |         |       |              |            |

### Test 1 — Ranking sanity

Run 4–5 reference frames (grab stills from any test video, or generate synthetic ones —
e.g. a solid-color image for "flat", a high-frequency noise image for "busy") through the
pipeline. Confirm complexity scores are monotonically ordered the way a human would
intuitively rank them (flat < landscape < busy street/face). If the ordering is wrong,
adjust scale factors before touching weights.

### Test 2 — Determinism

Run the _same_ static frame through `analyze_frame` + `estimate_complexity` twice.
Confirm identical output both times (no hidden randomness, no OS-timing-dependent
behavior). Log both runs side by side in the report.

### Test 3 — Frame-to-frame stability

Pull 2–3 adjacent frames from a slow-moving clip (not a hard cut). Confirm the complexity
score does not jump wildly frame-to-frame — a jittery signal here will cause the Decision
Engine to flicker between models once Phase 4 is wired up, which directly undermines the
"Decision Stability" metric already specified in the evaluation chapter. Report the
frame-to-frame delta explicitly (e.g. `|complexity[t] - complexity[t-1]|`), not just the
raw scores.

### Test 4 — First-frame edge case

Confirm `analyze_frame(frame, prev_frame=None)` runs without error and returns
`motion=0.0` (not a crash, not `NaN`) for the very first frame of a video, where there's
no previous frame to diff against.

---

## 5. Design Decision to Make Explicit in the Report

**Per-frame analysis vs. sampling**: decide and document whether every frame gets analyzed,
or whether you sample every Nth frame and reuse the score for frames in between. This
isn't specified in the original design doc and has real tradeoffs:

- Full per-frame: more responsive to sudden scene changes (cuts, a face entering frame),
  but costs more compute and is more prone to jitter (Test 3 above).
- Sampling: cheaper, naturally smoother, but can miss sudden complexity spikes until the
  next sample point.

Recommendation for v1: analyze every frame (simplest, matches the original pipeline
design, and compute cost here is trivial compared to the SR models coming in Phase 5).
Revisit only if benchmarking in Phase 7 shows scene analysis itself is a bottleneck.
State whichever choice you make explicitly in the validation report, since it affects how
later phases (Decision Engine, benchmarking) interpret the complexity signal.

---

## 6. Definition of Done

- [ ] `scene_analyzer.analyze_frame()` implemented, pure function, handles `prev_frame=None`
- [ ] `complexity_estimator.estimate_complexity()` implemented, weights sourced from
      `decision_config.yaml`
- [ ] `SceneDescriptor` dataclass added to `state_types.py`
- [ ] Validation report written (ranking test, determinism test, stability test, edge
      case test) in the same format as the Phase 1/2 reports
- [ ] Per-frame vs. sampling decision stated explicitly
- [ ] Scale factor constants documented as "starting values, not final" if not yet tuned
      against real footage
