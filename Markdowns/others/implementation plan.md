# Implementation Plan

## Adaptive Resource- and Content-Aware Edge Video Super-Resolution Framework

This turns the design doc into an actual build order. Total estimate: **10–14 weeks** for a working prototype + benchmarks, assuming solo/duo B.Tech team, part-time.

---

## 0. Tech Stack Decisions (lock these before writing code)

| Concern           | Choice                                                                                                   | Why                                                                                           |
| ----------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Language          | Python 3.10/3.11                                                                                         | Ecosystem match for all target models                                                         |
| DL framework      | PyTorch                                                                                                  | Real-ESRGAN, BasicVSR++, FSRCNN all have PyTorch weights                                      |
| Video I/O         | OpenCV + FFmpeg (via subprocess)                                                                         | OpenCV for frame access, FFmpeg for reliable encode/decode & audio passthrough                |
| Device monitoring | `psutil` (CPU/RAM/battery), `GPUtil` or `pynvml` (NVIDIA GPU), `wmi`/`clr` on Windows for temp if needed | Cross-platform baseline; GPU temp is the hard part — see note below                           |
| Config            | YAML via `pyyaml`                                                                                        | Matches your `decision_config.yaml` plan                                                      |
| Logging           | Python `logging` + CSV/JSONL sink                                                                        | Needed for the adaptive-behaviour metrics later                                               |
| Packaging         | `pip` + `requirements.txt`, optionally `venv`                                                            | Keep it simple, no need for Poetry/Docker unless you want reproducibility points in the paper |

**Temperature/thermal caveat:** consumer laptops rarely expose clean thermal APIs cross-platform. Practical fallback: use `psutil.sensors_temperatures()` on Linux (often works), and on Windows either accept "unavailable → treat as normal" or use `pythonnet` + OpenHardwareMonitor's WMI namespace. Don't let this block Phase 2 — build the module with a `None`-safe fallback from day one.

---

## 1. Week-by-Week Roadmap

### Weeks 1: Phase 0 — Environment

- Repo, folder structure (from your doc), `requirements.txt`, `.gitignore`
- `configs/system.yaml`, `configs/models.yaml`, `configs/decision_config.yaml` as **empty stubs** — fill in as modules land
- A `logging_setup.py` that every module imports (build this first, not last — you'll want logs from Phase 1 onward)

### Weeks 2: Phase 1 — Video Pipeline

- `video_loader.py`: open file, pull metadata (fps, resolution, frame count, codec) via OpenCV + `ffprobe`
- `frame_extractor.py`: yield frames as a generator (don't load a whole video into RAM — this matters once you hit multi-minute clips)
- `encoder.py`: reassemble frames → video with FFmpeg, preserving original audio track (`-map 0:a`) and fps
- **Milestone check:** input.mp4 → identical output.mp4 (no enhancement) — passes if diff in frame count/duration is zero

### Week 3: Phase 2 — Device Monitor

- `device_monitor.py` running as a background thread, polling every 0.5s, exposing a thread-safe `get_state()` → `DeviceState` dataclass
- Normalize every field to [0,1] at the source (don't leave normalization to the Decision Engine — keeps that module simpler)
- **Milestone check:** print live CPU/RAM/battery while a dummy CPU-burn loop runs in parallel; values should visibly move

### Weeks 4–5: Phase 3 — Scene Analysis

- `scene_analyzer.py`:
  - Motion: frame differencing or sparse optical flow (`cv2.calcOpticalFlowFarneback` is fine, don't need RAFT for v1)
  - Edges: `cv2.Canny` density
  - Texture: Laplacian variance or local binary pattern variance
  - Blur/noise: Laplacian variance (reuse), simple noise estimate via high-pass filter
- `complexity_estimator.py`: weighted sum → single 0–1 score. **Start with equal weights**, tune later — don't over-engineer the formula before you have data
- **Milestone check:** run against 3 hand-picked frames (blue sky, busy street, face close-up) and confirm the ordering of complexity scores matches intuition. This is your first "sanity" evidence for the thesis.

### Week 6: Phase 4 — Decision Engine (rules only, no models yet)

- `decision_engine.py` takes `DeviceState` + `SceneDescriptor`, applies the IF/THEN rules from your doc, returns a `Decision` object naming a model **string**
- Load thresholds from `decision_config.yaml` — nothing hardcoded
- **Milestone check:** feed synthetic device/scene states and confirm decisions match your decision matrix table exactly. Write this as a small pytest suite — it becomes your ablation baseline later.

### Weeks 7–8: Phase 5 — Model Integration

- `model_registry.py`: dict of model name → {loader function, metadata (memory, latency, quality rating, supported scales)}
- Start with **one** model working end-to-end before adding the others — Real-ESRGAN has the best-documented inference API and pretrained weights (`RealESRGAN_x4plus.pth`, also x2 variants)
- Then FSRCNN (tiny, easy, good "lightweight" baseline — plenty of PyTorch reimplementations with pretrained weights)
- BasicVSR++ last — it's heavier to integrate (needs sequences of frames, not single frames, since it's a _video_ SR model with temporal propagation). Budget extra time here; this is the highest-risk integration item.
- `enhancement_engine.py`: given a model name + frame(s), run inference, return enhanced frame. No decision logic here — literally just dispatch + inference.
- **Milestone check:** each model independently: LR frame in → HR frame out, visually inspect

### Week 9: Phase 6 — Pipeline Integration

- Wire everything: Loader → Extractor → (parallel) Scene Analyzer + Device Monitor → Decision Engine → Enhancement Engine → Frame Buffer → Encoder
- `frame_buffer.py`: since BasicVSR++ needs sequences, this is also where you handle the "model needs N frames of context" problem — buffer a small sliding window even for single-frame models, so switching models mid-video doesn't break anything
- **Milestone check:** one full uploaded video → adaptive enhanced output, model selection logged per frame

### Week 10: Phase 7 — Benchmarking harness

- `benchmark/run_baselines.py`: runs the same video through 4 configs (TinySR-only, Real-ESRGAN-only, BasicVSR++-only, Adaptive) and dumps a CSV per run
- Log schema (per frame): frame_no, timestamp, selected_model, complexity, cpu, gpu, ram, battery, temp, inference_ms, decision_reason
- **This is the deliverable your whole thesis argument rests on** — don't rush it

### Week 11: Phase 8 — Optimization (only if time allows)

- Easy wins first: batch inference where possible, cache loaded model weights instead of reloading per frame, async I/O for frame read/write
- Skip anything exotic (async device polling threads are enough; don't build a task scheduler)

### Week 12: Phase 10 — Experimental Evaluation

- Run against your categorized test set (landscape, faces, sports, text-heavy, etc.)
- Compute PSNR/SSIM/LPIPS (use `scikit-image` for PSNR/SSIM, `lpips` pip package for LPIPS) against the original HR source if you have one, or against the best static baseline if not
- Produce the comparison tables/plots your doc already specifies (model selection distribution, complexity distribution, CPU/GPU/battery deltas vs baselines)

### Weeks 13–14: Phases 11–12 — Documentation + Paper

- Architecture diagrams (your pipeline diagrams from the doc, cleaned up)
- Draft IEEE sections in parallel with finishing benchmarks, not after — results interpretation is faster while the run details are fresh

---

## 2. Minimal Folder Structure to Create Now

```
AdaptiveEdgeSR/
├── src/
│   ├── modules/
│   │   ├── video_loader.py
│   │   ├── frame_extractor.py
│   │   ├── device_monitor.py
│   │   ├── scene_analyzer.py
│   │   ├── complexity_estimator.py
│   │   ├── decision_engine.py
│   │   ├── model_registry.py
│   │   ├── enhancement_engine.py
│   │   ├── frame_buffer.py
│   │   └── encoder.py
│   ├── utils/
│   │   ├── logging_setup.py
│   │   └── state_types.py      # DeviceState, SceneDescriptor, Decision dataclasses
│   └── main.py
├── configs/
│   ├── decision_config.yaml
│   ├── models.yaml
│   └── system.yaml
├── models/
│   ├── realesrgan/
│   ├── basicvsr/
│   └── tinysr/
├── benchmark/
│   └── run_baselines.py
├── experiments/
├── outputs/
├── logs/
├── tests/
│   └── test_decision_engine.py
├── requirements.txt
└── README.md
```

---

## 3. Starter Code Skeletons

### `src/utils/state_types.py`

```python
from dataclasses import dataclass

@dataclass
class DeviceState:
    cpu: float          # 0-1
    gpu: float | None    # 0-1, None if no GPU
    ram: float           # 0-1 used
    battery: float | None  # 0-1, None if desktop/no battery
    charging: bool | None
    temperature: float | None  # 0-1 normalized, None if unavailable
    fps: float            # current measured processing fps

@dataclass
class SceneDescriptor:
    motion: float
    texture: float
    edges: float
    complexity: float
    degradation: float = 0.0

@dataclass
class Decision:
    model: str
    scale: int
    reason: str
    priority: str = "medium"
```

### `src/modules/device_monitor.py` (skeleton)

```python
import threading, time
import psutil
from src.utils.state_types import DeviceState

class DeviceMonitor:
    def __init__(self, poll_interval=0.5):
        self.poll_interval = poll_interval
        self._state = DeviceState(cpu=0, gpu=None, ram=0, battery=None,
                                   charging=None, temperature=None, fps=0)
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            cpu = psutil.cpu_percent() / 100.0
            ram = psutil.virtual_memory().percent / 100.0
            batt = psutil.sensors_battery()
            battery = batt.percent / 100.0 if batt else None
            charging = batt.power_plugged if batt else None
            # GPU + temp: fill in with GPUtil/pynvml, guard with try/except
            with self._lock:
                self._state = DeviceState(cpu=cpu, gpu=None, ram=ram,
                                           battery=battery, charging=charging,
                                           temperature=None, fps=self._state.fps)
            time.sleep(self.poll_interval)

    def get_state(self) -> DeviceState:
        with self._lock:
            return self._state
```

### `src/modules/decision_engine.py` (skeleton)

```python
import yaml
from src.utils.state_types import DeviceState, SceneDescriptor, Decision

class DecisionEngine:
    def __init__(self, config_path="configs/decision_config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

    def decide(self, device: DeviceState, scene: SceneDescriptor) -> Decision:
        c = self.cfg["thresholds"]

        if device.battery is not None and device.battery < c["low_battery"] and \
           device.temperature is not None and device.temperature > c["high_temp"]:
            return Decision(model="tinysr", scale=2, reason="low battery + high temp")

        if scene.complexity < c["low_complexity"]:
            return Decision(model="tinysr", scale=2, reason="low complexity frame")

        if scene.complexity > c["high_complexity"] and device.cpu < c["cpu_headroom"]:
            return Decision(model="real_esrgan", scale=2, reason="high complexity, cpu available")

        if scene.complexity > c["very_high_complexity"] and (device.gpu or 0) < c["gpu_headroom"]:
            return Decision(model="basicvsr++", scale=2, reason="very high complexity, gpu available")

        return Decision(model="real_esrgan", scale=2, reason="default")
```

### `configs/decision_config.yaml` (starter)

```yaml
thresholds:
  low_battery: 0.20
  high_temp: 0.75 # normalized; calibrate against real sensor range
  low_complexity: 0.25
  high_complexity: 0.75
  very_high_complexity: 0.90
  cpu_headroom: 0.60
  gpu_headroom: 0.80
```

Write the pytest suite against this file's exact numbers — that gives you a regression test for free when you start tuning thresholds later.

---

## 4. Where to Get Pretrained Weights

- **Real-ESRGAN** — official repo (`xinntao/Real-ESRGAN`) ships `RealESRGAN_x2plus.pth` / `x4plus.pth`, plus a pip-installable inference wrapper (`realesrgan` package) — fastest path to a working model
- **BasicVSR++** — official weights via `mmediting`/`mmagic` (OpenMMLab). Note: MMagic has a real learning curve and its own config system — budget the extra week mentioned above specifically for wrangling this dependency
- **FSRCNN (lightweight)** — several minimal PyTorch reimplementations with pretrained `.pth` files on GitHub; if none fit cleanly, it's small enough to train from scratch on a tiny dataset in a few hours on a single GPU (this wouldn't violate your "no training VSR models" decision since FSRCNN is your lightweight fallback, not your research contribution)

---

## 5. Risk Management — What Will Actually Slow You Down

1. **BasicVSR++ integration** (temporal model, different input shape than the other two) — highest-risk single item. Consider treating it as optional/stretch and shipping the thesis with Real-ESRGAN + FSRCNN + adaptive switching as the core result if time runs short.
2. **GPU temperature monitoring** — don't chase this hard; battery + CPU + GPU utilization alone are enough to demonstrate the adaptive behavior your evaluation needs.
3. **Frame-buffer complexity from mixing single-frame and multi-frame models** — solve this early (Week 9) rather than patching it in during benchmarking.
4. **LPIPS/PSNR need a reference HR video** — for downloaded YouTube clips you won't have true ground truth. Standard workaround: downscale a genuinely high-res source to create your LR input, so the original stays as ground truth (common practice, and it's literally what Base Paper 3 in your lit review addresses re: real-world degradation).

---

## 6. Immediate Next Steps (this week)

1. Create the repo + folder skeleton above
2. Write `requirements.txt`: `torch`, `opencv-python`, `psutil`, `pyyaml`, `numpy`, `ffmpeg-python` (or just shell out to ffmpeg directly), `pytest`
3. Get Phase 1 (video in → video out, no enhancement) working — this is the fastest path to a runnable `main.py` and gives you a scaffold everything else plugs into
4. In parallel, install and test the Real-ESRGAN pip package standalone (outside your pipeline) so integration risk is retired early
