# AdaptiveSR: Telemetry-Aware & Scene-Adaptive Video Super-Resolution

AdaptiveSR is a source-grounded, real-time **Telemetry-Aware & Scene-Adaptive Video Super-Resolution Framework** designed for low-latency edge devices, intelligent offloading, and adaptive video streaming applications.

Instead of running heavy neural network models statically on every video frame—which causes thermal throttling, VRAM exhaustion, and battery drain—AdaptiveSR dynamically analyzes per-frame **scene complexity** (motion, texture, edge density), **device & network telemetry**, and **fuzzy decision rules** to select the optimal enhancement model (FSRCNN, Real-ESRGAN, BasicVSR++), bitrate tier, or edge offloading target.

---

## 🌟 Key Features

1. **Scene Complexity Analysis:** Real-time multi-feature extraction combining Laplacian texture variance (`cv2.Laplacian`), Canny edge density (`cv2.Canny`), and macroblock motion vector estimation.
2. **Telemetry-Aware Decision Engine:** Configurable hierarchical rule tree and **Fuzzy Inference Engine** (`fuzzy_engine.py`) routing frames smoothly without harsh boundary flutter.
3. **Multi-Model SR Enhancement Suite:**
   - **`tinysr` (FSRCNN):** Ultra-fast lightweight model (~100 KB, <3ms latency).
   - **`real_esrgan`:** High-perceptual GAN-based single-frame upscaling.
   - **`basicvsr++` / Recurrent:** Temporal sequence-dispatch upscaling with graceful boundary fallback to single-frame models.
4. **Real-Time FPS & Bitrate Adaptation:** 
   - **FPS Feasibility (`fps_adapter.py`):** Calculates frame budget vs. processing latency ($1000/\text{source\_fps}$ vs. measured latency) assigning deterministic adaptation tiers (`realtime`, `near_realtime`, `below_realtime`, `severely_below_realtime`).
   - **Bitrate Adaptation (`bitrate_adapter.py`):** Evaluates representation switching (e.g. 360p → 720p) and target quality under bandwidth constraints.
5. **Edge Selection & Resource Allocation (`edge_evaluator.py`):** Evaluates local vs. remote edge node capacity (RTT, GPU load, network bandwidth) for offloading.
6. **End-to-End Runtime Orchestrator (`orchestrator.py`, `telemetry.py`):** Unified real-time execution loop linking telemetry polling, fuzzy inference, SR execution, and stream re-encoding.
7. **Zero-Power Bypass & Dynamic Scale Reduction:** Skips upscaling under low battery (<10%) or low complexity (<15%) and dynamically caps target resolution (e.g., scale $2\times$ vs $4\times$).

---

## 🏗️ System Architecture

```
                                [ Input Video Stream ]
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │  Video Frame Buffer │
                               └──────────┬──────────┘
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
       ┌───────────────────────┐                     ┌───────────────────────┐
       │   Scene Analyzer      │                     │   Telemetry Monitor   │
       │ (Motion/Texture/Edges)│                     │ (GPU/VRAM/Batt/Temp)  │
       └───────────┬───────────┘                     └───────────┬───────────┘
                   │                                             │
                   └──────────────────────┬──────────────────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │ Fuzzy Decision      │
                               │ & Edge Evaluator    │
                               └──────────┬──────────┘
                                          │
         ┌────────────────────────────────┼────────────────────────────────┐
         ▼                                ▼                                ▼
┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
│ Bypass / Skip   │              │  FSRCNN / TinySR│              │  Real-ESRGAN    │
│ (0.0 ms Bypass) │              │  (Lightweight)  │              │  (Perceptual)   │
└────────┬────────┘              └────────┬────────┘              └────────┬────────┘
         │                                │                                │
         └────────────────────────────────┼────────────────────────────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │  Runtime            │
                               │  Orchestrator       │
                               └──────────┬──────────┘
                                          │
                                          ▼
                               [ Enhanced Output Video ]
```

---

## 📅 Project Implementation Roadmap

The framework is built across progressive milestone phases:

### 🔹 Phase 0 & 1: Foundation & Baseline Pipeline
- Formulated modular project structure under `adaptive_sr/` and `src/`.
- Built core video extraction, frame buffering, and OpenCV re-encoding pipelines maintaining FPS and duration synchronization.

### 🔹 Phase 2 & 3: Feature Extraction & Telemetry Sensors
- **Scene Complexity Extractor:** Formulated weighted complexity equation: $\text{complexity} = 0.25 \cdot \text{motion} + 0.50 \cdot \text{texture} + 0.20 \cdot \text{edges} + 0.05 \cdot \text{blur}$.
- **Telemetry Monitor:** Integrated `psutil` and `nvidia-ml-py` to monitor VRAM, GPU utilization, CPU temperature, and battery percentage with fallback defaults.

### 🔹 Phase 4: Dynamic Rule-Based Decision Engine
- Implemented hierarchical decision tree routing logic:
  - *Rule 1 (Device Constraint):* Low battery + hot CPU → Route 100% to `tinysr` (FSRCNN).
  - *Rule 2 (Simple Scene):* Low complexity (<0.15) → Route to `tinysr`.
  - *Rule 3 (High Headroom):* High complexity + GPU headroom → Route to `real_esrgan` / Recurrent.
  - *Rule 4 (Fallback):* Moderate complexity → Default to `real_esrgan`.

### 🔹 Phase 5: Multi-Backend SR Integration & Sequence Fallbacks
- Integrated PyTorch backends for FSRCNN and Real-ESRGAN with Turing GTX FP16 precision underflow fixes.
- Created `SequenceDispatch` for temporal models (BasicVSR++), gracefully routing boundary frames to single-frame fallbacks.

### 🔹 Phase 6 & 7: Remote Edge Service & FPS Adaptation Layer
- Built FastAPI edge service (`adaptive_sr/services/edge/app.py`) for cloud/edge video chunk upscaling and telemetry logging.
- Developed `adaptive_sr/adaptation/fps_adapter.py` calculating frame budget vs. processing latency with deterministic computational proximity tiering.

### 🔹 Phase 8 & 9: Bitrate Adaptation & Edge Resource Allocation
- **Bitrate Adaptation (`bitrate_adapter.py`):** Evaluates representation switching (e.g. 360p → 720p) and target quality under throughput constraints.
- **Edge Selection (`edge_evaluator.py`):** Evaluates local vs. remote edge node capacity (RTT, GPU load, bandwidth) for offloading decisions.

### 🔹 Phase 10: Fuzzy Adaptive Decision Engine
- Implemented triangular and trapezoidal fuzzy membership functions (`fuzzy_engine.py`) to provide smooth, flutter-free multi-variable adaptation across system load and scene complexity.

### 🔹 Phase 11: End-to-End Real-Time Orchestration
- Developed unified runtime orchestrator (`orchestrator.py`, `telemetry.py`) combining telemetry polling, fuzzy inference, model execution, and output video stream re-assembly.

---

## 📊 Benchmark & Quality Performance

Evaluated against ground-truth high-resolution videos using PSNR, SSIM, and LPIPS perceptual quality metrics:

| Test Category | Architecture / Configuration | Average PSNR (dB) | Average SSIM | Latency (s) | GPU Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Simple Category** | Static FSRCNN (`tinysr`) | 39.42 dB | 0.9612 | 2.82 s | Baseline |
| | Static Real-ESRGAN | 39.45 dB | 0.9615 | 70.41 s | 1.0x |
| | **AdaptiveSR (Ours)** | **39.42 dB** | **0.9612** | **2.82 s** | **96.0% Faster** |
| **Mixed Category** | Static Real-ESRGAN | 33.53 dB | 0.9120 | 72.10 s | 1.0x |
| | **AdaptiveSR (Ours)** | **33.39 dB** | **0.9095** | **48.15 s** | **33.2% Faster** |
| **Futbol (Real Clip)** | Static Real-ESRGAN | 29.81 dB | 0.8410 | 73.30 s | 1.0x |
| | **AdaptiveSR (Ours)** | **29.24 dB** | **0.8290** | **65.97 s** | **10.0% Faster** |

---

## 📁 Project Structure

```
AdaptiveSR/
├── adaptive_sr/                # Core Python Package
│   ├── adaptation/             # Real-time adaptation & decision modules
│   │   ├── bitrate_adapter.py  # Bitrate & representation quality adapter (Step 8)
│   │   ├── edge_evaluator.py   # Edge selection & resource allocation (Step 9)
│   │   ├── fps_adapter.py      # Real-time FPS feasibility & adaptation (Step 7)
│   │   └── fuzzy_engine.py     # Fuzzy Adaptive Decision Engine (Step 10)
│   ├── benchmarking/           # Dataset builder & reproducibility validator
│   │   ├── dataset_builder.py
│   │   └── validator.py
│   ├── runtime/                # End-to-end runtime loop (Step 11)
│   │   ├── orchestrator.py     # Real-time execution orchestrator
│   │   └── telemetry.py        # System telemetry polling interface
│   └── services/               # Edge & Cloud Microservices
│       └── edge/
│           └── app.py          # FastAPI edge inference server
├── Architecture Diagrams/      # System architecture & flowchart diagrams
├── benchmark_data/             # Test video pairs (simple, mixed, complex, futbol)
├── benchmark_results/          # Enhanced video outputs & quality logs
├── configs/                    # Telemetry & decision thresholds configuration
│   └── decision_config.yaml
├── data/                       # Benchmark dataset CSV/JSON results
│   └── benchmarks/sr/results/
│       ├── unified_benchmark_dataset.csv
│       └── unified_benchmark_dataset.json
├── Markdowns/                  # Phase-by-phase implementation specs & reports
│   ├── Phase 0,1/
│   ├── Phase 2/
│   ├── Phase 3/
│   ├── Phase 4/
│   ├── Phase 5/
│   ├── Phase 6/
│   ├── Phase 7/
│   ├── Phase 8/
│   ├── Phase 9/
│   ├── Phase 10/
│   ├── Phase 11/
│   └── Results/
├── src/                        # Legacy Pipeline Modules
│   └── modules/
│       ├── decision_engine.py  # Rule-based decision routing
│       ├── enhancement_engine.py # Multi-backend model executor
│       ├── scene_analyzer.py   # Motion/Texture/Edge complexity extractor
│       └── telemetry_monitor.py # GPU/CPU/Battery sensor monitor
├── tests/                      # Pytest Test Suite
│   ├── test_benchmark_dataset.py
│   ├── test_bitrate_adaptation.py # Step 8 tests
│   ├── test_edge_selection.py     # Step 9 tests
│   ├── test_end_to_end_runtime.py # Step 11 tests
│   ├── test_fps_adaptation.py     # Step 7 tests
│   ├── test_fuzzy_decision.py     # Step 10 tests
│   ├── test_remote_sr.py
│   └── test_validation_report.py
├── requirements.txt            # Pinned dependencies
├── STEP10_IMPLEMENTATION.md   # Step 10 Fuzzy Decision Engine documentation
├── STEP11_IMPLEMENTATION.md   # Step 11 End-to-End Runtime documentation
└── README.md                   # Project Documentation
```

---

## ⚙️ Quick Start Setup

### Prerequisites
* **Python:** 3.11.9
* **CUDA GPU:** NVIDIA GPU with CUDA 11.8+ (e.g. GTX 1650 / RTX 3060/4060+)
* **Dependencies:** PyTorch, OpenCV, PyYAML, psutil, FastAPI, uvicorn

### 1. Installation
```bash
git clone https://github.com/AbishekR05/AdaptiveSR.git
cd AdaptiveSR

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Test Suite
Execute unit and integration tests to verify the pipeline:
```bash
pytest tests/
```

### 3. Launching Edge Microservice
Start the FastAPI Edge server on port 8000:
```bash
uvicorn adaptive_sr.services.edge.app:app --host 0.0.0.0 --port 8000 --reload
```
Visit `http://localhost:8000/docs` to test endpoints via Swagger UI.

---

## 📜 License & Citation

Developed as part of the **AdaptiveSR Project** for real-time video super-resolution streaming.