# AdaptiveSR: Telemetry-Aware & Scene-Adaptive Video Super-Resolution

AdaptiveSR is a source-grounded, real-time **Telemetry-Aware & Scene-Adaptive Video Super-Resolution Framework** designed for low-latency edge devices and streaming applications.

Instead of running heavy neural network models statically on every video frame—which causes thermal throttling, VRAM exhaustion, and battery drain—AdaptiveSR dynamically analyzes per-frame **scene complexity** (motion, texture, edge density) and real-time **device telemetry** (GPU load, VRAM, battery level, thermals) to select the optimal enhancement model (FSRCNN, Real-ESRGAN, BasicVSR++) or frame bypass tier.

---

## 🌟 Key Features

1. **Scene Complexity Analysis:** Real-time multi-feature extraction combining Laplacian texture variance (`cv2.Laplacian`), Canny edge density (`cv2.Canny`), and macroblock motion vector estimation.
2. **Telemetry-Aware Decision Engine:** Configurable hierarchical rule tree (`decision_config.yaml`) routing frames based on live sensor metrics (GPU load, VRAM, CPU thermal state, battery level).
3. **Multi-Model SR Enhancement Suite:**
   - **`tinysr` (FSRCNN):** Ultra-fast lightweight model (~100 KB, <3ms latency).
   - **`real_esrgan`:** High-perceptual GAN-based single-frame upscaling.
   - **`basicvsr++` / Recurrent:** Temporal sequence-dispatch upscaling with graceful boundary fallback to single-frame models.
4. **Real-Time FPS Feasibility & Adaptation Layer:** Calculates computational headroom ($1000/\text{source\_fps}$ vs. measured latency) and assigns deterministic adaptation tiers (`realtime`, `near_realtime`, `below_realtime`, `severely_below_realtime`).
5. **Zero-Power Bypass & Dynamic Scale Reduction:** Skips upscaling under low battery (<10%) or low complexity (<15%) and dynamically caps target resolution (e.g., scale $2\times$ vs $4\times$) to prevent device shutdown.
6. **Edge & Cloud Service Integration:** FastAPI edge application supporting remote inference, cache management, and telemetry streaming.

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
                               │   Decision Engine   │
                               │ (Rule Tree Router)  │
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
                               │  Video Encoder /    │
                               │ FPS Adaptation Layer│
                               └──────────┬──────────┘
                                          │
                                          ▼
                               [ Enhanced Output Video ]
```

---

## 📅 Project Implementation Roadmap

The framework was implemented across progressive milestone phases:

### 🔹 Phase 0 & 1: Foundation & Baseline Pipeline
- Formulated the modular project structure under `adaptive_sr/` and `src/`.
- Built core video extraction, frame buffering, and OpenCV re-encoding pipelines maintaining FPS and duration synchronization.

### 🔹 Phase 2 & 3: Feature Extraction & Telemetry Sensors
- **Scene Complexity Extractor:** Formulated weighted complexity equation: $\text{complexity} = 0.25 \cdot \text{motion} + 0.50 \cdot \text{texture} + 0.20 \cdot \text{edges} + 0.05 \cdot \text{blur}$.
- **Telemetry Monitor:** Integrated `psutil` and `nvidia-ml-py` to monitor VRAM, GPU utilization, CPU temperature, and battery percentage with fallback defaults.

### 🔹 Phase 4: Dynamic Rule-Based Decision Engine
- Implemented a hierarchical decision tree routing logic:
  - *Rule 1 (Device Constraint):* Low battery + hot CPU $\rightarrow$ Route 100% to `tinysr` (FSRCNN).
  - *Rule 2 (Simple Scene):* Low complexity (<0.15) $\rightarrow$ Route to `tinysr`.
  - *Rule 3 (High Headroom):* High complexity + GPU headroom $\rightarrow$ Route to `real_esrgan` / Recurrent.
  - *Rule 4 (Fallback):* Moderate complexity $\rightarrow$ Default to `real_esrgan`.

### 🔹 Phase 5: Multi-Backend SR Integration & Sequence Fallbacks
- Integrated PyTorch backends for FSRCNN and Real-ESRGAN with Turing GTX FP16 precision underflow fixes.
- Created `SequenceDispatch` for temporal models (BasicVSR++), gracefully routing boundary frames to single-frame fallbacks.

### 🔹 Phase 6: Remote SR Inference & Edge Microservice
- Built a FastAPI edge service (`adaptive_sr/services/edge/app.py`) for cloud/edge video chunk upscaling, cache lookups, and telemetry logging.

### 🔹 Phase 7: Real-Time FPS Adaptation & Feasibility
- Developed `adaptive_sr/adaptation/fps_adapter.py` calculating frame budget vs. processing latency ($\text{real\_time\_ratio} = \frac{\text{frame\_budget\_ms}}{\text{measured\_latency\_ms}}$).
- Generated machine-readable `FPSAdaptationSignal` payload with deterministic computational proximity tiering.

### 🔹 Phase 8: Hardening & Quantization Benchmarks
- Implemented **Rule 0 Skip Tier** (<10% battery + <15% complexity) achieving **0.0 ms instant bypass**.
- Benchmarked ONNX Runtime INT8 dynamic quantization for FSRCNN on CPU (`tinysr_int8`).

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
│   ├── adaptation/             # Real-time FPS adaptation & feasibility signal layer
│   │   └── fps_adapter.py
│   ├── benchmarking/           # Dataset builder & reproducibility validator
│   │   ├── dataset_builder.py
│   │   └── validator.py
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
│   └── Phase 7/
├── src/                        # Legacy Pipeline Modules
│   └── modules/
│       ├── decision_engine.py  # Rule-based decision routing
│       ├── enhancement_engine.py # Multi-backend model executor
│       ├── scene_analyzer.py   # Motion/Texture/Edge complexity extractor
│       └── telemetry_monitor.py # GPU/CPU/Battery sensor monitor
├── tests/                      # Pytest Test Suite
│   ├── test_benchmark_dataset.py
│   ├── test_fps_adaptation.py
│   ├── test_fps_feasibility.py
│   ├── test_remote_sr.py
│   └── test_validation_report.py
├── requirements.txt            # Pinned dependencies
├── STEP6_IMPLEMENTATION.md    # Step 6 Remote SR documentation
├── STEP7_IMPLEMENTATION.md    # Step 7 FPS Adaptation documentation
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
