# Step 1 — Video & Content Profiling Implementation

This document describes the implementation of the source-side offline profiling pipeline (Step 1) for the AdaptiveSR project.

The profiler processes a source video offline to extract metadata, segment it deterministically, analyze spatial and temporal visual characteristics, and export structured content profiles and integrity manifests.

---

## 1. Purpose & Core Goal

Step 1 acts as an **offline preprocessing step** at the content source. Before transmission or super-resolution scheduling starts, the profiler indexes the video to capture scene complexity and motion. This information is saved to a reproducible profile dataset, which later stages use to make optimal ABR (Adaptive Bitrate) and SR (Super-Resolution) decisions.

---

## 2. Reused & Adapted Legacy Components

To preserve compatibility while improving robustness, we wrap and adapt the legacy `src/` modules:
* **`VideoLoader`** ([`src/modules/video_loader.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/video_loader.py)): Reused to extract input metadata (width, height, FPS, frame count, codec) and analyze individual chunk segments.
* **`FrameExtractor`** ([`src/modules/frame_extractor.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/frame_extractor.py)): Reused to sequentially iterate frames for each chunk file.
* **`analyze_frame`** ([`src/modules/scene_analyzer.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/scene_analyzer.py)): Extracted visual metrics (edge density via Canny, texture density and blur clarity via Laplacian variance).
* **`estimate_complexity`** ([`src/modules/complexity_estimator.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/complexity_estimator.py)): Reused to evaluate spatial complexity using weights loaded from the YAML config.

---

## 3. Dynamic FPS & Temporal Motion Strategy

The legacy motion comparison compares frame $t$ with frame $t-1$. At higher frame rates (60/120 FPS), the inter-frame time gap is smaller, which artificially reduces raw motion deltas.

To make motion comparable across 30, 60, and 120 FPS sources, Step 1 implements a **constant temporal comparison window**:
* The profiler compares frame $t$ with frame $t - N$, where:
  $$N = \max(1, \text{round}(\text{source\_fps} \times \text{motion\_temporal\_window\_seconds}))$$
* By default, `motion_temporal_window_seconds = 0.0333` ($\approx 1/30\text{ s}$), ensuring:
  * **30 FPS**: $N = 1$ (compare frame $t$ with $t-1$)
  * **60 FPS**: $N = 2$ (compare frame $t$ with $t-2$)
  * **120 FPS**: $N = 4$ (compare frame $t$ with $t-4$)
* We maintain a circular buffer of size $N$ to evaluate frame $t$ against frame $t - N$.

---

## 4. Deterministic Chunking

The video is physically segmented into chunks of `--chunk-duration` seconds using keyframe-aligned copy mode in FFmpeg:
```powershell
ffmpeg -y -i <input> -c copy -f segment -segment_time <duration> -reset_timestamps 1 <pattern>
```
* Slicing physically ensures that the frames analyzed are identical to the segments served to players.
* The starting/ending frame indices, frame count, and actual durations are derived dynamically for each chunk by running `VideoLoader` on the generated chunk file.

---

## 5. Aggregation Rules

To prevent short high-motion or high-complexity spikes from being hidden by averages, we store multiple statistics for each feature:
* **Motion**: `mean`, `p95` (95th percentile), and `max`.
* **Spatial Complexity**: `mean`, `p95`, and `max`.
* **Texture Density**: `mean` and `p95`.
* **Edge Density**: `mean` and `p95`.
* **Blur / Clarity**: `mean` and `p95`.

Aggregations are computed using `numpy.percentile` for numerical correctness.

---

## 6. Dataset Schema & Manifest Structure

### Content Profile JSON (`data/profiles/{video_id}_profile.json`)
Contains schemas versioning, source video properties, profiler configurations, and the chunk-by-chunk metrics list. File hashes are excluded to prevent data clutter:
```json
{
    "schema_version": "1.0.0",
    "video_id": "sample",
    "source": {
        "filename": "sample.mp4",
        "duration_seconds": 6.0,
        "fps": 30.0,
        "width": 640,
        "height": 480,
        "frame_count": 180,
        "codec": "h264",
        "pixel_format": null,
        "bitrate": null,
        "has_audio": false
    },
    "profiling_config": {
        "chunk_duration_seconds": 2.0,
        "motion_temporal_window_seconds": 0.0333,
        "aggregation": {
            "motion": ["mean", "p95", "max"],
            "texture": ["mean", "p95"],
            "edge_density": ["mean", "p95"],
            "blur": ["mean", "p95"],
            "complexity": ["mean", "p95", "max"]
        }
    },
    "chunks": [
        {
            "chunk_id": "0000",
            "start_time_seconds": 0.0,
            "end_time_seconds": 2.0,
            "duration_seconds": 2.0,
            "start_frame": 0,
            "end_frame": 59,
            "frame_count": 60,
            "motion": { "mean": 0.12, "p95": 0.18, "max": 0.22 },
            "texture_density": { "mean": 0.35, "p95": 0.42 },
            "edge_density": { "mean": 0.08, "p95": 0.11 },
            "blur": { "mean": 0.85, "p95": 0.90 },
            "spatial_complexity": { "mean": 0.28, "p95": 0.33, "max": 0.38 }
        }
    ]
}
```

### Integrity Manifest JSON (`data/manifests/{video_id}_manifest.json`)
Maintains repeatability and verification mapping, linking file hashes for the source video and each individual segment:
```json
{
    "schema_version": "1.0.0",
    "video_id": "sample",
    "source_file_path": "/absolute/path/to/sample.mp4",
    "source_file_hash": "sha256_hash_here",
    "generated_profile_path": "/absolute/path/to/sample_profile.json",
    "chunks": [
        {
            "chunk_id": "0000",
            "file_path": "chunks/sample_0000.mp4",
            "file_hash": "sha256_hash_here"
        }
    ]
}
```

---

## 7. Data Leakage Rules

To prevent information leakage, the source profiler is **strictly prohibited** from referencing:
* Output quality metrics (PSNR, SSIM, LPIPS, VMAF).
* Runtime Edge Server loads (CPU, GPU, VRAM).
* Network state (RTT, bandwidth, chunk fetch times).
* Future scheduling decisions or playback stall results.

---

## 8. CLI Usage

Generate video profile and segment files using the command line:

```powershell
python -m adaptive_sr.profiling.profile_video \
    --input benchmark_data/mixed_lr.mp4 \
    --output ./data \
    --chunk-duration 2.0
```

* `--input` (required): Source video file.
* `--output` (required): Output directory.
* `--chunk-duration` (optional): Duration per segment in seconds (default `2.0`).
* `--motion-temporal-window` (optional): Real-time temporal window for frame comparisons in seconds (default `0.0333`).

---

## 9. Test Verification

Automated tests are located in [`tests/test_profiling.py`](file:///d:/Full%20Stack/AdaptiveSR/tests/test_profiling.py).
Run the suite with:
```powershell
python -m pytest tests/test_foundation.py tests/test_profiling.py -v
```

This verifies:
1. Video metadata extraction accuracy.
2. Deterministic chunk boundaries.
3. Proper chunk timings and frame offsets at 30, 60, and 120 FPS.
4. Correct percentile and max aggregations.
5. Strict schema conformity and data-leakage protection.

---

## 10. Known Limitations

* **Keyframe Dependency**: Slicing using FFmpeg copy mode (`-c copy`) splits the video at the closest preceding keyframe. If keyframe intervals in the source are not perfectly uniform, chunk boundaries may deviate slightly from the target duration (e.g. 2.05s instead of 2.0s). However, our profiling dynamically recalculates chunk frame counts and durations on the actual output chunks to maintain consistency.
