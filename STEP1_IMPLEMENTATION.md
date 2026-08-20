# Step 1 — Video & Content Profiling Implementation

This document describes the implementation of the source-side offline profiling pipeline (Step 1) for the AdaptiveSR project.

The profiler processes a source video offline to extract metadata, assign frames to chunks, analyze spatial and temporal visual characteristics continuously, and export structured content profiles and integrity manifests.

---

## 1. Continuous Source Profiling & Temporal Continuity

In segment-based video analysis, treating each chunk as an independent video causes a systematic loss of temporal context at chunk boundaries. For example, to estimate motion for frame $t$, the system compares it against frame $t - N$. When a segment boundaries reset occurs, the first $N$ frames of every chunk $k$ cannot access context from chunk $k-1$, causing errors and underestimating motion.

To resolve this, Step 1 implements a **Continuous Profiling Pass**:
* The profiler opens the original source video exactly once using OpenCV `cv2.VideoCapture` and parses all frames continuously from beginning to end.
* The temporal motion comparison state (circular buffer of size $N$) is maintained globally across the entire video.
* Chunk boundaries do **not** trigger a buffer reset.
* Only the true beginning of the source video (the first $N$ frames of Chunk 0) lacks prior temporal context. Frame 0 of Chunk $k$ is correctly compared against frame $t-N$ belonging to Chunk $k-1$.

---

## 2. Dynamic FPS & Constant Temporal Comparison Window

The temporal motion comparison compares frame $t$ with frame $t - N$. To ensure that motion metrics are comparable across 30, 60, and 120 FPS sources, the comparison frame offset $N$ scales dynamically based on parsed FPS:
$$N = \max(1, \text{round}(\text{source\_fps} \times \text{motion\_temporal\_window\_seconds}))$$

By default, the temporal comparison window is configured to `0.0333` seconds ($\approx 1/30\text{ s}$), resulting in:
* **30 FPS**: $N = 1$ (compare frame $t$ with $t-1$)
* **60 FPS**: $N = 2$ (compare frame $t$ with $t-2$)
* **120 FPS**: $N = 4$ (compare frame $t$ with $t-4$)

This practical normalization compares frames separated by approximately the same amount of real time.

---

## 3. Timestamp-Based Chunk Bucketing

After calculating per-frame metrics continuously, frames are mapped to their corresponding chunk boundaries using **Timestamp-Based/Frame Offset Bucketing**:
1. We segment the video physically first using FFmpeg copy segmenting.
2. We query each generated chunk file using OpenCV to discover its exact frame count and duration.
3. This creates a list of deterministic start/end frame boundaries for each chunk.
4. During the continuous profiling pass, a frame at index `idx` is assigned to chunk $k$ if:
   $$\text{start\_frame}_k \le \text{idx} \le \text{end\_frame}_k$$
5. Chunk aggregations are performed on the bucketed frame metrics belonging to that chunk. This ensures every frame belongs to exactly one chunk.

---

## 4. Target vs. Actual Chunk Duration (Hard Contract)

Step 1 establishes a strict distinction between requested and actual segment metrics:
* **`target_chunk_duration_seconds`**: The requested configuration target (e.g., `2.0` seconds).
* **`chunk.duration_seconds`**: The **actual** logical profiling duration of that specific chunk.

Because copy-codec FFmpeg (`-c copy`) segmenting must split videos at keyframe boundaries (IDR frames) to avoid re-encoding latency, chunk durations will vary slightly depending on keyframe distribution:
* Chunk 0000: `2.000` s
* Chunk 0001: `2.033` s
* Chunk 0002: `1.967` s

**Downstream consumers (buffer evolution calculators, transmission deadlines, and bitrate allocators) MUST use the per-chunk actual duration** (`chunk.duration_seconds`) rather than assuming a static target duration.

---

## 5. Reused & Adapted Legacy Components

* **`VideoLoader`** ([`src/modules/video_loader.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/video_loader.py)): Reused to extract input metadata and inspect generated chunks.
* **`FrameExtractor`** ([`src/modules/frame_extractor.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/frame_extractor.py)): Reused to load frames sequentially.
* **`analyze_frame`** ([`src/modules/scene_analyzer.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/scene_analyzer.py)): Extracted visual metrics (edges, texture, blur clarity).
* **`estimate_complexity`** ([`src/modules/complexity_estimator.py`](file:///d:/Full%20Stack/AdaptiveSR/src/modules/complexity_estimator.py)): Reused to calculate spatial complexity.

---

## 6. Aggregation Rules

* **Motion**: `mean`, `p95` (95th percentile), and `max`.
* **Spatial Complexity**: `mean`, `p95`, and `max`.
* **Texture Density**: `mean` and `p95`.
* **Edge Density**: `mean` and `p95`.
* **Blur / Clarity**: `mean` and `p95`.

Aggregations are computed using `numpy.percentile` for numerical correctness.

---

## 7. Dataset Schema & Manifest Structure

### Content Profile JSON (`data/profiles/{video_id}_profile.json`)
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

## 8. Data Leakage Rules

The source profiler is **strictly prohibited** from referencing output quality metrics (PSNR, SSIM, LPIPS, VMAF) or Edge runtime load metrics (telemetry, RTT, cache hit status). These are deferred to later evaluation stages.

---

## 9. CLI Usage

```powershell
python -m adaptive_sr.profiling.profile_video \
    --input benchmark_data/mixed_lr.mp4 \
    --output ./data \
    --chunk-duration 2.0
```

---

## 10. Automated Tests

Run tests using:
```powershell
python -m pytest tests/test_foundation.py tests/test_profiling.py -v
```

This verifies:
1. Video metadata extraction.
2. Deterministic chunk boundaries.
3. 30/60/120 FPS frame counts and comparison offsets.
4. Numerical percentile aggregation.
5. Repeatability and schema completeness.
6. Temporal continuity across segment boundaries.
7. Insulated data-leakage protection.
