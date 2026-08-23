# Step 5 — Empirical Model Benchmarking

## Step 5.1 — Benchmark Dataset / Test-Video Preparation

### 1. Purpose
This step establishes a deterministic, reproducible benchmark dataset (manifest, videos, chunks, and metadata) to support empirical Super-Resolution (SR) model evaluation. Every model analyzed in later steps will run on the exact same inputs (resolutions, frame counts, FPS, and dynamic content patterns).

> [!IMPORTANT]
> **Step 5.1 prepares benchmark INPUTS. It does NOT load SR models, define model runner adapters, or run SR inference.**

---

### 2. Benchmark Corpus Design
The benchmark corpus is designed to cover:
- **Frame Rates**: 30 FPS, 60 FPS, and 120 FPS.
- **Content Diversity**: Exercises three distinct motion profiles:
  - `lowmotion`: Moves dynamic targets at 20 pixels/second.
  - `moderatemotion`: Moves dynamic targets at 100 pixels/second.
  - `highmotion`: Moves dynamic targets at 300 pixels/second.
- **Duration**: Configured to exactly 4.0 seconds by default. Under a target chunk duration of 2.0 seconds, this guarantees exactly 2 chunks per video, allowing fast unit test execution while retaining multi-chunk behavior.

---

### 3. Synthetic Video Generation
Videos are deterministically generated frame-by-frame using OpenCV (`cv2.VideoWriter`) with the `mp4v` codec.
The frames contain:
- A dark blue-gray background with a high-contrast grid pattern (high spatial frequencies).
- Multiple text lines showing the target frame rate and current frame indices.
- A dynamic intersecting crosshair pattern and concentric circles (vibrant cyan/magenta colors) moving along a diagonal bouncing path. The movement speed (pixels/sec) scales with the target frame rate so that spatial displacement per second remains constant across all FPS variants.

---

### 4. Real-World Video Policy
If a real-world test clip is already present in the repository, it is registered. Currently, the initial benchmark dataset relies on the synthetic corpus to ensure perfect pixel-level reproducibility and speed. The prepare script accepts any input video, meaning real-world videos can be added and profiled without any code redesign.

---

### 5. Source Metadata
For every benchmark video, the following metadata is verified and recorded:
- `benchmark_video_id`
- `filename`
- `source_fps`
- `width`
- `height`
- `duration_seconds`
- `frame_count`
- `codec`
- `pixel_format` (default: `yuv420p` for OpenCV containers)
- `source_bitrate` (null if not available)
- `audio_presence`

---

### 6. Chunk Association Mechanism
Step 5.1 reuses the Step 1 profiling pipeline (`run_profiler` from `profile_video.py`) to segment the generated video into dynamic chunks and profile its content complexity.
The chunks are associated by reading the profiler's output artifacts (`{video_id}_profile.json` and `{video_id}_manifest.json`). We merge:
- The chunk timeline ranges (`start_frame`, `end_frame`, `start_time_seconds`, etc.)
- The chunk files relative paths and SHA-256 hashes.

This ensures we consume the authoritative Step 1 timeline without duplicate computation or conflicting definitions.

---

### 7. Dataset Hashing & Integrity
For every generated source video and chunk, a stable SHA-256 cryptographic hash is calculated. Files are checked against these hashes during validation to catch files modified or corrupted in transport. Timestamps and file modification times are not used for identity, ensuring absolute reproducibility across machines.

---

### 8. Directory Structure
All benchmark files are located inside `data/benchmarks/sr/` to keep them cleanly separated from runtime Edge caches and network emulation logs:
```
data/
    benchmarks/
        sr/
            videos/      # High-quality synthetic source MP4s
            chunks/      # Dynamic FFmpeg copy-mode segmented chunk files
            profiles/    # Step 1 content profiles (JSON files)
            manifests/   # Step 1 manifests & the main benchmark_manifest.json
```

---

### 9. CLI Usage

To generate or overwrite the benchmark dataset:
```powershell
python -m adaptive_sr.benchmarking.prepare_dataset --output data/benchmarks/sr/ --overwrite
```

**Supported Options**:
- `--output`: Target folder (defaults to `data/benchmarks/sr`).
- `--duration`: Bounded duration in seconds (defaults to 4.0).
- `--width` / `--height`: Generation resolution (defaults to 640x360).
- `--seed`: Deterministic seed (defaults to 42).
- `--overwrite`: Forces regeneration of existing videos and profiles.
- `--validate`: Path to a dataset manifest to validate.

---

### 10. Validation Mechanism
A dataset validation routine is exposed via:
```powershell
python -m adaptive_sr.benchmarking.prepare_dataset --validate data/benchmarks/sr/manifests/benchmark_manifest.json
```

The validator checks:
1. Manifest structure complies with the Step 5.1 schema.
2. Every video file and chunk file exists.
3. Actual file SHA-256 hashes match the hashes written in the manifest.
4. Actual video container properties (resolution, FPS, frames) match manifest specifications.
5. Chunk timelines start at frame 0, end at the last frame, and have no gaps or overlaps.
6. All `benchmark_video_id`s are unique.

---

### 11. Automated Tests
Unit tests in [`tests/test_benchmark_preparation.py`](file:///d:/Full%20Stack/AdaptiveSR/tests/test_benchmark_preparation.py) cover all 16 specified requirements (FPS detection, hash checks, chunk continuity, duplicate prevention, and tamper detection).

---

### 12. Limitations
- **OpenCV VideoWriter Container timing**: On some operating systems, VideoWriter might introduce tiny floating-point rounding differences in duration. The validator handles this using a small floating-point tolerance check.
- **FFmpeg copy-segmentation boundaries**: Since synthetic videos have keyframes at every frame (by default in raw mpeg4), the chunks segment exactly at the requested 2.0s boundary.
