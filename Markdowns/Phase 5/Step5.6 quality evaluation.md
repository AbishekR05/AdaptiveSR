# Step 5.6 — Quality Evaluation Specification

## Pre-Implementation Gaps Identified

The following gaps must be acknowledged before implementation begins:

1. **No Layer B corpus exists.** The Step 5.1 synthetic corpus (Layer A) is explicitly
   disqualified for perceptual quality conclusions by Step 5.1's own scope statement. Layer B
   (real-world reference clips) was deferred to this step. Antigravity must source and commit
   it; the specification defines the requirements below.

2. **VMAF toolchain is absent.** No `ffmpeg` VMAF build, `vmaf` Python binding, or
   `libvmaf` dependency appears anywhere in the project. VMAF is conditionally specified here
   — it is computed only if the toolchain is present at runtime; otherwise the result field
   is `null` with reason `"vmaf_unavailable"`. Do not silently omit or fabricate VMAF values.

3. **Color space of adapter I/O is BGR uint8.** All adapters return BGR. PSNR/SSIM
   implementations vary in expected color space. This must be handled explicitly; see §6.

4. **No frame-level join key exists in 5.5.** Step 5.5 records `benchmark_video_id` and
   `chunk_id` but quality evaluation operates over individual frames within a chunk.
   A `frame_index` field (representation-local, 0-based within the chunk) is added in 5.6
   to enable per-frame join.

---

## 1. Purpose

Step 5.6 measures the perceptual quality output of each SR model against reference frames.
It is entirely separate from the Step 5.5 latency/resource harness. Quality metric
computation time must not appear in any Step 5.5 latency record.

The goal is a joinable, per-frame quality result set across (model, scale, clip, chunk,
frame) that can later inform scheduler quality estimates.

---

## 2. Scope

**In scope:**
- Constructing the Layer B real-world reference corpus.
- Computing PSNR and SSIM per frame, per model, per scale.
- Conditionally computing VMAF per clip/chunk if toolchain is available.
- Aggregating per-frame results into per-chunk and per-clip summaries.
- Producing results joinable with Step 5.5 records.

**Out of scope** (see §15):
- CPU-vs-GPU decision logic.
- Latency or resource measurement of any kind.
- Steps 5.7–5.9.
- Layer C (production-like streaming inputs).
- Content-aware scheduling or quality prediction.
- BasicVSR++ (unavailable; stub only).

---

## 3. Inputs

### 3.1 Layer B Reference Corpus

A set of short natural-video clips satisfying:

| Property | Requirement |
|---|---|
| Content | Natural video — not synthetic, not animation |
| Clip count | ≥ 3 clips, covering at least: low-motion, moderate-motion, high-motion |
| Duration per clip | 4–10 seconds |
| Source resolution | ≥ 720p (1280×720) for ×2 scale evaluation; ≥ 1080p (1920×1080) preferred |
| Frame rate | 30 FPS (primary); 60 FPS clips may be included but 30 FPS is required |
| Codec | Lossless or near-lossless source (e.g. `libx264 -crf 0`, `ffv1`, or raw) |
| License | Freely distributable for research (e.g. Derf's Test Media, MPEG test sequences, or equivalent public-domain clips) |
| SHA-256 hash | Recorded in Layer B manifest for reproducibility |

**Acceptable public sources (non-exhaustive):**
- Derf's Test Media Collection (xiph.org)
- Ultra Video Group (UVG) dataset
- REDS dataset clips (license: research use)

**Reference-frame derivation:** The uncompressed / lossless source frame is the ground-truth
(GT) reference. Do not derive GT from a re-encoded copy of the same clip.

### 3.2 Degraded Input Frames

For each reference clip and each SR scale S ∈ {2, 4} (restricted to scales supported by
the model under evaluation):

```
GT frame (H × W) ──bicubic downsample──▶ LR frame (H/S × W/S)
```

Downsampling is performed with `cv2.resize(..., interpolation=cv2.INTER_CUBIC)`. This is
the standard degradation model used by FSRCNN and Real-ESRGAN training pipelines; using a
different downsampler would invalidate comparison.

Rules:
- H and W must be divisible by S before downsampling. If not, the GT frame is
  **center-cropped** to the nearest divisible dimensions before downsampling. The crop is
  recorded in the clip manifest. The cropped GT becomes the reference for that evaluation.
- The LR frame is never separately encoded to a video container for quality evaluation;
  it is kept as an in-memory numpy array to eliminate re-encoding artifacts.

### 3.3 SR Output Frames

Each LR frame is passed through a Step 5.2 adapter:

```
LR frame ──adapter.process()──▶ SR frame (H × W)
```

The adapter is initialized with the same scale S used for downsampling. The SR output must
satisfy the existing Step 5.2 output validation contract (exact dimensions H × W) before
quality metrics are computed.

Quality metric computation is called **after** the adapter returns. The adapter's
`process()` call is NOT timed during quality evaluation runs.

---

## 4. Reference / Ground-Truth Requirements

- GT frame = center-cropped lossless source frame (after any divisibility crop, §3.2).
- LR frame = bicubic downsample of GT at scale S.
- SR frame = adapter output from LR, must be identical dimensions to GT.
- The triple (GT, LR, SR) must be pixel-aligned — no spatial offset, no temporal offset.
- GT and SR must have identical (H, W) before any metric is computed; if they differ,
  the frame is marked `invalid: "dimension_mismatch"` and skipped (see §9).

---

## 5. Metric Definitions and Computation Rules

### 5.1 PSNR

**Definition:**
$$\text{PSNR} = 10 \cdot \log_{10}\!\left(\frac{255^2}{\text{MSE}}\right)$$

**Color space:** Convert both GT and SR from BGR to **Y channel (luma) only** in YCbCr
before computing MSE. This matches the dominant SR literature convention and avoids
chroma subsampling artifacts inflating scores.

Conversion: `cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]`

**dtype:** Both Y arrays cast to `float64` before MSE computation.

**Edge case:** If MSE == 0 (perfect reconstruction), PSNR is recorded as `null` with
reason `"perfect_reconstruction"` — not as `inf` or an arbitrarily large number.

**Interpretation note:** PSNR is computed relative to 8-bit range (max=255). This is
consistent across all models and clips.

### 5.2 SSIM

**Definition:** Structural Similarity Index (Wang et al., 2004).

**Implementation:** `skimage.metrics.structural_similarity` with:
- `data_range=255`
- `channel_axis=None` (single-channel Y input)
- All other parameters at library defaults

**Color space:** Same Y-channel-only extraction as PSNR (§5.1).

**Why not multichannel SSIM:** Y-channel SSIM is standard in SR benchmarking. Multichannel
SSIM over BGR introduces chroma noise that is not meaningful for luma-sharpness evaluation.

**Range:** [−1, 1]. Values below 0 are valid (though anomalous); do not clamp.

### 5.3 VMAF

**Conditionality:** Computed only if `ffmpeg` with `libvmaf` support is available at
runtime. Detection: `ffmpeg -filters 2>&1 | grep vmaf`. If absent, all VMAF fields are
`null` with `vmaf_unavailable: true`.

**Granularity:** VMAF is computed **per chunk** (sequence of frames), not per frame.
VMAF is a temporal metric; per-frame VMAF scores exist but are less meaningful than the
clip/chunk aggregate. If the ffmpeg VMAF filter produces per-frame scores, preserve them
in `vmaf_per_frame: List[float | null]`; the chunk-level score is the mean of valid
per-frame scores.

**Color space for VMAF:** VMAF operates on YUV 4:2:0. Convert SR and GT sequences to
YUV before piping to ffmpeg. Do not pass BGR frames directly.

**Reference/distorted assignment:**
- Reference: GT sequence
- Distorted: SR sequence

**Model:** Use `vmaf_v0.6.1` (the default model). Do not use phone or 4K models unless
explicitly directed.

**VMAF is per-chunk, not per-frame** — do not store VMAF in the per-frame result schema.
Store it in the per-chunk summary schema (§11).

---

## 6. Color Space and Dimension Handling

| Operation | Rule |
|---|---|
| Adapter I/O | BGR uint8 — matches Step 5.2 contract |
| PSNR/SSIM input | Convert BGR→YCrCb, extract Y channel, cast to float64 |
| VMAF input | Convert BGR→YUV420p via ffmpeg pipe |
| Downsampling (LR gen) | cv2.INTER_CUBIC on BGR frame |
| GT crop (divisibility) | Center crop on BGR frame; record crop_h, crop_w in manifest |

No implicit color conversion is permitted. Every conversion is logged in the clip manifest.

**Dimension invariant (checked before metrics):**
```
assert gt_y.shape == sr_y.shape, "GT and SR Y-channel shapes must match before metric computation"
```
Failure → frame marked `invalid: "dimension_mismatch"`, skipped.

---

## 7. Frame Alignment Rules

- Frame index is 0-based within the chunk (representation-local, consistent with Step 2).
- GT frame at index `i` is compared only to SR frame at index `i`. No temporal offset search.
- If a chunk has N frames, exactly N (GT, LR, SR) triples are evaluated, in order.
- If the adapter returns fewer frames than the chunk contains (possible for temporal
  models with warmup), the missing frames are marked `invalid: "adapter_output_short"`.
  Do not substitute or repeat frames.

---

## 8. Preprocessing before Metric Computation

The following operations are permitted and must be recorded when applied:

| Operation | When applied | Recorded field |
|---|---|---|
| GT center crop (divisibility) | H or W not divisible by scale | `gt_crop_applied: true`, `gt_h_before`, `gt_w_before` |
| Real-ESRGAN output crop | Always potentially; §C2 of hardening pass | Already recorded in `crop_metadata` from adapter |
| BGR→Y conversion | Always, for PSNR/SSIM | Implicit; logged in clip manifest once |

**Prohibited operations:**
- Resizing SR output to match GT after the fact. If dimensions still don't match after
  the adapter's own crop, the frame is invalid.
- Any sharpening, denoising, or post-processing of GT or SR before metric computation.
- Dropping frames silently to make counts align.

---

## 9. Invalid / Anomalous Frame Handling

A frame is marked invalid when any of the following occur:

| Condition | `invalid_reason` value |
|---|---|
| GT and SR Y-channel shapes differ | `"dimension_mismatch"` |
| Adapter raised an exception for this frame | `"adapter_exception"` |
| Adapter returned fewer frames than expected | `"adapter_output_short"` |
| MSE == 0 (PSNR infinite) | Not invalid; PSNR recorded as `null`, reason `"perfect_reconstruction"` |
| SSIM computation raises exception | `"ssim_computation_error"` |

Invalid frames are **excluded** from aggregate statistics. The count of invalid frames is
reported in the per-chunk summary as `invalid_frame_count`. If more than 20% of frames in
a chunk are invalid, the chunk-level aggregate is marked `aggregate_valid: false` — the
raw per-frame records are still preserved.

---

## 10. Per-Frame Result Schema

One record per (model_id, scale, clip_id, chunk_id, frame_index):

```json
{
  "model_id":        "tinysr | tinysr_int8 | real_esrgan",
  "scale":           2,
  "device":          "cpu | cuda",
  "clip_id":         "clip_001_lowmotion_30fps",
  "chunk_id":        "chunk_000",
  "frame_index":     0,
  "gt_shape":        [360, 640],
  "sr_shape":        [360, 640],
  "psnr_db":         32.14,
  "ssim":            0.891,
  "invalid":         false,
  "invalid_reason":  null,
  "crop_metadata":   { ... }
}
```

`crop_metadata` is copied from the adapter's output if `crop_applied == true`; otherwise
`null`.

`device` is recorded to allow future split of quality-by-device (quantization can affect
output, so CPU vs GPU results may legitimately differ for INT8 models).

---

## 11. Per-Chunk and Per-Clip Aggregate Schema

### Per-chunk aggregate

Computed over valid frames only:

```json
{
  "model_id":             "tinysr",
  "scale":                2,
  "device":               "cpu",
  "clip_id":              "clip_001_lowmotion_30fps",
  "chunk_id":             "chunk_000",
  "frame_count":          60,
  "invalid_frame_count":  0,
  "aggregate_valid":      true,
  "psnr_mean":            32.14,
  "psnr_median":          32.20,
  "psnr_min":             28.50,
  "psnr_max":             35.10,
  "psnr_stdev":           1.22,
  "ssim_mean":            0.891,
  "ssim_median":          0.893,
  "ssim_min":             0.831,
  "ssim_max":             0.921,
  "ssim_stdev":           0.018,
  "vmaf_mean":            72.3,
  "vmaf_per_frame":       [71.1, 72.8, ...],
  "vmaf_unavailable":     false
}
```

**Why mean and median, not p95:** Quality metrics are not tail-risk metrics. Mean reflects
average delivered quality; median is robust to occasional frame anomalies. p95 has no
meaningful interpretation for quality scores (high p95 SSIM is not a concern; low scores
are). Min is reported to surface worst-case frames. p95 latency methodology does not
transfer here.

### Per-clip aggregate

Computed by aggregating per-chunk means:

```json
{
  "model_id":      "tinysr",
  "scale":         2,
  "device":        "cpu",
  "clip_id":       "clip_001_lowmotion_30fps",
  "chunk_count":   2,
  "psnr_mean":     32.07,
  "ssim_mean":     0.889,
  "vmaf_mean":     72.1,
  "vmaf_unavailable": false
}
```

Per-clip aggregation is a simple mean of per-chunk means (not a frame-weighted mean),
because chunks are equal-duration by construction.

---

## 12. Integration with Step 5.5 Results

Join key: `(model_id, scale, device, clip_id, chunk_id)`

- `clip_id` in Step 5.6 maps to `benchmark_video_id` in Step 5.5 by convention:
  Layer B clips are named `clip_NNN_<motion>_<fps>fps` and their `benchmark_video_id`
  field in the Layer B manifest uses the same string.
- `chunk_id` is shared — Step 5.1's chunking pipeline produces the authoritative chunk IDs.
- `frame_index` is a Step 5.6 addition with no Step 5.5 counterpart (Step 5.5 does not
  record per-frame data). Per-frame quality records cannot be joined to per-trial latency
  records at the frame level; they can only be joined at the chunk level.

**Step 5.5 latency records are not modified.** Quality evaluation is additive.

---

## 13. Reproducibility Requirements

- Layer B manifest records SHA-256 hash of every source clip.
- All downsampling parameters (interpolation method, scale, crop dimensions) are recorded
  per clip in the manifest.
- Adapter model weights are identified by SHA-256 hash of the `.pth`/`.onnx` file,
  consistent with Step 5.2.
- VMAF model version string (`vmaf_v0.6.1`) is recorded in every chunk result that uses
  VMAF.
- `skimage` version is recorded in the run metadata (SSIM implementation details differ
  across versions).

---

## 14. Tests

| Test | GPU required? |
|---|---|
| GT crop applied correctly for non-divisible resolution | No |
| LR dimensions = GT / scale after downsample | No |
| SR dimensions = GT after adapter call | No |
| PSNR == null when MSE == 0, not inf | No |
| PSNR/SSIM computed on Y channel, not BGR | No |
| Dimension mismatch → frame marked invalid, excluded from aggregate | No |
| > 20% invalid frames → aggregate_valid = false | No |
| Per-frame records preserved even when aggregate_valid = false | No |
| VMAF fields null when ffmpeg libvmaf absent | No |
| Join key fields present and consistent with Step 5.1 chunk IDs | No |
| Real FSRCNN inference on one Layer B clip, PSNR > 25 dB at scale 2 | No (CPU) |
| crop_metadata propagated from adapter to per-frame record | No |

All tests must pass without a GPU. GPU-path quality tests (Real-ESRGAN CUDA) are skipped
with `pytest.mark.skipif` if CUDA is unavailable.

---

## 15. Acceptance Criteria

Step 5.6 is complete when:
1. Layer B manifest exists with ≥ 3 clips and all SHA-256 hashes verified.
2. Per-frame quality records exist for all (model, scale, clip, chunk, frame) combinations
   where the model is available and the scale is supported.
3. Per-chunk and per-clip aggregate records exist and are consistent with per-frame records.
4. VMAF fields are either populated or explicitly `null` with `vmaf_unavailable: true`.
5. No PSNR value is `inf` or `nan` in any persisted record.
6. All tests in §14 pass.
7. Step 5.5 test suite (331 tests) remains green — no regressions.

---

## 16. Known Limitations

- **VMAF toolchain not guaranteed.** If `libvmaf` is absent, VMAF results are missing
  entirely. This is documented explicitly, not silently omitted.
- **Bicubic degradation model.** FSRCNN and Real-ESRGAN were trained on bicubic
  degradation. This evaluation therefore measures in-distribution quality. Real streaming
  degradation (compression artifacts, transmission noise) is not modeled here.
- **No temporal consistency metric.** PSNR and SSIM are per-frame. Temporal flickering
  between frames from spatial-only models (FSRCNN, Real-ESRGAN) is not captured. VMAF
  partially accounts for this. A dedicated temporal metric is out of scope for 5.6.
- **BasicVSR++ excluded.** The adapter is a non-available stub. Quality evaluation for
  BasicVSR++ is deferred until the model is deployable.
- **Layer A synthetic corpus not used for quality conclusions.** Any PSNR/SSIM computed
  over Layer A frames is methodologically invalid for model comparison and must not appear
  in any quality report.
- **Y-channel only.** Chroma quality is not measured. This matches SR literature convention
  but means color fringing artifacts from SR are not captured.

---

## 17. Out-of-Scope Items

- CPU-vs-GPU decision logic (any step).
- Content-aware scheduling or quality prediction.
- Steps 5.7–5.9.
- Layer C production-like streaming inputs.
- Modifying Steps 0–5.5 implementations.
- Training or fine-tuning SR models.
- Perceptual loss metrics (LPIPS) — existing `compute_quality_metrics.py` computes LPIPS
  but it is not included here; LPIPS requires a neural network pass and its cost must be
  separately justified.
- Psychovisual user studies.