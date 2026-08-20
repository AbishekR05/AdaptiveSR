# Step 2 — Video Representation and Mapping Contract

This document describes the implementation of video representations and chunk-to-representation mapping (Step 2.1 and Step 2.2) for the AdaptiveSR project.

---

## 1. Purpose & Objectives

Step 2 establishes the delivery data contract linking source-side visual profiling results (Step 1) with delivery representations served by Cloud Origin and processed by Edge Nodes:
* **Step 2.1 (Representations)** defines the data model for the set of pre-encoded video qualities (resolutions, bitrates, codecs, framerates).
* **Step 2.2 (Mapping)** maps the authoritative logical chunks from the profiling timeline to representation-specific metadata objects.

---

## 2. Representation Schema Details (Step 2.1)

We implemented `VideoRepresentation` and `RepresentationConfig` in [`adaptive_sr/shared/schemas.py`](file:///d:/Full%20Stack/AdaptiveSR/adaptive_sr/shared/schemas.py) using Pydantic:

### `VideoRepresentation`
Defines the metrics and identifiers of a single variant:
* **`representation_id`** (`str`): Unique identifier (e.g. `"360p"`, `"720p_h265"`).
* **`width`** (`int`): Target spatial width (must be $> 0$).
* **`height`** (`int`): Target spatial height (must be $> 0$).
* **`resolution_label`** (`str`): Friendly name descriptor (e.g. `"360p"`).
* **`bitrate_kbps`** (`int`): Encoded target bitrate in kbps (must be $> 0$).
* **`codec`** (`str`): Video compression format (e.g. `"h264"`, `"h265"`, `"vp9"`, `"av1"`).
* **`fps`** (`Union[int, Literal["source"]]`): Framerate value, which can either be a positive integer (e.g. `30`, `60`, `120`) or the string literal `"source"`.

### `RepresentationConfig`
Contains a list of `VideoRepresentation` instances to configure the entire set of available profiles.

#### Validation Rules
1. **Unique Representation IDs**: Duplicate IDs inside the same config are rejected.
2. **Positive Resolution & Bitrate**: Resolving dimensions and bandwidth constraints must be positive.
3. **Supported Codec**: Checks standard formats (`h264`, `h265`, `hevc`, `vp9`, `av1`).
4. **Valid FPS**: Restricted to `30`, `60`, `120` or `"source"`.
   > [!NOTE]
   > **FPS Scope Statement**: 30/60/120 FPS are the project's evaluated source/delivery FPS scenarios. This is not a universal streaming limitation.
5. **No Duplicate Materialized Variants**: 
   * Pre-materialization: Unique `(width, height, fps)` tuples. Identical resolutions are accepted if framerates are different (e.g. `720p @ 30` and `720p @ 60`).
   * Post-materialization: Resolving `"source"` must not cause overlapping conflicts. If resolving `"source"` at `60` results in two identical `(width, height, 60)` variants, it is rejected.

---

## 3. Chunk-to-Representation Mapping (Step 2.2)

We implemented `RepresentationChunk` and `RepresentationChunkMapping` in [`adaptive_sr/shared/schemas.py`](file:///d:/Full%20Stack/AdaptiveSR/adaptive_sr/shared/schemas.py):

### Concept Separation
* **Logical Chunk**: The authoritative temporal unit defined in Step 1. It acts as the master timeline. All representations inherit this logical timeline.
* **Representation Chunk**: The representation-specific metadata mapping indicating the actual file path, file size in bytes, and referencing the logical boundaries.

### `RepresentationChunk` Model
Represents a chunk mapped to a representation:
* `chunk_id` (`str`): Master chunk ID matching Step 1.
* `representation_id` (`str`): Mapped representation identifier.
* `frame_start` (`int`), `frame_end` (`int`): Frame bounds derived from Step 1.
* `start_time_seconds` (`float`), `end_time_seconds` (`float`): Logical timestamps.
* `duration_seconds` (`float`): Actual logical duration.
* `file_path` (`str`): Path to the encoded representation chunk file.
* `size_bytes` (`int`): Encoded file size in bytes.

---

## 4. Mapping Invariant Rules

The mapping manager validates the following **12 strict validation invariants**:
1. **Full Coverage**: Every logical chunk maps to every configured representation.
2. **No Duplicates**: No duplicate `(representation_id, chunk_id)` pairs are allowed.
3. **Identical Frames**: Frame ranges must be identical across representations.
4. **Identical Timestamps**: Logical start/end timestamps and durations must be identical across representations.
5. **Sanity Ranges**: `frame_start <= frame_end` and `start_time_seconds <= end_time_seconds`.
6. **Positive Duration**: `duration_seconds > 0.0`.
7. **Monotonic Order**: Chunk indexing must be sorted chronologically and monotonically.
8. **No Gaps**: Frame sequences must contain no gaps between consecutive chunks.
9. **No Overlaps**: Frame sequences must contain no overlaps between chunks.
10. **First Chunk Bound**: The first chunk starts exactly at frame 0.
11. **Final Chunk Bound**: The final chunk ends exactly at `frame_count - 1` of the source.
12. **Config Existence**: Mapped representation IDs must exist in the configuration.

---

## 5. Deferred Audio Scope

Audio representation fields are explicitly **deferred** from the representation schema. The current contract covers only the video streaming path used by the scheduling and enhancement components. Existing Step 0 `has_audio` metadata remains completely unaffected.

---

## 6. Non-Scheduling Policy

To maintain clear boundary separation, the representation schema strictly defines **what exists** on the source/delivery servers. It does not contain scheduling/adaptive state variables such as `target_representation_id` or `base_representation_id`.

---

## 7. Automated Testing

* Video representation validation is tested in [`tests/test_representation.py`](file:///d:/Full%20Stack/AdaptiveSR/tests/test_representation.py).
* Mapping invariants are tested in [`tests/test_mapping.py`](file:///d:/Full%20Stack/AdaptiveSR/tests/test_mapping.py) verifying:
  * Full coverage and missing chunks detection.
  * Collision detection for duplicate mapping pairs.
  * Chronological sorting, gaps, and overlaps errors.
  * Correct frame scaling on 30, 60, and 120 FPS timelines.
  * Acceptance of variable final chunk duration (proving no hardcoded 2.0s duration assumptions).
