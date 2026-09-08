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

### Representation-Local Frame Ranges
Because representations run at different framerates, the number of encoded frames for the same logical chunk differs between representations. Therefore, **frame indices are representation-local**, referring to the frame indices inside that specific representation's encoded stream rather than source-global frame indices.

```text
Same Logical Time Range
         ↓
   ┌───────────────┐
   │ chunk_007     │ (Authoritative logical interval)
   │ 14s → 16s     │ (Logical duration = 2.0s)
   └───────┬───────┘
           │
      ┌────┴────┐
      ▼         ▼
   720p@60   720p@30
   ~120 fr   ~60 fr   (Different representation-local frame counts)
```

---

## 4. Mapping Invariant Rules

The mapping manager validates the following **11 strict validation invariants**:
1. **Full Coverage**: Every logical chunk maps to every configured representation.
2. **No Duplicates**: No duplicate `(representation_id, chunk_id)` pairs are allowed.
3. **Logical Temporal Boundaries**: Logical start/end timestamps and durations must be identical across representations and match the Step 1 authoritative timeline.
4. **Representation-Local Consistency**: Representation-local frame counts must match the expected frame count derived from the logical duration and materialized representation FPS:
   $$\text{local\_frame\_count} == \text{round}(\text{logical\_duration} \times \text{rep\_fps})$$
5. **Ordered Local Range**: Each representation's local frame indices start at frame 0, sort monotonically, and have no gaps or overlaps.
6. **Sanity Ranges**: `frame_start <= frame_end` and `start_time_seconds <= end_time_seconds`.
7. **Positive Duration**: `duration_seconds > 0.0`.
8. **Chronological Timestamps**: Logical chunk timestamps contain no temporal gaps or overlaps.
9. **First Chunk Bound**: The first logical chunk starts at time 0.0s.
10. **Final Chunk Bound**: The final logical chunk ends at the source timeline end.
11. **Config Existence**: Mapped representation IDs must exist in the representation config.

---

## 5. Deferred Audio Scope

Audio representation fields are explicitly **deferred** from the representation schema. The current contract covers only the video streaming path used by the scheduling and enhancement components. Existing Step 0 `has_audio` metadata remains completely unaffected.

---

## 6. Future Validation: Size/Bitrate Sanity Check (Step 2.3 TODO)
> [!TIP]
> **Step 2.3 TODO**: Once actual physical video encoding ladders exist (Step 2.3), we should implement a soft sanity check comparing:
> $$\text{size\_bytes} \approx \frac{\text{bitrate\_kbps} \times 1000 \times \text{duration\_seconds}}{8}$$
> This should produce warning alerts on deviation rather than rejecting the mapping, to accommodate encoding bitrate variability.

---

## 7. Automated Testing

* Video representation validation is tested in [`tests/test_representation.py`](file:///d:/Full%20Stack/AdaptiveSR/tests/test_representation.py).
* Mapping invariants are tested in [`tests/test_mapping.py`](file:///d:/Full%20Stack/AdaptiveSR/tests/test_mapping.py) verifying:
  * Full coverage and missing chunk checks.
  * Local frame count checks on 30, 60, and 120 FPS mappings.
  * Validating combinations of source FPS and representation FPS (such as 60 FPS source + 30 FPS rep).
  * Variable final chunk duration acceptance.
