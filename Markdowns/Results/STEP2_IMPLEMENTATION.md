# Step 2.1 — Representation Schema and Configuration Contract

This document describes the implementation of the representation schema and configuration contract (Step 2.1) for the AdaptiveSR project.

---

## 1. Purpose & Objectives

Step 2.1 establishes the data models and schemas that define multiple encoded video representations. A video representation represents a specific encoded variant of the source video (with configured resolution, bitrate, codec, and frame rate). These models serve as the foundational configuration contract shared by the Cloud Origin, Edge Servers, Client player, and downstream scheduling modules.

---

## 2. Representation Schema Details

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

---

## 3. Validation Rules

The schema enforces strict validation checks:
1. **Unique Representation IDs**: Duplicate `representation_id` entries inside the same config are rejected.
2. **Positive Resolution**: Width and height must be strictly positive integers (`gt=0`).
3. **Positive Bitrate**: Bitrate in kbps must be strictly positive (`gt=0`).
4. **Supported Codec**: Codec string must map to supported standards (case-insensitive: `h264`, `h265`, `hevc`, `vp9`, `av1`).
5. **Valid FPS**: Framerate must be either `30`, `60`, `120` or the literal string `"source"`.
   > [!NOTE]
   > **FPS Scope Statement**: 30/60/120 FPS are the project's evaluated source/delivery FPS scenarios. This is not a universal streaming limitation.
6. **No Duplicate Materialized Variants**: 
   * Pre-materialization: Representations must not have identical `(width, height, fps)` configurations. We permit multiple configurations with the same resolution (e.g., `720p @ 30` and `720p @ 60`) to coexist because framerate is an independent variant dimension.
   * Post-materialization: Materialized configurations must not contain duplicate resolved `(width, height, fps)` variants.

---

## 4. FPS Materialization & late-binding collision checking

If `"fps": "source"` is configured, the model supports lazy resolution:
* Calling `representation.materialize(source_fps=X)` returns a new `VideoRepresentation` instance where `"source"` is resolved to the actual integer framerate `X` of the source video.
* Calling `config.materialize(source_fps=X)` materializes all entries in the config.
* **Late-binding duplicate check**: The materialization method dynamically validates that resolving `"source"` does not produce a variant conflict. For example, if a configuration has both `720p @ source` and `720p @ 60`, materializing with `source_fps=60` resolves both to `720p @ 60`, which is correctly caught and rejected with a `ValueError`. If materialized with `source_fps=30`, both remain distinct and are accepted.

---

## 5. Deferred Audio Scope

Audio representation fields are explicitly **deferred** from the Step 2.1 schema. The current representation configuration contract only covers the video streaming paths used by the AdaptiveSR scheduler. Step 0's existing `has_audio` video metadata field remains completely unaffected.

---

## 6. Non-Scheduling Policy

To maintain clear boundary separation, the representation schema strictly defines **what exists** on the source/delivery servers. It does not contain scheduling/adaptive state variables such as `target_representation_id` or `base_representation_id`.

---

## 7. Automated Testing

Tests are implemented in [`tests/test_representation.py`](file:///d:/Full%20Stack/AdaptiveSR/tests/test_representation.py) verifying:
* Valid configuration parses correctly.
* Duplicate IDs or identical `(width, height, fps)` variants are rejected.
* Same resolution at different FPS are accepted.
* Source FPS materialization works.
* Resolving source FPS to a colliding explicit FPS causes a ValueError.
* Target/base selection remains uncoupled.
