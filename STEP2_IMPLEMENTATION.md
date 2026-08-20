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
6. **No Duplicate Resolutions**: Multiple representations with identical spatial dimensions (width and height) inside the same config are rejected to prevent representation configuration conflicts.

---

## 4. FPS Materialization Contract

If `"fps": "source"` is configured, the model supports lazy resolution:
* Calling `representation.materialize(source_fps=X)` returns a new `VideoRepresentation` instance where `"source"` is resolved to the actual integer framerate `X` of the source video.
* If a fixed integer FPS is specified (e.g. `60`), it is preserved during materialization.

---

## 5. Non-Scheduling Policy

To maintain clear boundary separation, the representation schema strictly defines **what exists** on the source/delivery servers. It does not contain scheduling/adaptive state variables such as `target_representation_id` or `base_representation_id`.

---

## 6. Automated Testing

Tests are implemented in [`tests/test_representation.py`](file:///d:/Full%20Stack/AdaptiveSR/tests/test_representation.py) verifying:
* Valid configuration parses correctly.
* Duplicate IDs or resolutions are caught and rejected.
* Out-of-bounds resolutions, bitrates, and codecs are blocked.
* Source FPS materialization works.
* Target/base selection remains uncoupled.
