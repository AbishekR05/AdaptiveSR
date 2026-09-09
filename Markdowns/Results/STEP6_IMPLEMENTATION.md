# STEP 6 — REMOTE SR INFERENCE AT THE EDGE
## IMPLEMENTATION DOCUMENTATION

### 1. Objective & Goal
Step 6 integrates the Super-Resolution (SR) inference engine developed in Step 5 into the live Cloud $\rightarrow$ Edge $\rightarrow$ Client runtime architecture built in Steps 0–4.
It enables the Edge server to receive SR chunk requests, obtain source representations (from cache or Cloud origin), execute SR model inference locally at the Edge, and serve the enhanced video chunk back to the client.

> [!IMPORTANT]
> **Edge Execution Boundary & Frozen Rules:**
> - SR execution occurs strictly at the Edge (never at the Client).
> - Reuses the existing Step 5 SR adapter contract (`get_adapter(model_id)`).
> - Silently falling back from GPU to CPU is **prohibited**. If CUDA execution is requested and CUDA is unavailable, the Edge explicitly fails the SR request (`sr_status: "failed"`).
> - Model selection, adaptive bitrate (ABR), FPS adaptation, edge scheduling, and the decision engine remain OUT OF SCOPE.
> - Steps 0 through 5.9 remain 100% frozen.

---

### 2. Runtime Flow Architecture

$$\begin{array}{ccccccc}
\text{CLIENT} & \xrightarrow{\text{GET /chunks/chunk\_001?sr\_requested=true\&model\_id=tinysr}} & \text{EDGE SERVER} \\
& & \downarrow \text{Fetch/Cache Base Chunk} \\
& & \text{CLOUD ORIGIN} \quad (\text{if Cache MISS}) \\
& & \downarrow \text{Source Chunk} \\
& & \text{SR INFERENCE ENGINE} \quad (\text{Edge GPU / CPU}) \\
& & \downarrow \text{Enhanced Chunk (x2 / x4)} \\
\text{CLIENT} & \xleftarrow{\text{FileResponse + HTTP Headers (X-SR-Status: executed)}} & \text{EDGE SERVER}
\end{array}$$

---

### 3. Edge Endpoint API Specification

#### `GET /videos/{video_id}/chunks/{chunk_id}`
Query Parameters:
- `representation_id`: (Required, `str`) Target base video quality representation (e.g. `480p`).
- `sr_requested`: (Optional, `bool`, default `false`) Whether SR enhancement is requested.
- `model_id`: (Optional, `str`) SR model identifier (`tinysr`, `real_esrgan`, `tinysr_int8`).
- `scale`: (Optional, `int`, default `2`) Scaling factor (`2` or `4`).
- `device`: (Optional, `str`) Target device (`cpu` or `cuda`). Defaults to `cuda` if GPU available, else `cpu`.

#### HTTP Response Headers
- `X-Request-ID`: Unique UUID string for request tracing.
- `X-Cache`: `HIT` or `MISS`.
- `X-Edge-Processing-Time`: Total processing time at Edge in seconds.
- `X-SR-Requested`: `True` or `False`.
- `X-SR-Status`: `executed`, `failed`, or `not_requested`.
- `X-SR-Model`: Model ID used (`tinysr`, `real_esrgan`, etc.) or `N/A`.
- `X-SR-Scale`: Scaling factor (`2`, `4`) or `N/A`.
- `X-SR-Device`: Target device (`cpu`, `cuda`) or `N/A`.
- `X-SR-Processing-Time`: Measured SR inference execution time in seconds.

---

### 4. Telemetry Logging Schema
Structured JSON telemetry emitted by Edge server:
```json
{
    "event": "edge_telemetry",
    "cluster_id": "cluster-local",
    "edge_id": "edge-01",
    "telemetry": {
        "request_id": "d9f8e7d6-...",
        "video_id": "test_vid",
        "chunk_id": "chunk_001",
        "representation_id": "480p",
        "cache_hit": true,
        "edge_processing_time": 0.045123,
        "sr_requested": true,
        "sr_status": "executed",
        "sr_model_id": "tinysr",
        "sr_scale": 2,
        "sr_device": "cuda",
        "sr_processing_time": 0.041029,
        "input_resolution": [240, 320],
        "output_resolution": [480, 640]
    }
}
```

---

### 5. Non-Fallback & Failure Handling Rules
1. **GPU Safety**: If `device="cuda"` is requested and CUDA is not available on the Edge host, the Edge logs an error and returns `500 Internal Server Error` with `X-SR-Status: failed`. No silent CPU fallback occurs.
2. **Invalid Model**: If an unregistered `model_id` is requested, the Edge returns `400 Bad Request`.
3. **Pipeline Protection**: Failures during SR processing do not corrupt base video chunks or non-SR streaming endpoints.

---

### 6. Verification & Test Results
Run focused Step 6 test suite:
```bash
python -m pytest tests/test_remote_sr.py -v
python -m pytest tests/ -v
```
All Step 6 unit and integration tests pass, and the full regression suite remains passing without breaking changes.
