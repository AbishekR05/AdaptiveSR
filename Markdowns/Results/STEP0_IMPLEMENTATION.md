# Step 0 — Distributed Foundation Implementation

This document describes the implementation of the minimum working distributed prototype (Step 0) and the Step 0.1 foundation hardening pass for the AdaptiveSR project.

The system uses actual network communication (HTTP REST) between three independently running processes/services running on localhost. This represents a **local distributed emulation of the Cloud → Edge → Client architecture** before final cloud deployment.

---

## 1. Architecture Overview

```text
       [ CLIENT ]
           │
           │ HTTP (GET)
           ▼
     [ EDGE SERVER ]
           │
           │ HTTP (GET) [on Cache Miss]
           ▼
     [ CLOUD ORIGIN ]
```

* **Cloud Origin** (Port 8000): Serves as the central repository for video assets, exposes available representations, manifests, and serves raw video segment chunks.
* **Edge Server** (Port 8001 / Port 8003): Intercepts client segment requests, manages a local disk cache of segments, requests missing segments from the Cloud Origin over HTTP, caches them, and streams segments back to the client.
* **Client Player** (Port 8002 / Client CLI): Emulates player client playback, downloads segments sequentially, maintains a mathematical playback buffer, calculates download throughput, and generates local playback telemetry.

> [!NOTE]
> **Representation Mapping Model**: Currently, the requested representation matches the delivered representation. However, the data schema is designed to distinguish `target_representation_id` and `base_representation_id`. In future steps involving Super-Resolution (SR) integration, a scheme where `target_representation_id != base_representation_id` may be supported (where the base resolution is upscaled by the Edge to match the target representation quality).

---

## 2. Shared Schemas (`adaptive_sr/shared/schemas.py`)

Data structures are defined using Pydantic (Pydantic v2):
* `Representation`: Profiles video qualities (resolution, target resolution, and bitrate).
* `ChunkMetadata`: Profiles chunk properties (chunk ID and chunk duration in seconds).
* `VideoManifest`: Holds video metadata, available representations, and chunk sequence lists.
* `ChunkRequest`: Formats the payload, including future-proof fields for `target_representation_id` and `base_representation_id`.
* `ClientTelemetry`: Logs download times, RTT, buffer states, cache status, stall counts, and `stall_duration_seconds`.
* `EdgeTelemetry`: Logs chunk processing latency, cloud fetch overhead, cache state, request IDs, Edge-to-Cloud RTT, and bytes transferred.

---

## 3. API Endpoints

### Cloud Origin Service
* `GET /health`: Returns service status.
* `GET /videos`: Returns a list of available video IDs on disk.
* `GET /videos/{video_id}/manifest`: Serves the video JSON manifest.
* `GET /videos/{video_id}/{representation_id}/chunks/{chunk_id}`: Streams the raw segment file.

### Edge Server Service
* `GET /health`: Returns service metadata, including `cluster_id` and `edge_id`.
* `GET /videos/{video_id}/manifest`: Proxies manifest request to the Cloud.
* `GET /videos/{video_id}/chunks/{chunk_id}?representation_id=...`: Retrieves the chunk. If the chunk is cached, serves it immediately from disk (Cache HIT). Otherwise, fetches it from Cloud over HTTP first, caches it on disk, and serves it (Cache MISS). Telemetry headers are injected:
  * `X-Request-ID`: Trace ID for the request.
  * `X-Cache`: `HIT` or `MISS`.
  * `X-Cloud-Fetch-Time`: Cloud fetch latency in seconds.
  * `X-Edge-Processing-Time`: Edge processing latency in seconds.
  * `X-Cluster-ID` / `X-Edge-ID`: Identity of the responding node.
  * `X-Edge-Cloud-RTT`: Active Edge-to-Cloud round-trip time in seconds.

---

## 4. Playback Buffer Model

The Client Player maintains a mathematical playback buffer to simulate stream playout:
* **Depletion**: While a chunk is downloading, simulated playback consumes the buffer:
  $$\text{buffer\_seconds} = \max(0.0, \text{buffer\_seconds} - \text{download\_duration})$$
* **Stalls**: If `buffer_seconds` reaches $0.0$ during download, a stall event is registered.
* **Stall Duration**: The duration of the playback stall (seconds) for each chunk request is calculated as:
  $$\text{stall\_duration\_seconds} = \max(0.0, \text{download\_duration} - \text{buffer\_before})$$
* **Replenishment**: Once download completes successfully, the buffer is refilled:
  $$\text{buffer\_seconds} += \text{chunk\_duration}$$

> [!NOTE]
> The current buffer model intentionally assumes serial chunk download/playback for foundation testing. Future streaming steps may introduce concurrent prefetch and therefore require a more realistic buffer evolution model.

---

## 5. RTT Ping Latency Tracking

To measure network round-trip time (RTT) independently of video segment download sizes, both the Client and Edge servers implement active health pings:
* **Client-to-Edge RTT**: Right before requesting a video chunk, the Client queries the Edge Server's `GET /health` endpoint. The latency is measured using a monotonic clock and logged in telemetry as `RTT`.
* **Edge-to-Cloud RTT**: Right before serving or fetching a video chunk, the Edge queries the Cloud Origin's `GET /health` endpoint. The RTT is measured using a monotonic clock, included in the Edge telemetry log as `rtt`, and returned to the client in the response header `X-Edge-Cloud-RTT`.

---

## 6. Disk Caching (`adaptive_sr/services/edge/cache.py`)

A simple file cache manager storing segment files on disk at `adaptive_sr/services/edge/cache/` (or configuration directory).
* **Key structure**: `{video_id}__{chunk_id}__{representation_id}.mp4`
* **Features**:
  * `get(key)`: Returns path if cached.
  * `put(key, bytes)`: Writes segment payload to cache.
  * `contains(key)`: Fast boolean existence check.

---

## 7. How to Run the Services

You can start each service in a separate terminal shell:

### Cloud Server (runs on Port 8000)
```powershell
python -m adaptive_sr.services.cloud.app
```

### Running Multiple Edge Nodes
The architecture supports running multiple Edge nodes distinguished by identity variables:

```powershell
# Shell A: Start Edge Node 1 (runs on Port 8001)
$env:EDGE_PORT="8001"
$env:EDGE_ID="edge_01"
$env:EDGE_CACHE_DIR="./adaptive_sr/services/edge/cache_edge_01"
python -m adaptive_sr.services.edge.app

# Shell B: Start Edge Node 2 (runs on Port 8003)
$env:EDGE_PORT="8003"
$env:EDGE_ID="edge_02"
$env:EDGE_CACHE_DIR="./adaptive_sr/services/edge/cache_edge_02"
python -m adaptive_sr.services.edge.app
```

### Running the Player Simulation
```powershell
# Target Edge Node 1 (Port 8001)
python -m adaptive_sr.services.client.app --video sample --repr 360p --edge http://localhost:8001

# Target Edge Node 2 (Port 8003)
python -m adaptive_sr.services.client.app --video sample --repr 360p --edge http://localhost:8003
```

---

## 8. How to Test

Run the automated Pytest suite:
```powershell
python -m pytest tests/test_foundation.py -v
```

This verifies:
1. Cloud health and manifest endpoints.
2. Direct chunk streaming.
3. Edge proxy routing and multiple Edge server identities.
4. Edge cache miss-to-hit transition sequence.
5. End-to-end data boundary validation.
6. Playback buffer depletion, throughput, RTT pings, and stall duration math.
7. Zero-buffer playback stall duration.
8. Per-edge cache directory isolation.

---

## 9. Success Verification Metrics

### First Run (Cache MISS)
The Edge downloads the chunk from the Cloud Origin. Edge telemetry shows a cache miss, Edge-to-Cloud RTT, and records the Cloud fetch latency:
```json
{"event": "edge_telemetry", "cluster_id": "cluster_01", "edge_id": "edge_01", "telemetry": {"request_id": "35975dd9-340e-4a0b-95ab-ad594e02bec0", "video_id": "sample", "chunk_id": "0000", "representation_id": "360p", "target_representation_id": "360p", "base_representation_id": "360p", "cache_hit": false, "cloud_fetch_time": 2.0309, "edge_processing_time": 2.0470, "response_time": 2.0470, "bytes_sent": 61188, "rtt": 0.0012, "sr_processing_time": 0.0}}
```

### Second Run (Cache HIT)
The Edge serves the chunk directly from local disk. Cloud fetch latency is zero, RTT is logged, and processing time drops close to zero:
```json
{"event": "edge_telemetry", "cluster_id": "cluster_01", "edge_id": "edge_01", "telemetry": {"request_id": "1dc6ade1-62f6-4b66-8127-3566aa4dd293", "video_id": "sample", "chunk_id": "0000", "representation_id": "360p", "target_representation_id": "360p", "base_representation_id": "360p", "cache_hit": true, "cloud_fetch_time": 0.0, "edge_processing_time": 0.0, "response_time": 0.0, "bytes_sent": 61188, "rtt": 0.0010, "sr_processing_time": 0.0}}
```

---

## 10. Next Steps (Step 1 Roadmap)

In Step 1, we will build upon this foundation to introduce:
* **Super-Resolution Inference**: Deploying FSRCNN/Real-ESRGAN upsamplers inside the Edge Server segment routing pipeline.
* **CPU Core Allocation**: Pinning server processing threads to specific core allocations as modeled in the Rosevin paper.
* **Adaptive Bitrate Selection**: Client-side logic that selects target bitrates dynamically depending on telemetry.
