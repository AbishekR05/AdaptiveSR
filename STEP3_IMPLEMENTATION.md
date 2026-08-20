# Step 3 — Network Measurement Contract

This document describes the implementation of the network measurement contract (Step 3.1) for the AdaptiveSR project.

---

## 1. Architecture Diagram

The AdaptiveSR project treats network communication as two independent paths:

```text
CLIENT
   │
   │ client_edge path (measured RTT + transfer speed)
   ▼
 EDGE
   │
   │ edge_cloud path (measured RTT + transfer speed)
   ▼
 CLOUD / ORIGIN
```

These communication paths must not be collapsed into a single latency or throughput variable.

---

## 2. Network Path Identities

We define two explicit path keys:
* **`client_edge`**: Spans the Client ↔ Edge Node path.
* **`edge_cloud`**: Spans the Edge Node ↔ Cloud Origin path.

Persisted measurements are traceable to the corresponding path identity.

---

## 3. RTT Definition

* **Independent Measurement**: RTT (Round Trip Time) is a separate network characteristic and is **never** calculated from `chunk_size / throughput` or request transfer durations.
* **Probing Mechanism**: Measured independently using lightweight health pings (`GET /health` requests) that carry no payload.
* **Established telemetry**:
  * Client-to-Edge RTT is tracked via the client player health ping queries (captured in `ClientTelemetry.RTT` or `NetworkMeasurement.rtt_ms`).
  * Edge-to-Cloud RTT is tracked via active edge node queries (captured in `EdgeTelemetry.rtt` or `NetworkMeasurement.rtt_ms`).
* **Probe Timings**: Ping measurements represent the round-trip latency in milliseconds. Probes run independently of chunk payload transfer, meaning independent RTT probes do not fabricate or carry fake `chunk_id` attributes.

---

## 4. Throughput & Payload Transfer Durations

* **Payload Transfer Duration (`transfer_duration_seconds`)**: Represents the time required to transfer the payload bytes, excluding initial RTT pings.
* **Throughput Calculation**: Throughput in Mbps (Megabits per second) is computed as:
  $$\text{Throughput (Mbps)} = \frac{\text{Bytes Transferred} \times 8}{\text{Transfer Duration (Seconds)} \times 1,000,000}$$
* **Validation Constraint**: Zero-byte RTT probes do not produce a fake or non-zero throughput speed. If bytes transferred equals 0 or is absent, throughput must be absent.

---

## 5. Cache Semantics (HIT vs. MISS)

The measurement contract must distinguish Cache HITs from Cache MISSes to properly trace traffic origins:
* **Cache HIT**: 
  * The `client_edge` payload transfer exists.
  * The `edge_cloud` payload transfer is **absent** (since the chunk is served locally from the Edge's cache directory). We do not represent this absent transfer as zero-speed or zero-duration network traffic.
* **Cache MISS**:
  * The `client_edge` payload transfer exists.
  * The `edge_cloud` payload transfer exists (Edge fetches the chunk from Cloud).

---

## 6. Persisted Telemetry Schema

We defined the `NetworkMeasurement` model in [`adaptive_sr/shared/schemas.py`](file:///d:/Full%20Stack/AdaptiveSR/adaptive_sr/shared/schemas.py):

```python
class NetworkMeasurement(BaseModel):
    request_id: str
    network_path: Literal["client_edge", "edge_cloud"]
    timestamp: str  # ISO-8601 UTC format
    bytes_transferred: Optional[int] = None
    rtt_ms: Optional[float] = None
    transfer_duration_seconds: Optional[float] = None
    measured_throughput_mbps: Optional[float] = None
```

* **Request and Chunk Association**: All measurements related to video requests retain trace links to `request_id`, `chunk_id`, and `representation_id`.
* **Timestamp format**: Telemetry records use consistent ISO-8601 UTC string formats (e.g. `"2026-08-20T12:00:00Z"`).

---

## 7. No Emulation Statement

> [!IMPORTANT]
> **No Network Emulation**: No bandwidth throttling, artificial latency injection, packet loss modeling, or `tc`/`netem` emulation tools were introduced in Step 3.1. These capabilities are deferred to Step 3.2. Only the formal measurement contract was established.
