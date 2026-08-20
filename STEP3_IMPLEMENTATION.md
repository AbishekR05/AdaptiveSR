# Step 3 — Network Measurement Contract & Controlled Emulation

---

## Step 3.1 — Network Measurement Contract

### 1. Architecture

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

The two paths must remain independently configured and independently measured.

### 2. Network Path Identities

* **`client_edge`**: Client ↔ Edge Node.
* **`edge_cloud`**: Edge Node ↔ Cloud Origin.

### 3. RTT Definition

RTT is measured independently from payload transfers using lightweight `/health` ping requests.

> [!WARNING]
> **RTT Contamination (addressed in Step 3.2):**
> RTT measurements currently use independent `/health` probes. Because the codebase uses bare `requests.get()` with no persistent HTTP session, every call — including health pings — opens a new TCP connection. RTT therefore includes TCP connection-establishment overhead and is **connection-inclusive latency**, not pure propagation delay. Step 3.2 documents this explicitly and defers persistent-session control to a future step.

### 4. Throughput & Transfer Duration

$$\text{Throughput (Mbps)} = \frac{\text{bytes\_transferred} \times 8}{\text{transfer\_duration\_seconds} \times 1{,}000{,}000}$$

Zero-byte RTT probes must not produce a non-zero throughput value.

### 5. Cache Semantics

* **Cache HIT**: `client_edge` transfer exists; `edge_cloud` payload transfer is **absent** (not zero-speed).
* **Cache MISS**: Both `client_edge` and `edge_cloud` transfers exist.

### 6. NetworkMeasurement Schema

```python
class NetworkMeasurement(BaseModel):
    request_id: str
    network_path: Literal["client_edge", "edge_cloud"]
    timestamp: str          # ISO-8601 UTC, e.g. "2026-08-20T12:00:00.000000Z"
    chunk_id: Optional[str] = None
    representation_id: Optional[str] = None
    bytes_transferred: Optional[int] = None
    rtt_ms: Optional[float] = None
    transfer_duration_seconds: Optional[float] = None
    measured_throughput_mbps: Optional[float] = None
```

* **RTT probes**: `chunk_id = None`, `representation_id = None`.
* **Chunk transfers**: `chunk_id` and `representation_id` populated.

---

## Step 3.2 — Controlled Network Emulation

### 1. Emulation Architecture

```text
CLIENT
   │ ← EmulatedHttpAdapter(config, "client_edge") applied here
   │   delay_ms + bandwidth throttle
   ▼
 EDGE
   │ ← EmulatedHttpAdapter(config, "edge_cloud") applied here
   │   delay_ms + bandwidth throttle
   ▼
 CLOUD / ORIGIN
```

> [!IMPORTANT]
> **This is application-level emulation, NOT kernel/network-stack-level packet shaping.**
> `tc`/`netem` (Linux kernel traffic control) is not available on Windows. The adapter injects delay via `time.sleep()` and throttles bandwidth by reading the response body in timed chunks. Telemetry measures the resulting actual transfer durations — it is never manually set to the configured bandwidth value.

### 2. Windows Limitations

* `tc`/`netem` requires Linux and elevated privileges — not available in this development environment.
* The emulation adapter is designed as a **swappable layer**: a future Linux/Azure deployment replaces `EmulatedHttpAdapter` with a real tc/netem shaper with no other code changes.

### 3. Connection Reuse Behavior

The current codebase uses `requests.get()` with no persistent HTTP session. Every call opens a **new TCP connection**. Therefore:
* RTT measurements (from health probes) include TCP handshake overhead.
* RTT is **connection-inclusive latency**, not pure propagation delay.
* Emulated delay (`delay_ms`) is injected **at the application layer** and adds to this connection-inclusive RTT.

### 4. Per-Path Configuration Model

```python
class NetworkPathEmulationConfig(BaseModel):
    bandwidth_mbps: Optional[float] = None  # None = unlimited
    delay_ms: float = 0.0
    packet_loss_rate: float = 0.0           # [0.0, 1.0]

class NetworkEmulationConfig(BaseModel):
    enabled: bool = True
    client_edge: NetworkPathEmulationConfig
    edge_cloud: NetworkPathEmulationConfig
    scenario_name: Optional[str] = None
```

Each path is independently configurable. Setting `enabled=False` disables all emulation (passthrough).

### 5. Bandwidth Emulation

The response body is streamed in 4 KB chunks. After writing each chunk, the adapter sleeps for:

$$\text{sleep} = \frac{\text{chunk\_bytes}}{\text{bytes\_per\_second}} - \text{elapsed}$$

This makes `transfer_duration_seconds` grow proportionally as configured bandwidth decreases. Throughput is then **measured** from actual bytes and actual duration using the Step 3.1 formula — not assigned from the configured value.

### 6. Delay Emulation

`time.sleep(delay_ms / 1000.0)` is called before streaming the response body. The sleep is only applied when `emulation.enabled = True` and `delay_ms > 0`.

### 7. Packet Loss

Application-layer fault injection: with probability `packet_loss_rate`, an `IOError` is raised before the response body is read. This simulates a dropped connection at the application layer. **It is not true packet-level loss.** Callers must handle the exception.

If reliable packet-level loss cannot be implemented, `packet_loss_rate` remains a documented adapter capability for the future Linux/Azure implementation.

### 8. Named Scenarios

| Scenario | client_edge BW | client_edge delay | edge_cloud BW | edge_cloud delay |
|----------|---------------|-------------------|---------------|------------------|
| good | 20 Mbps | 5 ms | 100 Mbps | 2 ms |
| moderate | 5 Mbps | 30 ms | 20 Mbps | 10 ms |
| poor | 1 Mbps | 100 ms | 5 Mbps | 50 ms |
| disabled | unlimited | 0 ms | unlimited | 0 ms |

Values are stored in `SCENARIOS` dict in `emulation.py` — not hardcoded throughout the codebase.

### 9. Experiment Reproducibility

Each `NetworkEmulationConfig` carries a `scenario_name` field. Logging the scenario name with run telemetry is sufficient to reconstruct bandwidth, delay, and loss configuration deterministically from `SCENARIOS`.

### 10. No SR/ABR/ML

No super-resolution, ABR, ML, QoE optimization, or scheduling functionality was introduced in Step 3.2.
