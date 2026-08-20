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
> **RTT Contamination — characterized, not yet fixed**
>
> The current implementation uses bare `requests.get()` for all HTTP calls, including RTT health probes. Each call opens a **new TCP connection** — there is no persistent session or connection pool. Therefore:
> - The measured RTT includes TCP connection-establishment overhead.
> - RTT is **connection-inclusive latency**, not pure propagation delay.
>
> Step 3.2 characterizes and documents this behavior. **Step 3.2 does NOT eliminate the contamination.** Persistent HTTP session / connection reuse remains a future networking refinement.
>
> Do not interpret any RTT values from this implementation as pure propagation delay.

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
   │   delay_ms + bandwidth throttle (application layer)
   ▼
 EDGE
   │ ← EmulatedHttpAdapter(config, "edge_cloud") applied here
   │   delay_ms + bandwidth throttle (application layer)
   ▼
 CLOUD / ORIGIN
```

> [!IMPORTANT]
> **This is application-level emulation, NOT kernel/network-stack-level packet shaping.**
>
> `tc`/`netem` (Linux kernel traffic control) is not available on Windows. The adapter injects delay via `time.sleep()` and throttles bandwidth by reading the response body in timed chunks. Telemetry measures the resulting actual transfer durations — the measured `throughput_mbps` is **never manually assigned** from the configured bandwidth value.

### 2. Windows Limitations

* `tc`/`netem` requires Linux and elevated privileges — not available in this development environment.
* The emulation adapter is a **swappable layer**: a future Linux/Azure deployment can replace `EmulatedHttpAdapter` with a tc/netem shaper with no other code changes.

### 3. Connection Reuse Behavior

The current codebase uses `requests.get()` with no persistent HTTP session. Every call — including health pings — opens a new TCP connection. Therefore:
* RTT measurements include TCP handshake overhead.
* Emulated `delay_ms` adds to this connection-inclusive round-trip.
* Persistent session / connection reuse remains a future networking refinement.

### 4. `delay_ms` vs Measured RTT — Explicit Semantics

> [!IMPORTANT]
> **`delay_ms` is NOT measured RTT.**

| Concept | Description |
|---------|-------------|
| `delay_ms` | Application-layer delay injected **before the response body is returned** to the caller. This is a configuration parameter, not an observation. |
| Measured RTT | Wall-clock round-trip time observed by a live health probe. Includes TCP handshake overhead. |
| `ROSEVIN_RTT_REFERENCE_MS` | **Observed** RTT from the paper's physical testbed (~43 ms). A documentation constant, not an injected parameter. |

Injecting `delay_ms = D` on a path increases the **wall-clock** round-trip by approximately `D` (one-way injection point), not `2D`. This is not the same as configuring an RTT. The health-probe `rtt_ms` field in `NetworkMeasurement` must be populated from the **actual probe observation**, not copied from the configured `delay_ms`.

### 5. Bandwidth Throttling

The response body is streamed in 4 KB chunks. After writing each chunk, the adapter computes:

$$\text{sleep} = \frac{\text{chunk\_bytes}}{\text{bytes\_per\_second}} - \text{elapsed}$$

This makes `transfer_duration_seconds` grow proportionally as configured bandwidth decreases. `measured_throughput_mbps` is then **computed** from actual bytes and actual duration — never assigned from the configured value.

> [!NOTE]
> Measured throughput may deviate slightly from the configured target because `time.sleep()` is subject to OS scheduler granularity and scheduling jitter.

### 6. Packet Loss

Application-layer fault injection: with probability `packet_loss_rate`, an `IOError` is raised before the response body is read. This simulates a dropped connection at the application layer — **it is not true packet-level loss.** Callers must handle the exception.

`packet_loss_rate` remains a documented adapter capability for the future Linux/Azure implementation using true packet shaping.

---

### 7. Rosevin (2024) Paper Reference

The base paper reports the following actual edge-cloud testbed measurements:

| Metric | Value | Source |
|--------|-------|--------|
| RTT | ≈ 43 ms | Rosevin (2024) physical testbed |
| Bandwidth | ≈ 186 Mbps | Rosevin (2024) physical testbed |

These values are stored as module-level constants in `emulation.py`:

```python
ROSEVIN_RTT_REFERENCE_MS: float = 43.0   # ms — paper-reported RTT (observed)
ROSEVIN_BANDWIDTH_MBPS: float   = 186.0  # Mbps — paper-reported edge↔cloud BW
```

The paper does **not** define GOOD / MODERATE / POOR tiers — those are our experimental scenario parameters.

---

### 8. Named Scenarios

| Scenario | Purpose | Client→Edge BW | Client→Edge delay_ms | Edge→Cloud BW | Edge→Cloud delay_ms | Relationship to Rosevin |
|----------|---------|---------------|---------------------|--------------|---------------------|------------------------|
| `rosevin_baseline` | Paper reference conditions | 50 Mbps | 10 ms | 186 Mbps | 21 ms (≈ ½ × 43 ms RTT)¹ | **PAPER REFERENCE** — edge_cloud BW is the paper's reported value |
| `good` | Favorable CDN-like conditions | 20 Mbps | 5 ms | 500 Mbps | 5 ms | Better than Rosevin baseline |
| `moderate` | Conditions broadly around Rosevin | 8 Mbps | 25 ms | 186 Mbps | 21 ms | Aligned with Rosevin on edge_cloud |
| `poor` | Substantially degraded network | 1 Mbps | 100 ms | 20 Mbps | 60 ms | Significantly worse than Rosevin |
| `disabled` | No emulation (passthrough) | — | 0 ms | — | 0 ms | Not applicable |

> **¹ delay_ms vs RTT note**: The Rosevin paper reports an observed RTT of ~43 ms. `delay_ms` is an **application-layer injected one-way delay**, not RTT. The value 21 ms is used as a rough one-way approximation of the 43 ms observed round-trip. These are **not equivalent** and must not be treated as such.

> [!CAUTION]
> **GOOD, MODERATE, and POOR are OUR experimental scenario parameters.** They were NOT directly taken from the Rosevin paper. The paper does not define these three tiers. The `rosevin_baseline` scenario is the only scenario explicitly derived from paper-reported measurements.

### 9. Per-Path Configuration Model

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

### 10. Experiment Reproducibility

Each `NetworkEmulationConfig` carries a `scenario_name` field. Logging the scenario name with run telemetry is sufficient to reconstruct bandwidth, delay, and loss configuration deterministically from `SCENARIOS`.

### 11. No SR/ABR/ML

No super-resolution, ABR, ML, QoE optimization, or scheduling functionality was introduced in Step 3.2.
