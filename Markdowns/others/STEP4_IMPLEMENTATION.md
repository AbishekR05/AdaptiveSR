# Step 4 — Edge Resource Monitoring

## 1. Objective

Step 4 establishes the resource telemetry foundation required by the AdaptiveSR Edge service. The Edge must be able to report its current resource state in a structured, timestamped telemetry record that is:

- **Real**: Measured from the live system via `psutil`.
- **Structured**: A Pydantic-validated `EdgeResourceTelemetry` model.
- **Independent**: Separate from the network telemetry (`NetworkMeasurement`) established in Step 3.
- **Observability-only**: No resource allocation, scheduling, or SR inference is introduced.

> [!IMPORTANT]
> **Step 4 provides OBSERVABILITY. It does NOT provide RESOURCE ALLOCATION.**

---

## 2. Resource Monitoring Architecture

### Step 4.1 — Dual-Level Monitoring

Step 4.1 introduces two distinct monitoring layers:

```
┌─────────────────────────────────────────────────┐
│  HOST-LEVEL (ResourceMonitor)                    │
│  "How busy is the Edge compute environment?"     │
│                                                  │
│  .snapshot()  →  EdgeResourceTelemetry           │
│  .per_core_utilization()  →  List[float]         │
│                                                  │
│  psutil.cpu_percent()       host-wide CPU %      │
│  psutil.cpu_count()         logical core count   │
│  psutil.virtual_memory()    host RAM             │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  PROCESS-LEVEL (ProcessMonitor)                  │
│  "How much CPU does THIS process consume?"       │
│                                                  │
│  .snapshot(interval)  →  ProcessResourceSnapshot │
│                                                  │
│  psutil.Process.cpu_percent()   per-process CPU  │
│  psutil.Process.memory_info()   per-process RSS  │
└─────────────────────────────────────────────────┘
```

The monitor is a **pure measurement object**. The Edge service calls `monitor.snapshot()` and receives a fully populated telemetry record. All OS-specific measurement logic is isolated inside `ResourceMonitor` — the Edge request handler does not call `psutil` directly.

### Module Locations

| File | Purpose |
|------|---------|
| `adaptive_sr/monitoring/__init__.py` | Package init |
| `adaptive_sr/monitoring/resource_monitor.py` | `ResourceMonitor` (host-level) + `ProcessMonitor` (process-level) |
| `adaptive_sr/shared/schemas.py` | `EdgeResourceTelemetry` + `ProcessResourceSnapshot` Pydantic models |

---

## 3. CPU Metric Definitions

### `cpu_cores_total`

**Definition**: The number of logical CPU cores exposed to the Edge process by the operating system.

**Implementation**: `psutil.cpu_count(logical=True)`

Logical cores include hyperthreaded virtual cores. This reflects the CPU parallelism available to Python processes on the current machine.

---

### `cpu_utilization`

**Definition**: System-wide CPU utilisation expressed as a percentage, averaged across all logical cores.

**Range**: `[0.0, 100.0]`

**Implementation**: `psutil.cpu_percent(interval=None)` — non-blocking; returns the delta since the most recent prior measurement. The monitor primes the counter on initialisation with a short blocking call (`interval=0.1`) to ensure subsequent non-blocking calls return meaningful values rather than `0.0`.

> [!IMPORTANT]
> **Real measurement, NOT synthesised.** The implementation never hardcodes or fabricates `cpu_utilization`.

---

### `cpu_cores_available`

**Definition**: An **observational estimate** of logical CPU cores not currently consumed by observed workloads.

**Formula**: `max(0, cpu_cores_total × (1 − cpu_utilization / 100))`

> [!CAUTION]
> **`cpu_cores_available` is an OBSERVATIONAL ESTIMATE ONLY.**
>
> - It is derived from host-wide CPU utilization, which includes ALL processes on the machine (Cloud, Edge, Client, network emulation, OS services, tests, etc.).
> - It does NOT represent physically allocatable or reserved cores.
> - It MUST NOT be consumed as a scheduler allocation quantity.
> - A value of 7.0 on an 8-core machine does NOT mean 7 cores are free for SR.
> - Step 4 has no OS-level core reservation mechanism.

This is a monitoring metric. A proper core reservation mechanism belongs to a later step.

---

## 4. Memory Metric Definitions

**RAM = SECONDARY observed resource** (not yet used for allocation decisions).

| Field | Definition | Implementation |
|-------|-----------|---------------|
| `memory_total_bytes` | Total physical RAM in bytes | `psutil.virtual_memory().total` |
| `memory_used_bytes` | Used physical RAM in bytes | `psutil.virtual_memory().used` |
| `memory_utilization` | `used / total × 100` | Derived |

Memory is included as a secondary telemetry dimension for future situational awareness. No memory-based allocation decision is made in Step 4.

---

## 5. `active_requests` Definition

**Definition**: The number of requests currently being processed by the Edge service at the moment of the snapshot.

**Semantics**: Caller-supplied. The Edge request handler passes the current in-flight request count when calling `monitor.snapshot(active_requests=N)`.

**Step 4 behaviour**: Because the synchronous FastAPI Edge implementation processes one request at a time per worker, `active_requests` is typically `1` during a request and `0` when idle.

---

## 6. `queue_depth` Definition

**Definition**: The number of requests pending after admission but before execution.

> [!NOTE]
> **No explicit application-level work queue exists in the current synchronous Edge implementation.**
>
> The synchronous FastAPI Edge has no application-level scheduler queue. Callers pass `queue_depth=0`. This value is documented explicitly and is NOT fabricated as a placeholder for a real queue.
>
> A real SR scheduling queue belongs to a later step. Do not confuse `queue_depth=0` with "the queue is empty" — there is no queue to be empty.

---

## 7. Sampling Mechanism

```python
monitor = ResourceMonitor(
    cluster_id="cluster_01",
    edge_id="edge_01",
    sampling_interval_seconds=1.0,   # configurable
)
telemetry = monitor.snapshot(active_requests=1, queue_depth=0)
```

`sampling_interval_seconds` controls how often a non-blocking CPU delta is used versus a fresh blocking short-interval read. Magic numbers do not appear in calling code — the interval is a constructor parameter.

The monitor is capable of producing periodic snapshots but does NOT create a permanently running background thread. It is a modular measurement utility that can later be attached to the Edge service's request lifecycle or a periodic health reporter.

---

## 8. Windows Implementation

| Capability | Windows support | Notes |
|-----------|----------------|-------|
| `psutil.cpu_percent()` | ✅ Native | No elevated privileges required |
| `psutil.cpu_count()` | ✅ Native | Returns logical core count |
| `psutil.virtual_memory()` | ✅ Native | Returns physical RAM stats |
| `tc`/`netem` (kernel shaping) | ❌ Not available | Not needed for Step 4 |
| GPU / CUDA | ❌ Not used | See §10 |

`psutil` 7.2.2 is already installed. No new dependencies are required.

---

## 9. Future Linux / Azure Compatibility

The `ResourceMonitor` abstraction is fully portable:
- `psutil` is cross-platform (Windows, Linux, macOS).
- No Windows-specific API is used.
- The same code runs inside an Azure Linux Edge VM without modification.
- A future step may swap the psutil backend for a cgroups-aware or container-aware resource reader without changing the `EdgeResourceTelemetry` schema.

---

## 10. GPU — Not Part of Step 4

GPU/VRAM monitoring is **not included** in Step 4.

**Reasons**:
1. The current development environment does not have CUDA enabled.
2. GPU scheduling and allocation are not required until SR workload integration.
3. Adding a GPU dependency (e.g. `pynvml`) now would create an unnecessary hardware requirement.

The `EdgeResourceTelemetry` schema is extensible; `gpu_utilization` and `vram_*` fields may be added in a future step when the Azure GPU Edge VM is targeted.

---

## 11. Why Monitoring is Separate from Allocation

`EdgeResourceTelemetry` answers: **"What is the resource state?"**

It does NOT answer: **"What should the scheduler do?"**

Allocation decisions (reserve N cores, reject requests, trigger SR) belong to a later step. Separating observability from control makes each component independently testable and replaceable.

---

## 12. Why `EdgeResourceTelemetry` is Separate from `NetworkMeasurement`

| Schema | Answers |
|--------|---------|
| `NetworkMeasurement` | "What is happening to the network? (bandwidth, RTT, transfer time)" |
| `EdgeResourceTelemetry` | "What is happening to the Edge compute environment? (CPU, RAM, requests)" |

These are independent telemetry dimensions collected at different points in the request lifecycle. Merging them would make each harder to reason about, test, and extend.

---

## 13. Known Limitations

| Limitation | Description | Future resolution |
|-----------|-------------|------------------|
| `cpu_cores_available` is an estimation | OS scheduling makes exact availability unknowable without reservation | Add cgroups/CPU affinity control in later step |
| `cpu_utilization` includes all system processes | psutil reports system-wide CPU, not per-process | Future: use `psutil.Process().cpu_percent()` for Edge-process-specific utilization |
| `queue_depth = 0` always in Step 4 | No application queue exists | Step 5+ adds SR scheduling queue |
| No GPU telemetry | CUDA not available in dev environment | Future step adds GPU fields to schema |
| RTT contamination (from Step 3) | Connection-inclusive RTT, not pure propagation delay | Persistent session pool in future networking step |
| `time.sleep()` jitter | OS scheduler granularity affects CPU sampling precision | Acceptable for Step 4 monitoring; not a control-plane metric |

---

## 14. Example Resource Telemetry Output

```json
{
  "timestamp": "2026-08-20T08:00:00.123456Z",
  "cluster_id": "cluster_01",
  "edge_id": "edge_01",
  "cpu_cores_total": 8,
  "cpu_utilization": 12.50,
  "cpu_cores_available": 7.0,
  "memory_total_bytes": 17179869184,
  "memory_used_bytes": 8730419200,
  "memory_utilization": 50.82,
  "active_requests": 1,
  "queue_depth": 0
}
```

`cpu_cores_available = 8 × (1 − 0.125) = 7.0`  
Memory utilization = 8,730,419,200 / 17,179,869,184 × 100 ≈ 50.82%

---

## 15. Relationship to Other Steps

| Step | What it provides | Step 4 relationship |
|------|-----------------|-------------------|
| Step 0 | Service foundation, basic telemetry | `cluster_id`, `edge_id` pattern reused |
| Step 1 | Video profiling | Independent |
| Step 2 | Representation chunking | Independent |
| Step 3 | Network measurement + emulation | `NetworkMeasurement` kept separate |
| **Step 4 / 4.1** | **Edge resource observability (host + process level)** | Foundation for Steps 5+ |
| Step 5+ | SR benchmarking, scheduling, allocation | Will use both `EdgeResourceTelemetry` and `ProcessResourceSnapshot` |

---

## 16. Host-Level vs Process-Level Resource Measurements

This section documents the fundamental distinction between the two monitoring layers introduced in Step 4.1.

### What each layer measures

| Layer | Class | Schema | Answers |
|-------|-------|--------|---------|
| Host-level | `ResourceMonitor` | `EdgeResourceTelemetry` | "How busy is the Edge compute environment?" |
| Process-level | `ProcessMonitor` | `ProcessResourceSnapshot` | "How much CPU and RAM does this specific process consume?" |

### Why they are different

The development machine runs multiple concurrent processes:
- Cloud service
- Edge service
- Client service
- Network emulation
- Tests
- OS services

Host-wide `cpu_utilization` aggregates ALL of these. It is therefore a poor proxy for the resource consumption of any individual workload.

**Example**:

```
Host cpu_utilization = 60%
```

This does **NOT** mean:
- SR is using 60% CPU
- SR is using 4.8 cores (`8 × 0.6 = 4.8`)
- Any specific process is responsible for 60% of load

**Correct measurement for SR benchmarking**:

```
ProcessMonitor(pid=sr_process_pid).snapshot(interval=0.5)
→ ProcessResourceSnapshot.cpu_percent = 320.0
→ Interpretation: SR is using ≈ 3.2 logical cores worth of CPU time
```

### Step 5 Contract

> [!IMPORTANT]
> **Step 5 SR benchmarking MUST use `ProcessMonitor` (process-level CPU measurements) to characterize SR workload resource consumption.**
>
> Host-wide `cpu_utilization` from `EdgeResourceTelemetry` MUST NOT be interpreted as the CPU consumption of the SR workload.
>
> The `cpu_cores_available` field from `EdgeResourceTelemetry` MUST NOT be used as the basis for SR scheduler decisions — it is an observational estimate derived from host-wide utilization, not a per-process measurement.

### `cpu_percent` semantics in `ProcessResourceSnapshot`

psutil `Process.cpu_percent()` returns a percentage of **one logical CPU core**:

| cpu_percent value | Interpretation |
|------------------|----------------|
| 0–100% | Process using 0–1 logical core |
| 100–200% | Process using 1–2 logical cores (multi-threaded) |
| 320% | Process using ≈ 3.2 logical cores |
| > 100 × N cores | Physically impossible; would indicate measurement error |

Values above 100% are normal and expected for multi-threaded SR workloads.

### Per-Core Utilization

`ResourceMonitor.per_core_utilization()` exposes per-logical-core CPU utilization via `psutil.cpu_percent(percpu=True)`. This is an **observational** capability provided to allow future reasoning about uneven CPU loading (e.g., detecting whether SR threads concentrate on specific cores). It is NOT a scheduler feature and does not allocate or reserve cores.
