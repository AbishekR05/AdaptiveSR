STEP 3.1 — NETWORK MEASUREMENT CONTRACT

We have frozen:

- Step 0 — Foundation
- Step 0.1 — Foundation Hardening
- Step 1 — Video Profiling / Logical Timeline
- Step 2.1 — Representation Schema
- Step 2.2 — Chunk-to-Representation Mapping

Begin ONLY Step 3.1.

Do NOT implement Step 3.2 or any later Step 3 functionality.

============================================================
OBJECTIVE
============================================================

Establish the formal network measurement contract for the
AdaptiveSR system.

The eventual architecture is:

    CLIENT
       │
       │ Client ↔ Edge network path
       │
       ▼
     EDGE
       │
       │ Edge ↔ Cloud network path
       │
       ▼
     CLOUD / ORIGIN

The project must treat these as TWO independent network paths.

Do NOT collapse them into a single "network latency" or
"network throughput" value.

============================================================

1. # NETWORK PATH IDENTITY

Define two explicit paths:

    client_edge

and:

    edge_cloud

Each measurement must identify which path produced it.

The existing cluster_id and edge_id fields must remain intact.

A future measurement should be attributable to:

    cluster_id
    edge_id
    network_path

============================================================ 2. RTT DEFINITION
============================================================

Define RTT as an independent measurement from data transfer.

RTT must NOT be calculated from:

    chunk_size / throughput

and must NOT be inferred from chunk download duration.

Use lightweight health/ping requests.

Existing Step 0.1 RTT measurements should be reused rather than
duplicated.

The contract should distinguish:

    client_edge_rtt_ms

from:

    edge_cloud_rtt_ms

Do not rename existing telemetry fields unnecessarily if they
already have an established contract.

Document exactly:

- where the ping originates
- which endpoint responds
- whether the measurement is round-trip
- the unit
- what timestamp represents the sample

============================================================ 3. THROUGHPUT DEFINITION
============================================================

Define measured throughput as:

    transferred_bytes * 8
    ---------------------
    transfer_duration_seconds

Result:

    Mbps

Do NOT include RTT in transfer_duration_seconds.

Do NOT use the total request wall-clock duration as throughput
unless the contract explicitly defines it that way.

The measurement must clearly distinguish:

    RTT

from:

    payload transfer duration

============================================================ 4. TRANSFER TIME
============================================================

Define payload transfer duration separately from RTT.

For a chunk transfer, record:

    bytes_transferred
    transfer_duration_seconds
    measured_throughput_mbps

The contract must make clear that:

    transfer_duration_seconds

is the time required to transfer the payload, while RTT is a
separate network characteristic.

Do not assume:

    total request duration = RTT + transfer duration

as an exact physical model yet.

There may be application/server/scheduling overhead.

Those components must remain separately identifiable where
possible.

============================================================ 5. TIMESTAMPS
============================================================

Network measurements should have sufficient timestamp information
to reconstruct experiments.

Define a consistent timestamp representation.

Prefer UTC ISO-8601 timestamps for persisted telemetry.

A measurement should allow us to determine:

    when the measurement occurred
    which chunk it belongs to
    which representation it belongs to
    which network path it describes

Do not add unnecessary high-frequency timestamp fields if the
existing telemetry structure already provides sufficient timing.

============================================================ 6. CHUNK-LEVEL ASSOCIATION
============================================================

A network measurement associated with a video transfer must be
traceable to:

    request_id
    chunk_id
    representation_id
    cluster_id
    edge_id
    network_path

where applicable.

Health-ping RTT samples do not necessarily require chunk_id,
because they are independent of payload transfer.

Do not fabricate a chunk association for independent RTT probes.

============================================================ 7. CACHE EFFECT

Network measurement must distinguish:

    cache HIT
    cache MISS

because:

    Edge cache HIT

does not require Edge → Cloud payload retrieval.

For a cache HIT:

    client_edge transfer exists
    edge_cloud payload transfer may be absent

For a cache MISS:

    client_edge transfer exists
    edge_cloud payload retrieval exists

Do not represent a cache-hit Cloud transfer as zero-speed or
zero-duration network traffic.

Use explicit cache semantics.

============================================================ 8. NO SR / ABR / ML

Step 3.1 must NOT introduce:

- Super-resolution
- ML
- ABR
- resource allocation
- scheduler
- FPS adaptation
- QoE optimization
- Azure deployment
- GPU allocation

This step only establishes the measurement contract.

============================================================ 9. REUSE EXISTING TELEMETRY

Inspect the current schemas and telemetry generated by:

    Step 0
    Step 0.1
    Step 1
    Step 2

Before adding fields.

Do NOT create duplicate concepts such as:

    rtt
    round_trip_time
    latency
    network_latency

if one canonical field can represent the concept.

Reuse existing fields where their semantics already match.

If an existing field is semantically ambiguous, document the
problem and make the smallest necessary correction.

Do not casually break the frozen schemas.

============================================================ 10. DATA MODEL

Create or update the minimum necessary shared model for a
network measurement.

The model should conceptually support:

    request_id
    network_path
    timestamp
    bytes_transferred (nullable for RTT probes)
    rtt_ms (nullable for transfer-only measurements)
    transfer_duration_seconds (nullable for RTT-only measurements)
    measured_throughput_mbps (nullable for RTT-only measurements)

Use the project's existing Pydantic/schema conventions.

Do not duplicate ClientTelemetry or EdgeTelemetry if an existing
schema can cleanly contain the information.

Choose the smallest architecture-consistent change.

============================================================ 11. VALIDATION

Add tests proving:

1. Client ↔ Edge and Edge ↔ Cloud are distinct network paths.

2. RTT and transfer duration are separate quantities.

3. Throughput calculation follows:

   bytes \* 8 / seconds / 1,000,000

4. Zero-byte RTT probes do not produce a fake throughput value.

5. A cache HIT does not fabricate an Edge → Cloud transfer.

6. A cache MISS can record an Edge → Cloud transfer.

7. Measurements retain request identity.

8. Measurements retain network-path identity.

9. Units are consistent.

10. Existing Step 0/0.1 telemetry remains compatible.

11. Existing foundation tests still pass.

============================================================ 12. IMPORTANT — DO NOT EMULATE YET
============================================================

Do NOT implement bandwidth throttling.

Do NOT implement artificial latency.

Do NOT implement packet loss.

Do NOT use tc/netem or equivalent tools yet.

Those belong to Step 3.2.

Step 3.1 is ONLY the measurement contract.

============================================================ 13. DOCUMENTATION

Create/update:

    STEP3_IMPLEMENTATION.md

Add:

    Step 3.1 — Network Measurement Contract

Document:

    Client ↔ Edge path
    Edge ↔ Cloud path
    RTT definition
    throughput definition
    transfer duration
    cache HIT/MISS semantics
    chunk association
    timestamps
    telemetry schema

Include a small architecture diagram:

    CLIENT
       │
       │ client_edge
       │ RTT + transfer
       ▼
     EDGE
       │
       │ edge_cloud
       │ RTT + transfer
       ▼
     CLOUD

Explicitly state:

    RTT is measured independently from payload transfer.

============================================================ 14. REGRESSION

Run:

    python -m pytest tests/ -v

All existing tests must remain passing.

============================================================
STOP CONDITION

STOP after Step 3.1.

Do NOT implement Step 3.2.

Return:

1. Files changed
2. Final network measurement schema
3. Field semantics
4. Tests added
5. Full pytest result
6. Any compatibility concerns
7. Confirmation that no network emulation was introduced
