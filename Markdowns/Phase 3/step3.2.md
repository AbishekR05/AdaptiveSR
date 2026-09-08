STEP 3.2 — CONTROLLED NETWORK EMULATION

Step 3.1 is FROZEN.

Implement ONLY Step 3.2.

Do NOT implement Step 3.3 or any SR/ABR/ML/scheduler functionality.

============================================================
OBJECTIVE
============================================================

Introduce reproducible, controlled network conditions into the
existing Client → Edge → Cloud architecture.

The purpose is experimental network emulation.

We need to be able to reproduce different network conditions
while preserving the existing measurement contract.

The emulation must affect actual payload transfers rather than
simply modifying telemetry values.

Architecture:

    CLIENT
       │
       │ client_edge
       │
       ▼
     EDGE
       │
       │ edge_cloud
       │
       ▼
     CLOUD / ORIGIN

The two paths MUST remain independently configurable.

============================================================

1. # DO NOT BREAK STEP 3.1 CONTRACT

Preserve:

    network_path:
        client_edge
        edge_cloud

Preserve the distinction between:

    RTT
    transfer_duration_seconds
    measured_throughput_mbps

Preserve:

    request_id
    chunk_id
    representation_id

and the existing nullable semantics for RTT probes.

Do not replace NetworkMeasurement with a new incompatible schema.

Existing Step 0, Step 0.1, Step 1, and Step 2 behavior must remain
compatible.

============================================================ 2. EMULATION PARAMETERS
============================================================

Create a configuration model for a network path.

Conceptually:

    NetworkEmulationConfig

with parameters for:

    bandwidth limit
    artificial RTT / delay
    packet loss
    jitter (only if needed by the implementation)

Each path must have its own configuration:

    client_edge_config
    edge_cloud_config

Do not use one global network configuration for both paths.

Configuration must support:

    emulation disabled

so the system can run normally without network constraints.

============================================================ 3. IMPORTANT — REAL NETWORK EFFECT
============================================================

The implementation must affect the actual communication path.

Do NOT do this:

    configured_bandwidth = 5 Mbps
    telemetry.throughput = 5 Mbps

That is invalid.

Instead:

    request
      ↓
    actual network shaping / delay
      ↓
    response
      ↓
    measured transfer duration
      ↓
    calculated throughput

The telemetry must measure the resulting behavior.

============================================================ 4. PLATFORM / ENVIRONMENT

First inspect the development environment.

The project is currently developed on a Windows machine and
services may be running locally.

Do NOT assume Linux-only functionality such as tc/netem is
available.

Determine the most appropriate implementation strategy for the
current development environment.

If OS-level traffic shaping cannot be safely and reproducibly
applied on Windows without elevated privileges or external
dependencies, DO NOT fake it.

Instead implement a clearly isolated emulation layer that can
introduce controlled delay and bandwidth behavior at the
application/service boundary, while documenting that this is
application-level emulation rather than kernel-level packet
shaping.

The architecture must make it possible to replace this adapter
with tc/netem or another real network shaper later, especially
when deploying the services to Linux-based Azure infrastructure.

============================================================ 5. RTT / CONNECTION REUSE
============================================================

Step 3.1 explicitly identified a possible RTT contamination issue.

Step 3.2 must address it.

Inspect the existing HTTP client/server implementation.

Determine whether health probes and chunk requests:

    reuse the same HTTP session/connection

or:

    establish separate connections.

Do not blindly assume.

Document the behavior.

The goal is that RTT measurements used by the project have clearly
defined connection semantics.

Do not claim that RTT is pure propagation delay unless the
implementation actually supports that interpretation.

If connection setup remains part of the measured RTT, explicitly
document it as such.

============================================================ 6. BANDWIDTH EMULATION
============================================================

Implement controlled bandwidth limiting.

The implementation must make a payload transfer take longer when
the configured bandwidth decreases.

For example, conceptually:

    high bandwidth
        → faster transfer

    low bandwidth
        → slower transfer

Do not manipulate telemetry after the transfer.

The resulting:

    transfer_duration_seconds

must be measured from the actual transfer.

Then:

    measured_throughput_mbps

must be calculated from actual bytes and actual transfer duration
using the Step 3.1 contract.

============================================================ 7. ARTIFICIAL RTT / DELAY
============================================================

Implement configurable network delay.

The delay must be associated with the appropriate network path.

For example:

    client_edge delay

must affect Client ↔ Edge communication.

    edge_cloud delay

must affect Edge ↔ Cloud communication.

Do NOT accidentally apply the client-edge delay to the
edge-cloud path or vice versa.

Document exactly where the delay is injected.

============================================================ 8. PACKET LOSS

Add packet-loss support only if it can be implemented reliably
within the selected emulation layer.

Do not create fake packet-loss telemetry.

If packet loss is implemented at the application level, define
exactly what it means.

For example, if the emulation intentionally fails/retries a
request, document that behavior.

If reliable packet-level loss cannot be implemented in the current
Windows development environment, leave packet loss as a documented
adapter capability for the future Linux/Azure implementation rather
than pretending to emulate it.

Correctness is more important than having every feature immediately.

============================================================ 9. SCENARIO CONFIGURATION

Create named network scenarios.

At minimum provide the architecture for:

    GOOD
    MODERATE
    POOR

Do NOT arbitrarily invent research values if the project
documentation/base paper already defines values elsewhere.

If exact values are not currently specified in the repository,
make the scenario values explicit configuration rather than
hardcoding them throughout the code.

Each scenario must configure both paths independently.

Example conceptual structure:

    good:
        client_edge: {...}
        edge_cloud: {...}

    moderate:
        client_edge: {...}
        edge_cloud: {...}

    poor:
        client_edge: {...}
        edge_cloud: {...}

============================================================ 10. EXPERIMENT REPRODUCIBILITY

A run must identify which network scenario was active.

Network telemetry/logging should make it possible to reconstruct:

    scenario
    network_path
    bandwidth configuration
    delay configuration
    loss configuration

Do not put the entire configuration into every telemetry record
if that creates unnecessary duplication.

A run-level configuration identifier is acceptable if the
configuration can be recovered deterministically.

============================================================ 11. TESTS

Add tests proving:

1. Emulation can be disabled.

2. A configured delay measurably increases request latency.

3. Lower configured bandwidth produces a longer transfer for the
   same payload under controlled conditions.

4. Client-edge emulation affects only client-edge traffic.

5. Edge-cloud emulation affects only edge-cloud traffic.

6. The two path configurations can differ simultaneously.

7. Measured throughput is derived from actual bytes and actual
   transfer duration.

8. Telemetry is not manually overwritten with the configured
   bandwidth.

9. RTT measurements retain the correct network_path.

10. Existing cache HIT/MISS semantics remain correct.

11. Existing Step 0/0.1/1/2 tests continue to pass.

Tests must avoid relying on extremely precise timing thresholds.
Use tolerances and sufficiently large payloads/delays so the tests
are deterministic.

============================================================ 12. EXPERIMENTAL SANITY CHECK

Create one small reproducible experiment demonstrating something
like:

    Scenario: GOOD
        ↓
    transfer chunk
        ↓
    record duration + throughput

then:

    Scenario: POOR
        ↓
    transfer SAME chunk
        ↓
    record duration + throughput

The payload and representation must remain identical.

The expected result is that the constrained scenario produces
different measured network behavior.

Do not claim exact theoretical throughput if the implementation
uses application-level emulation.

============================================================ 13. NO SR / ABR / ML

Do NOT implement:

- Super-resolution
- frame selection
- FPS adaptation
- ABR
- resource allocation
- scheduler
- ML
- QoE optimization
- Azure deployment

Those belong to later steps.

============================================================ 14. DOCUMENTATION

Update:

    STEP3_IMPLEMENTATION.md

Add:

    Step 3.2 — Controlled Network Emulation

Document:

    - emulation architecture
    - path-specific configuration
    - bandwidth mechanism
    - delay mechanism
    - packet-loss status
    - Windows limitations
    - connection reuse behavior
    - Good / Moderate / Poor scenarios
    - reproducibility
    - distinction between application-level emulation and
      kernel-level network shaping

Explicitly state whether the current implementation is:

    application-level emulation

or:

    kernel/network-stack-level emulation.

Do not claim kernel-level shaping if it is not actually being used.

============================================================ 15. DATETIME CLEANUP

If the implementation touches timestamp generation, replace:

    datetime.utcnow()

with:

    datetime.now(timezone.utc)

using the appropriate timezone-aware import.

Preserve the existing ISO-8601 UTC output contract.

This is a small compatibility cleanup and should not alter the
telemetry semantics.

============================================================ 16. REGRESSION

Run:

    python -m pytest tests/ -v

All existing tests must pass.

============================================================
STOP CONDITION

STOP after Step 3.2.

Do NOT start Step 3.3.

Return:

1. Files changed
2. Network emulation architecture
3. Exact path-specific configuration model
4. How bandwidth is emulated
5. How RTT/delay is emulated
6. Packet-loss status
7. Connection reuse behavior
8. Windows limitations
9. Good/Moderate/Poor configuration
10. Tests added
11. Full pytest results
12. Confirmation that telemetry measures actual transfer behavior
    rather than manually assigned values
13. Confirmation that no SR/ABR/ML/scheduler functionality was
    introduced
