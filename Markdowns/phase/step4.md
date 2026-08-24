STEP 4 — EDGE RESOURCE MONITORING

STEP 3 IS FROZEN.

Implement ONLY Step 4.

Do NOT implement Step 5 or any SR inference.

============================================================
PROJECT CONTEXT
============================================================

AdaptiveSR is a cloud → edge → client video streaming system.

Current architecture:

    CLIENT
       │
       │ client_edge network path
       ▼
     EDGE
       │
       │ edge_cloud network path
       ▼
     CLOUD / ORIGIN

Completed:

    Step 0  — service foundation
    Step 1  — continuous video profiling
    Step 2  — multi-representation chunking
    Step 3  — network measurement + controlled emulation

The system is intended for real-time video streaming.

The eventual system will perform super-resolution at the Edge and
make resource-aware decisions.

Step 4 establishes the resource telemetry foundation required by
those later steps.

============================================================
STEP 4 OBJECTIVE
============================================================

Implement an Edge Resource Monitoring subsystem.

The Edge must be able to report its current resource state in a
structured, timestamped telemetry record.

The monitoring system must be independent of SR inference.

No SR model should be introduced in this step.

============================================================
1. PRIMARY RESOURCE MODEL
============================================================

The paper-faithful resource variable selected for this project is:

    CPU core allocation / availability.

CPU is therefore the PRIMARY resource dimension.

The resource telemetry must distinguish:

    cpu_cores_total
    cpu_cores_available
    cpu_utilization

Do not interpret:

    cpu_utilization

as equivalent to:

    cpu_cores_available

They represent different quantities.

For example:

    total cores = 8
    utilization = 50%

does not automatically mean that exactly 4 cores are available
for SR work.

Availability must have its own defined semantics.

============================================================
2. RESOURCE TELEMETRY SCHEMA
============================================================

Create a shared schema similar to:

    EdgeResourceTelemetry

The exact implementation should follow the project's existing
schema conventions.

It should contain at minimum:

    timestamp
    cluster_id
    edge_id

    cpu_cores_total
    cpu_utilization

    cpu_cores_available

    memory_total_bytes
    memory_used_bytes
    memory_utilization

    active_requests
    queue_depth

Do not add arbitrary fields merely because the operating system
provides them.

Every field must have a defined meaning.

============================================================
3. CPU CORE SEMANTICS
============================================================

Define clearly what:

    cpu_cores_total

means.

Use the logical CPU count exposed to the Edge process/environment.

Define:

    cpu_cores_available

as the number of logical CPU cores that are currently considered
available to the Edge workload under the monitoring model.

IMPORTANT:

Do NOT pretend that operating-system CPU utilization alone tells us
how many cores are "available for SR."

If the current Edge implementation has no resource reservation
mechanism yet, define the initial semantics explicitly.

For example, a reasonable Step 4 interpretation is:

    cpu_cores_available =
        cpu_cores_total × (1 - normalized CPU utilization)

with appropriate handling/rounding.

BUT:

Do not blindly use this formula if the existing architecture already
has a better resource model.

Document whichever definition is actually implemented.

This is a MONITORING metric, not yet a resource allocator.

============================================================
4. CPU UTILIZATION

Measure actual CPU utilization.

Do not generate synthetic values.

Use an appropriate cross-platform Python/system monitoring library
if one is already present or can be safely introduced.

The implementation must work on the current Windows development
environment.

Do not require Linux-only tools.

Do not hardcode:

    cpu_utilization = 50

or similar values.

============================================================
5. SAMPLING

Implement configurable resource sampling.

For example:

    ResourceMonitor(
        sampling_interval_seconds=...
    )

The interval must be configuration rather than scattered magic
numbers.

The monitor should be capable of producing periodic snapshots.

Do not create a permanently running background process unless the
existing service architecture requires it.

Keep the monitor modular so it can later be attached to the Edge
service.

============================================================
6. REQUEST / QUEUE STATE

The Edge telemetry must expose:

    active_requests
    queue_depth

Define these precisely.

For Step 4:

    active_requests
        = requests currently being processed by the Edge service.

    queue_depth
        = requests waiting for processing after admission but before
          execution.

If the current synchronous Edge implementation has no real queue,
do NOT invent one.

In that case:

    queue_depth = 0

is acceptable only if it is documented as:

    "No explicit application-level work queue exists in the current
     implementation."

Do not call this a scheduler queue.

A real SR scheduling queue belongs to later steps.

============================================================
7. MEMORY

Add RAM monitoring as SECONDARY telemetry.

At minimum:

    memory_total_bytes
    memory_used_bytes
    memory_utilization

Use actual system measurements.

Do not make memory part of the resource allocation decision yet.

Document:

    CPU = primary resource dimension
    RAM = observed secondary resource

============================================================
8. GPU

DO NOT make GPU/VRAM a required Step 4 dependency.

The project may eventually run SR workloads on Azure GPU infrastructure.

However:

    GPU scheduling/allocation

is NOT part of this step.

If the monitoring abstraction can optionally expose GPU information
without creating a dependency, it may be documented as future
extension.

Otherwise leave it out.

Do NOT install CUDA tooling merely for Step 4.

============================================================
9. CLUSTER / EDGE IDENTITY

Resource telemetry must contain:

    cluster_id
    edge_id

These must come from the Edge service configuration/identity.

Do not hardcode telemetry records to:

    cluster_01
    edge_01

unless those are actually the configured identity.

The existing multi-edge identity architecture must remain intact.

============================================================
10. TIMESTAMP

Use timezone-aware UTC timestamps.

Use:

    datetime.now(timezone.utc)

not:

    datetime.utcnow()

Preserve the project's existing timestamp serialization format.

============================================================
11. RESOURCE MONITOR API

Create a clean interface between:

    Edge service

and:

    Resource monitor.

Conceptually:

    monitor.snapshot()

→

    EdgeResourceTelemetry

The Edge service should not contain OS-specific resource-monitoring
logic throughout its request handlers.

Keep system measurement isolated behind the monitor abstraction.

This will make future Azure/Linux deployment easier.

============================================================
12. TELEMETRY STORAGE / OUTPUT

Follow the existing project's telemetry conventions.

A resource snapshot should be identifiable and timestamped.

Do not modify existing network telemetry semantics.

Do not merge:

    EdgeResourceTelemetry

into:

    NetworkMeasurement

They represent different dimensions.

Network telemetry answers:

    "What is happening to the network?"

Resource telemetry answers:

    "What is happening to the Edge compute environment?"

Keep them separate.

============================================================
13. RESOURCE MONITORING DURING ACTUAL WORK

Create a test/demo workload that produces measurable CPU activity.

The purpose is to demonstrate that resource telemetry responds to
actual system conditions.

Do NOT use SR.

For example, a controlled CPU-bound test workload may be used solely
for testing the monitor.

The monitor should demonstrate that:

    idle Edge

and:

    CPU-loaded Edge

produce different CPU utilization observations.

Avoid relying on exact utilization percentages.

Operating-system scheduling means exact values are inherently
variable.

============================================================
14. TESTS

Add tests covering:

1. Resource monitor initializes successfully.

2. CPU core count is positive.

3. CPU utilization is within:

       0 <= utilization <= 100

4. Memory utilization is within:

       0 <= utilization <= 100

5. cpu_cores_available follows the documented semantics.

6. cluster_id is preserved.

7. edge_id is preserved.

8. timestamps are timezone-aware UTC.

9. active_requests has the documented meaning.

10. queue_depth has the documented meaning.

11. Two Edge instances can produce distinct resource telemetry
    identities.

12. Resource telemetry is separate from NetworkMeasurement.

13. Existing Step 0–3 tests continue to pass.

Do NOT write tests that require exact CPU utilization values.

Use ranges/invariants instead.

============================================================
15. OPTIONAL RESOURCE LOAD TEST

Add a small deterministic test/demo utility that creates temporary
CPU load.

IMPORTANT:

This is only for validating the monitor.

It must:

    start
    create bounded CPU activity
    sample resource state
    stop
    release resources

It must not leave runaway processes or threads.

Do not make the test suite depend on a high CPU load being achieved.

============================================================
16. NO RESOURCE ALLOCATION YET

This is extremely important.

Step 4 ONLY MEASURES resources.

Do NOT implement:

    allocate 1 CPU core
    allocate 2 CPU cores
    scale CPU allocation
    migrate workload
    admission control
    scheduler decisions

Those belong to later steps.

The monitor should answer:

    "What is the resource state?"

It should NOT answer:

    "What should the scheduler do?"

============================================================
17. NO SR

Do not install or integrate:

    Real-ESRGAN
    FSRCNN
    BasicVSR++
    RealBasicVSR
    any SR model

Step 5 is SR benchmarking.

============================================================
18. NO AZURE

Do not deploy anything to Azure.

The monitor must be portable enough that the same abstraction can
later run inside an Azure Linux edge VM.

But Azure deployment is a later step.

============================================================
19. DOCUMENTATION

Create/update:

    STEP4_IMPLEMENTATION.md

Document:

    1. Step 4 objective
    2. Resource monitoring architecture
    3. CPU metric definitions
    4. cpu_cores_total semantics
    5. cpu_cores_available semantics
    6. CPU utilization semantics
    7. memory metric definitions
    8. active_requests definition
    9. queue_depth definition
    10. sampling mechanism
    11. Windows implementation
    12. future Linux/Azure compatibility
    13. why GPU is not part of Step 4
    14. why monitoring is separate from allocation
    15. known limitations

Explicitly state:

    Step 4 provides OBSERVABILITY.

    It does not provide RESOURCE ALLOCATION.

============================================================
20. REGRESSION

Run:

    python -m pytest tests/ -v

All previous tests from Steps 0–3 must pass.

============================================================
STOP CONDITION
============================================================

STOP after Step 4.

Do NOT start Step 5.

Return:

1. Files changed
2. Resource monitoring architecture
3. Exact telemetry schema
4. CPU metric semantics
5. Memory metric semantics
6. active_requests / queue_depth semantics
7. Sampling implementation
8. Windows compatibility
9. Tests added
10. Full pytest results
11. Example resource telemetry output
12. Confirmation that measurements are real system measurements
13. Confirmation that no resource allocation was implemented
14. Confirmation that no SR/ML/ABR functionality was introduced
15. Confirmation that Steps 0–3 remain frozen and passing