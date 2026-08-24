============================================================
ADAPTIVESR — STEP 5.3 IMPLEMENTATION
CPU AFFINITY + PROCESSMONITOR INTEGRATION
============================================================

IMPORTANT:

Steps 0–4 are COMPLETE and FROZEN.

Step 5 contains:

    5.1 Benchmark dataset / test-video preparation     [FROZEN]
    5.2 SR model runner adapter interface              [FROZEN]
    5.3 CPU affinity + ProcessMonitor integration      [THIS STEP]
    5.4 GPU measurement
    5.5 Inference benchmark harness
    5.6 Quality evaluation
    5.7 FPS / real-time feasibility analysis
    5.8 Machine-readable benchmark dataset
    5.9 Validation + reproducibility report

IMPLEMENT ONLY STEP 5.3.

DO NOT IMPLEMENT 5.4–5.9.

============================================================

1. # OBJECTIVE

Step 5.2 already provides controlled CPU thread configuration:

    initialize(device, scale, num_threads=None)

Step 5.3 now adds the second part required for trustworthy CPU
resource experiments:

    CPU affinity
        +
    existing ProcessMonitor integration

The purpose is to create a controlled execution environment in which
an SR inference process can be restricted to a known number/set of
logical CPU cores while the existing Step 4 ProcessMonitor observes
the actual process/resource state.

The future benchmark flow will eventually be:

    benchmark case
         ↓
    choose CPU core configuration
         ↓
    apply CPU affinity
         ↓
    configure model threads
         ↓
    run SR inference
         ↓
    ProcessMonitor observes process
         ↓
    Step 5.5 measures latency/FPS

BUT:

Step 5.3 must NOT implement the benchmark harness.

============================================================ 2. WHY CPU AFFINITY IS REQUIRED
============================================================

The project is investigating SR latency as a function of available
CPU resources.

The Rosevin reference implementation explicitly considers pure CPU
resource configurations ranging from:

    1 CPU core
    ...
    10 CPU cores

The benchmark therefore must not merely pass:

    num_threads=N

to the model.

Thread count and actual CPU availability are separate controls.

We need to distinguish:

    logical CPU affinity
        =
    which logical CPUs the process is allowed to execute on

from:

    model thread count
        =
    how many compute threads the backend attempts to use

Step 5.3 must provide both controls.

============================================================ 3. FIRST — INSPECT FROZEN IMPLEMENTATIONS
============================================================

Before changing anything, inspect:

    Step 4 ProcessMonitor
    Step 4 resource telemetry schema
    Step 4 tests
    Step 5.1 benchmark manifest
    Step 5.2 adapter interface
    existing benchmark utilities

Do NOT assume class names or module paths.

The Step 4 ProcessMonitor is already frozen.

REUSE IT.

Do not create a second resource-monitoring implementation.

============================================================ 4. FROZEN STEP 4 CONTRACT
============================================================

Step 4 provides observability.

It already measures resource state including CPU-related telemetry,
memory, active requests and queue state.

Step 4 explicitly does NOT perform resource allocation.

Step 5.3 is the first stage where controlled CPU affinity is added
for SR benchmarking.

Do NOT modify the semantics of Step 4 telemetry.

Do NOT rename existing telemetry fields.

Do NOT change:

    cpu_cores_total
    cpu_cores_available
    cpu_utilization
    memory
    active_requests
    queue_depth
    cluster_id
    edge_id
    timestamps

unless a genuine compatibility issue is discovered.

If Step 5.3 needs additional benchmark-specific metadata, create it
outside the frozen Step 4 telemetry schema.

============================================================ 5. CPU AFFINITY ABSTRACTION
============================================================

Create a small CPU affinity abstraction appropriate for the project.

It should conceptually support:

    get_available_cpus()
    get_current_affinity()
    set_affinity(cpu_ids)
    restore_affinity(previous_affinity)

Use the project's existing dependencies where possible.

Do not introduce unnecessary dependencies.

The implementation must support the current development environment
(Windows) and should remain portable to Linux/Azure later.

Do NOT hard-code Windows-only behavior into the public interface.

============================================================ 6. WINDOWS IMPLEMENTATION
============================================================

The current development environment is Windows.

Use the existing supported process-management library if already
present in the repository.

If psutil is already available, prefer it.

CPU affinity should operate on the actual process.

The implementation must:

    obtain current affinity
    apply requested affinity
    execute workload
    restore original affinity

Do not leave the developer's process permanently restricted after
the benchmark/test exits.

============================================================ 7. LINUX / AZURE COMPATIBILITY
============================================================

The eventual SR inference workload may run on a Linux cloud/edge
environment.

Do not design the abstraction around Windows-only APIs.

The public abstraction should represent:

    CPU IDs
    affinity set
    current affinity

and hide OS-specific implementation details.

If Linux behavior cannot be exercised on the current machine,
document it rather than pretending it was tested.

============================================================ 8. CORE CONFIGURATION
============================================================

The CPU affinity utility must support selecting an explicit number
of logical CPUs.

Examples:

    1 CPU
    2 CPUs
    4 CPUs
    8 CPUs
    10 CPUs

However:

DO NOT assume that every machine has 10 logical CPUs.

The available CPU count must be discovered dynamically.

If a requested configuration exceeds available logical CPUs:

    fail clearly

Do NOT silently clamp:

    requested 10
        ↓
    silently use 8

That would invalidate the experiment.

============================================================ 9. CPU ID SELECTION
============================================================

Support deterministic CPU selection.

For example, if the host exposes:

    [0,1,2,3,4,5,6,7]

then a request for 4 CPUs may select:

    [0,1,2,3]

unless the project later introduces another placement policy.

The important property is:

    same configuration
        →
    deterministic CPU set

Do not randomly choose cores.

Do not implement NUMA-aware scheduling yet.

============================================================ 10. AFFINITY CONTEXT MANAGER
============================================================

Prefer a safe context-manager style interface.

Conceptually:

    with cpu_affinity(cpu_ids):
        run_workload()

The context manager must:

    1. capture current affinity
    2. apply requested affinity
    3. yield execution
    4. restore original affinity even if an exception occurs

This is critical.

A failed benchmark must not leave the Python process pinned to a
small subset of CPUs.

============================================================ 11. PROCESS SCOPE
============================================================

Affinity must be applied to the actual process executing the SR
benchmark.

Do not accidentally pin:

    the entire machine
    unrelated processes
    the parent shell
    unrelated Edge services

Step 5.3 is concerned with the benchmark process.

If the current architecture launches inference in a subprocess,
determine the correct process boundary and apply affinity there.

Do not redesign the service architecture.

============================================================ 12. PROCESSMONITOR INTEGRATION
============================================================

Reuse the frozen Step 4 ProcessMonitor.

The integration must allow Step 5 benchmarking code to associate
resource observations with the SR process being executed.

At minimum, the integration should make it possible to identify:

    process ID
    edge/cluster identity where applicable
    CPU affinity configuration
    requested thread count
    monitoring start/stop lifecycle

Do NOT change the meaning of existing ProcessMonitor metrics.

============================================================ 13. PROCESS IDENTITY
============================================================

The future benchmark must be able to answer:

    "Which CPU/resource measurements belong to this SR inference
     process?"

Therefore expose the benchmark process identity cleanly.

If ProcessMonitor already accepts a PID or process identity,
reuse that interface.

Do not duplicate process monitoring logic.

Do not use system-wide CPU utilization as a substitute for
process-specific monitoring when process-level data is available.

============================================================ 14. RESOURCE MONITOR LIFECYCLE
============================================================

Provide a clean integration lifecycle:

    create/configure
        ↓
    identify benchmark process
        ↓
    start ProcessMonitor
        ↓
    apply CPU affinity
        ↓
    model execution happens later
        ↓
    stop ProcessMonitor
        ↓
    restore CPU affinity

Do not run an actual benchmark in this step.

The lifecycle must merely make the infrastructure ready.

============================================================ 15. THREAD COUNT + AFFINITY
============================================================

Step 5.2 already supports:

    num_threads=N

Step 5.3 must keep these concepts separate.

Example CPU configuration:

    affinity = [0,1,2,3]
    num_threads = 4

This means:

    process may execute on logical CPUs 0–3
    backend is requested to use 4 compute threads

Do not assume:

    num_threads == actual physical CPU utilization

That must be empirically evaluated later.

============================================================ 16. IMPORTANT: VERIFY THE CONTROL, DON'T CLAIM PERFORMANCE
============================================================

Claude's review of Step 5.2 identified an important concern:

    accepting num_threads=N is not proof that the backend actually
    honors it.

Step 5.3 should therefore add CONTROL-LEVEL verification where
possible.

Examples:

PyTorch:

    verify the configured torch thread count using the backend's
    observable configuration.

ONNX Runtime:

    verify the session's configured intra-op thread setting where
    the API exposes it.

CPU affinity:

    verify that the process's current affinity equals the requested
    CPU set while inside the affinity context.

However:

DO NOT use inference latency to prove this yet.

Latency benchmarking belongs to Step 5.5.

The objective here is:

    "Was the requested execution configuration actually applied?"

not:

    "Did this configuration make inference faster?"

============================================================ 17. DO NOT CONFUSE THREAD COUNT WITH CPU AFFINITY
============================================================

Document this distinction clearly.

Example:

    num_threads = 8
    affinity = 4 CPUs

does NOT mean the workload has 8 CPUs.

It means the backend may create eight threads while the process is
restricted to four logical CPUs.

This can create oversubscription.

That configuration may be useful experimentally later, but Step 5.3
must preserve the distinction.

The future benchmark matrix may intentionally test:

    threads × affinity

but do not build that matrix yet.

============================================================ 18. RESOURCE CONFIGURATION OBJECT
============================================================

If appropriate, introduce a small immutable configuration structure
representing the CPU execution configuration.

Conceptually:

    CPUExecutionConfig(
        cpu_ids=[...],
        num_threads=N
    )

It may also expose:

    requested_cpu_count

Do not include runtime results.

Do not include:

    latency
    FPS
    CPU utilization
    GPU utilization
    VRAM
    PSNR
    SSIM
    VMAF

Those are later benchmark outputs.

============================================================ 19. VALIDATION RULES
============================================================

Reject:

    empty CPU set
    negative CPU IDs
    duplicate CPU IDs
    CPU IDs not available on host
    num_threads <= 0
    requested CPU count > available CPUs

Do not silently repair invalid configurations.

Errors must be explicit and useful.

============================================================ 20. AFFINITY RESTORATION
============================================================

This is mandatory.

Tests must prove:

    original affinity
        ↓
    apply benchmark affinity
        ↓
    benchmark context exits
        ↓
    original affinity restored

Also test:

    original affinity
        ↓
    apply benchmark affinity
        ↓
    exception occurs
        ↓
    original affinity restored

Do not leave the development machine/process pinned.

============================================================ 21. PROCESSMONITOR TESTING
============================================================

Add tests that verify the Step 4 ProcessMonitor can be used from the
Step 5.3 execution context.

Tests should verify:

    monitor starts
    monitor identifies the intended process
    samples are produced
    monitor stops cleanly
    no runaway monitor thread/process remains

Do NOT assert exact CPU utilization numbers.

Use invariants/ranges.

This follows the frozen Step 4 testing policy.

============================================================ 22. NO SYNTHETIC CPU VALUES
============================================================

Do NOT generate fake values such as:

    CPU = 50%
    cores_available = 4

for benchmark output.

Where measurements are exposed, they must come from the actual
ProcessMonitor.

Do not mock system resource values in integration tests unless the
test is explicitly testing an interface boundary.

============================================================ 23. OPTIONAL CONTROL TEST
============================================================

Add a small deterministic control test that:

    1. obtains current affinity
    2. selects a small valid CPU subset
    3. applies affinity
    4. verifies active affinity
    5. performs a tiny bounded CPU workload
    6. exits
    7. verifies affinity restoration

The workload must be very small.

Do NOT turn this into a benchmark.

Do NOT measure latency.

Do NOT calculate FPS.

Do NOT make the test depend on achieving a specific CPU utilization.

============================================================ 24. NO CPU ALLOCATION SCHEDULER
============================================================

This is NOT the resource scheduler.

Do not implement:

    dynamic CPU allocation
    admission control
    task scheduling
    queue scheduling
    SR request placement
    multi-edge scheduling
    ML allocation
    online learning

Step 5.3 only provides controlled execution for experiments.

============================================================ 25. NO GPU
============================================================

Do NOT implement:

    GPU detection
    GPU utilization
    VRAM monitoring
    CUDA monitoring
    GPU allocation

That is Step 5.4.

============================================================ 26. NO BENCHMARK HARNESS
============================================================

Do NOT implement:

    inference timing
    warmup timing
    repeated trials
    mean latency
    median latency
    p95 latency
    inference FPS
    throughput
    chunk processing time

That is Step 5.5.

============================================================ 27. NO QUALITY METRICS
============================================================

Do NOT implement:

    PSNR
    SSIM
    LPIPS
    VMAF

That is Step 5.6.

============================================================ 28. NO REAL-TIME FEASIBILITY
============================================================

Do NOT calculate:

    real-time ratio
    deadline miss rate
    playback feasibility
    chunk deadline
    buffer impact

That is Step 5.7 and later.

============================================================ 29. FROZEN FILES
============================================================

Do NOT modify frozen Step 0–4 implementation unless a genuine
integration defect makes it impossible to reuse the existing
ProcessMonitor.

Preferred approach:

    create Step-5-specific integration wrappers

rather than altering Step 4.

If a Step 4 change is absolutely unavoidable:

    STOP

and report the exact incompatibility before modifying it.

============================================================ 30. TESTS
============================================================

Add tests for:

1. CPU discovery.

2. Valid affinity selection.

3. Invalid CPU ID rejection.

4. Duplicate CPU rejection.

5. Empty CPU set rejection.

6. Requested CPU count greater than available CPUs rejection.

7. num_threads validation.

8. CPU affinity is actually applied.

9. CPU affinity is restored after normal execution.

10. CPU affinity is restored after exception.

11. Deterministic CPU selection.

12. ProcessMonitor starts correctly.

13. ProcessMonitor observes the intended process.

14. ProcessMonitor stops correctly.

15. No monitor thread/process is leaked.

16. PyTorch thread configuration can be observed when supported.

17. ONNX Runtime thread configuration can be observed when supported.

18. Step 5.2 adapter smoke test still works.

19. Existing Step 0–4 tests still pass.

Tests must be safe on the current Windows development machine.

Do not assume a specific total CPU count.

============================================================ 31. TESTING ON LIMITED HARDWARE
============================================================

The tests must work on machines with fewer than 10 logical CPUs.

Do not hardcode:

    "10 CPUs must exist"

Instead:

    discover available CPUs
    select a safe subset

The 1–10 CPU experimental range comes from the Rosevin methodology,
but our implementation must adapt to the actual host.

============================================================ 32. DOCUMENTATION
============================================================

Update:

    STEP5_IMPLEMENTATION.md

Add:

    Step 5.3 — CPU Affinity + ProcessMonitor Integration

Document:

    purpose
    CPU affinity
    logical CPU selection
    num_threads distinction
    ProcessMonitor integration
    process identity
    monitoring lifecycle
    Windows implementation
    Linux/Azure portability
    validation
    restoration guarantees
    tests
    limitations

Explicitly document:

    CPU affinity controls where the benchmark process may execute.

    num_threads controls backend parallelism.

    ProcessMonitor observes the actual resource state.

Also explicitly state:

    Step 5.3 does NOT perform resource allocation decisions.

============================================================ 33. REPRODUCIBILITY
============================================================

CPU configurations must be representable deterministically.

A future benchmark record should be able to say:

    requested_cpu_count = 4
    cpu_ids = [0,1,2,3]
    num_threads = 4

Do not rely only on:

    "4 cores"

because the actual CPU set matters for reproducibility.

Do not yet create the final Step 5.8 machine-readable benchmark
dataset.

============================================================ 34. FULL REGRESSION
============================================================

Run:

    python -m pytest tests/ -v

All frozen Step 0–4 tests must continue passing.

Run the new Step 5.3 tests separately if useful.

Report:

    total
    passed
    skipped
    failed

If Linux-specific behavior cannot be tested on Windows, document:

    implemented
    not exercised on current OS

Do not claim cross-platform validation that was not performed.

============================================================ 35. FINAL REPORT
============================================================

When complete, report:

1. Files created.
2. Files modified.
3. Frozen Step 0–4 files left untouched.
4. CPU affinity abstraction.
5. CPU discovery mechanism.
6. Deterministic CPU selection.
7. Affinity restoration mechanism.
8. ProcessMonitor integration.
9. Process identity mechanism.
10. num_threads integration.
11. Backend configuration verification.
12. Windows behavior.
13. Linux portability status.
14. Tests added.
15. Full pytest results.
16. Any limitations.
17. Confirmation that no GPU monitoring was implemented.
18. Confirmation that no benchmark timing was implemented.
19. Confirmation that 5.4–5.9 were NOT implemented.

============================================================
STOP CONDITION
============================================================

STOP HERE.

Do NOT begin:

    5.4 GPU measurement
    5.5 Inference benchmark harness
    5.6 Quality evaluation
    5.7 FPS / real-time feasibility
    5.8 Machine-readable benchmark dataset
    5.9 Validation / reproducibility report

# Step 5.3 must be reviewed by Claude before proceeding.
