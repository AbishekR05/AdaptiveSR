============================================================
ADAPTIVESR — STEP 5.4 IMPLEMENTATION
GPU MEASUREMENT
============================================================

IMPORTANT:

Steps 0–4 are COMPLETE and FROZEN.

Step 5 contains:

    5.1 Benchmark dataset / test-video preparation     [FROZEN]
    5.2 SR model runner adapter interface              [FROZEN]
    5.3 CPU affinity + ProcessMonitor integration      [FROZEN]
    5.4 GPU measurement                                [THIS STEP]
    5.5 Inference benchmark harness
    5.6 Quality evaluation
    5.7 FPS / real-time feasibility analysis
    5.8 Machine-readable benchmark dataset
    5.9 Validation + reproducibility report

IMPLEMENT ONLY STEP 5.4.

DO NOT IMPLEMENT 5.5–5.9.

============================================================

1. # OBJECTIVE

Step 5.4 establishes the GPU-side measurement infrastructure needed
to compare SR inference under GPU execution.

Step 5.3 already established the controlled CPU execution side:

    CPU affinity
        +
    num_threads
        +
    ProcessMonitor

Step 5.4 must establish the equivalent GPU observability layer:

    GPU discovery
        +
    GPU identity
        +
    CUDA availability
        +
    GPU utilization measurement
        +
    GPU memory measurement
        +
    GPU execution metadata

The future benchmark flow will eventually be:

    benchmark case
         ↓
    select model
         ↓
    select scale
         ↓
    select device
         ↓
    GPU measurement infrastructure
         ↓
    inference
         ↓
    latency / throughput benchmark

BUT:

Step 5.4 itself must NOT implement the benchmark harness.

============================================================ 2. FIRST — INSPECT THE REPOSITORY
============================================================

Before modifying anything, inspect:

    Step 4 ProcessMonitor
    Step 5.2 adapter interface
    Step 5.3 CPU control implementation
    existing CUDA/PyTorch utilities
    existing GPU detection code
    existing configuration/schema files
    existing tests

Do not assume filenames.

Reuse existing infrastructure where appropriate.

Do not create duplicate GPU monitoring utilities if the repository
already contains an equivalent implementation.

============================================================ 3. SCOPE BOUNDARY
============================================================

Step 5.4 provides GPU measurement infrastructure.

It MUST implement:

    GPU discovery
    CUDA availability detection
    GPU identity
    GPU memory information
    GPU utilization sampling where supported
    GPU measurement lifecycle
    GPU measurement validation
    GPU execution metadata

It MUST NOT implement:

    inference timing
    FPS
    throughput
    latency statistics
    warmup
    repeated benchmark trials
    PSNR
    SSIM
    LPIPS
    VMAF
    real-time feasibility
    benchmark result dataset
    scheduling
    GPU allocation algorithms

Those belong to later steps.

============================================================ 4. GPU VS GPU ALLOCATION
============================================================

Do not confuse:

    GPU measurement

with:

    GPU resource allocation.

Step 5.4 should be able to identify and observe a selected GPU.

It does NOT implement:

    dynamic GPU scheduling
    GPU cluster allocation
    GPU partitioning
    MIG scheduling
    resource admission control
    multi-GPU load balancing

The project may later use these measurements when making resource
allocation decisions, but that is outside Step 5.4.

============================================================ 5. CUDA AVAILABILITY
============================================================

Provide a reliable mechanism to determine:

    CUDA available?
    number of CUDA devices
    device IDs

The result must distinguish:

    CUDA unavailable
    CUDA available but no device
    CUDA available with one device
    CUDA available with multiple devices

Do not silently fall back from CUDA to CPU.

If CUDA is requested but unavailable:

    raise a clear error.

Do not report CPU execution as GPU execution.

============================================================ 6. GPU DEVICE DISCOVERY
============================================================

Expose a device discovery mechanism conceptually equivalent to:

    list_gpus()

Each discovered GPU should expose stable metadata such as:

    device_id
    device_name
    compute capability if available
    total_memory
    CUDA availability
    backend/runtime information where available

Do not invent metadata.

If a property cannot be obtained from the current backend:

    return null/unknown
    or explicitly mark it unavailable

Do not fabricate values.

============================================================ 7. GPU IDENTITY
============================================================

The future benchmark record must be able to distinguish GPUs.

At minimum expose:

    logical CUDA device ID
    GPU name
    total VRAM

Where reliably available, also expose:

    compute capability
    CUDA runtime version
    driver version
    PyTorch CUDA version

Do not require every optional property if the current environment
cannot expose it.

The benchmark must never identify a GPU only as:

    "GPU"

because multiple GPUs may exist.

============================================================ 8. GPU MEMORY MEASUREMENT
============================================================

Expose GPU memory information.

At minimum:

    total_memory
    allocated_memory
    reserved_memory

where the backend provides those values.

Where possible, also expose:

    free_memory

Be explicit about what each value means.

For PyTorch:

    allocated memory
        =
    memory currently allocated by tensors

    reserved memory
        =
    memory reserved by PyTorch's caching allocator

Do not incorrectly label reserved memory as actual tensor usage.

Do not silently mix:

    process memory
    device-wide memory
    PyTorch allocator memory

These are different measurements.

============================================================ 9. DEVICE-WIDE VS PROCESS GPU MEMORY
============================================================

If the backend allows it, distinguish:

    device-wide GPU memory
    process-associated GPU memory

Do not claim process-level GPU memory if the backend only provides
device-level information.

The measurement schema should make the distinction explicit.

Example conceptual fields:

    gpu_memory_total
    gpu_memory_free
    process_gpu_memory_allocated
    process_gpu_memory_reserved

Only populate fields that are genuinely supported.

============================================================ 10. GPU UTILIZATION
============================================================

Implement GPU utilization sampling where supported.

The measurement should represent actual GPU utilization rather than:

    CUDA available = true

or:

    GPU exists = true

These are not utilization measurements.

Use the project's available GPU telemetry mechanism.

If NVIDIA Management Library / NVML is available in the environment,
it may be used for device-level utilization.

If PyTorch exposes only allocator information, do not pretend that
allocator usage is GPU utilization.

Keep these metrics separate:

    GPU utilization
    GPU memory utilization
    allocated VRAM

============================================================ 11. NVML / OPTIONAL DEPENDENCY
============================================================

If NVML is used:

    detect availability explicitly.

Do not make the entire AdaptiveSR test suite fail on a machine where
NVML/GPU hardware is unavailable.

GPU measurement should degrade gracefully for environments without
NVIDIA hardware.

However:

DO NOT silently replace missing GPU measurements with fake zeros.

For example:

    unavailable GPU utilization
        ≠
    0%

Use:

    null
    unavailable
    explicit exception

according to the project's existing conventions.

============================================================ 12. GPU SAMPLING
============================================================

Provide a measurement lifecycle conceptually equivalent to:

    start()
        ↓
    sample()
        ↓
    sample()
        ↓
    ...
        ↓
    stop()

or another project-consistent abstraction.

Sampling must produce structured observations.

Each observation should be associated with:

    timestamp
    GPU device ID

and, where available:

    GPU utilization
    memory utilization
    total memory
    free memory
    allocated memory
    reserved memory

Do not calculate benchmark latency here.

============================================================ 13. SAMPLING INTERVAL
============================================================

GPU sampling must use an explicit configurable interval.

Do not hard-code an undocumented sampling rate.

The default should be reasonable for telemetry without creating
significant overhead.

Document the default.

The future benchmark harness may override the interval.

Do not optimize the interval using benchmark results in this step.

============================================================ 14. GPU MEASUREMENT CONTEXT
============================================================

Prefer a safe lifecycle/context-manager pattern.

Conceptually:

    with gpu_measurement(device_id) as monitor:
        run_workload()

The monitor must:

    start
    collect observations
    stop cleanly
    expose collected samples

If an exception occurs:

    monitoring must stop cleanly.

Do not leave background threads/processes running.

============================================================ 15. RELATIONSHIP TO PROCESSMONITOR
============================================================

Do NOT modify the frozen Step 4 ProcessMonitor.

Step 4 ProcessMonitor observes CPU/process/system resource state.

Step 5.4 adds GPU-specific measurement.

Keep the responsibilities separate:

    ProcessMonitor
        →
    CPU/process/system telemetry

    GPUMonitor
        →
    GPU/device telemetry

Do not merge GPU fields into the frozen ProcessMonitor schema unless
there is an existing project-level contract explicitly requiring it.

============================================================ 16. PROCESS IDENTITY
============================================================

If GPU telemetry can identify the current process, expose that
information.

If the chosen GPU telemetry mechanism only exposes device-level
utilization, clearly document that limitation.

Do not fabricate per-process GPU utilization.

The future benchmark must be able to distinguish:

    "GPU utilization of device 0"

from:

    "GPU utilization caused specifically by this SR process."

These are not equivalent.

============================================================ 17. MULTI-GPU SUPPORT
============================================================

The implementation must support discovering multiple GPUs.

At minimum:

    device 0
    device 1
    ...

must be independently identifiable.

A caller must be able to request a specific device:

    cuda:0
    cuda:1

where available.

Do not automatically distribute inference across all GPUs.

Do not implement multi-GPU inference.

Do not implement GPU load balancing.

============================================================ 18. DEVICE VALIDATION
============================================================

Reject:

    negative GPU device IDs
    GPU IDs outside the discovered range
    CUDA device requests when CUDA is unavailable

Errors must be explicit.

Example:

    Requested CUDA device 1, but only device 0 is available.

Do not silently fall back to another GPU.

============================================================ 19. PYTORCH INTEGRATION
============================================================

If PyTorch is the SR backend:

    ensure the requested CUDA device can be selected explicitly.

Do not globally change the application's CUDA device state unless
the existing project architecture requires it.

Prefer explicit device objects/configuration.

The adapter from Step 5.2 remains responsible for model execution.

Step 5.4 provides GPU measurement around that execution.

============================================================ 20. CUDA SYNCHRONIZATION — IMPORTANT
============================================================

CUDA operations are asynchronous.

DO NOT implement benchmark timing in Step 5.4.

However, document the important distinction:

    GPU work may still be executing after the CPU-side model call
    returns.

Therefore future latency benchmarking in Step 5.5 must use appropriate
CUDA synchronization around timing boundaries.

Step 5.4 should NOT implement that timing mechanism.

This is a methodological requirement for Step 5.5.

============================================================ 21. GPU WARMUP
============================================================

Do not implement benchmark warmup here.

Document that future benchmark execution must distinguish:

    initialization
    CUDA context creation
    model loading
    first inference
    warmed inference

Step 5.4 only provides measurement infrastructure.

============================================================ 22. GPU MEASUREMENT OVERHEAD
============================================================

GPU monitoring itself may introduce overhead.

Do not attempt to subtract that overhead mathematically.

Document the measurement-system limitation.

Use a conservative sampling interval.

Future Step 5.5 must keep the same monitoring configuration across
comparable benchmark runs.

============================================================ 23. DETERMINISTIC METADATA
============================================================

GPU identity information should be stable enough for benchmark
records.

At minimum:

    device_id
    device_name
    total_memory

Where available, record:

    compute capability
    driver/runtime versions

Do not use only a user-defined label such as:

    "fast_gpu"

because that is not reproducible.

============================================================ 24. GPU MEASUREMENT SCHEMA
============================================================

Create a structured GPU measurement object appropriate to the
existing project architecture.

Conceptually:

    GPUSnapshot(
        timestamp,
        device_id,
        gpu_name,
        utilization_percent,
        memory_total,
        memory_free,
        memory_allocated,
        memory_reserved
    )

Use the project's naming conventions.

Do NOT blindly copy this exact schema if the repository already has
an established telemetry model.

Clearly distinguish:

    unavailable
    zero
    not measured

============================================================ 25. TESTING WITHOUT A GPU
============================================================

The current development environment may not have a usable CUDA GPU.

Tests must therefore support:

    GPU available
    GPU unavailable

When GPU hardware is unavailable:

    capability tests should still pass.

GPU-specific integration tests may be skipped with an explicit reason.

Do NOT fake a GPU.

Do NOT monkey-patch CUDA globally just to make the implementation
appear operational.

============================================================ 26. GPU TESTS
============================================================

Add tests for:

1. CUDA availability detection.

2. GPU enumeration.

3. Device metadata retrieval.

4. Invalid GPU ID rejection.

5. GPU memory snapshot schema.

6. GPU utilization availability handling.

7. Sampling lifecycle.

8. Sampling interval configuration.

9. Clean monitor shutdown.

10. Exception-safe shutdown.

11. Multi-GPU device selection logic where hardware permits.

12. No-GPU graceful behavior.

13. Step 5.2 CUDA adapter compatibility.

14. Existing Step 0–4 tests remain passing.

Do not assert exact GPU utilization percentages.

Do not assert exact memory values.

Use structural/range/invariant checks.

============================================================ 27. GPU SMOKE TEST
============================================================

If a usable CUDA GPU exists:

    run a very small bounded GPU operation.

The test should verify:

    CUDA device is selectable
    GPU measurement can start
    at least one valid snapshot can be obtained
    measurement can stop cleanly

This is NOT a benchmark.

Do not report latency.

Do not report FPS.

Do not compare models.

============================================================ 28. NO GPU BENCHMARK MATRIX
============================================================

Do NOT create combinations such as:

    FSRCNN × GPU0 × scale2
    FSRCNN × GPU0 × scale4
    Real-ESRGAN × GPU0 × scale2

Those belong to Step 5.5.

Step 5.4 only creates the infrastructure that will make those
experiments measurable later.

============================================================ 29. NO CPU CHANGES
============================================================

Do not modify Step 5.3 CPU affinity behavior.

Do not modify:

    CPUExecutionConfig
    cpu_affinity_context
    BenchmarkProcessMonitor

unless a genuine compatibility issue prevents GPU integration.

If such a conflict exists:

    STOP

and report it before modifying frozen 5.3 code.

============================================================ 30. DOCUMENTATION
============================================================

Update:

    STEP5_IMPLEMENTATION.md

Add:

    Step 5.4 — GPU Measurement

Document:

    purpose
    CUDA detection
    GPU discovery
    device identity
    memory metrics
    utilization metrics
    NVML availability
    sampling lifecycle
    process/device distinction
    multi-GPU support
    PyTorch integration
    CUDA asynchronous execution
    warmup considerations
    measurement overhead
    no-GPU behavior
    limitations
    tests

Explicitly state:

    Step 5.4 provides GPU measurement infrastructure.

Explicitly state:

    Step 5.4 does NOT benchmark SR models.

============================================================ 31. FULL REGRESSION
============================================================

Run:

    python -m pytest tests/ -v

All existing Step 0–4 tests must continue passing.

Run Step 5.4 GPU tests separately if useful.

Report:

    total
    passed
    skipped
    failed

If the current machine has no usable CUDA GPU:

    clearly report that GPU runtime smoke tests were skipped.

Do not claim GPU runtime validation that was not performed.

============================================================ 32. FINAL REPORT
============================================================

When complete, report:

1. Files created.
2. Files modified.
3. Frozen Step 0–4 files left untouched.
4. GPU discovery implementation.
5. CUDA availability behavior.
6. GPU identity metadata.
7. GPU memory measurement.
8. GPU utilization measurement.
9. NVML availability/limitations.
10. Sampling lifecycle.
11. Multi-GPU support.
12. Device validation.
13. No-GPU behavior.
14. Tests added.
15. Full pytest results.
16. Whether a real GPU smoke test was executed.
17. Any unavailable optional dependencies.
18. Confirmation that no benchmark timing was implemented.
19. Confirmation that 5.5–5.9 were NOT implemented.

============================================================
STOP CONDITION
============================================================

STOP HERE.

Do NOT begin:

    5.5 Inference benchmark harness
    5.6 Quality evaluation
    5.7 FPS / real-time feasibility
    5.8 Machine-readable benchmark dataset
    5.9 Validation / reproducibility report

# Step 5.4 must be reviewed by Claude before proceeding.
