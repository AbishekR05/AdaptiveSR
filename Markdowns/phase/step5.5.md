============================================================
ADAPTIVESR — STEP 5.5 IMPLEMENTATION
INFERENCE BENCHMARK HARNESS
============================================================

IMPORTANT:

Steps 0–4 are COMPLETE and FROZEN.

Step 5:

    5.1 Benchmark dataset / test-video preparation     [FROZEN]
    5.2 SR model runner adapter interface              [FROZEN]
    5.3 CPU affinity + ProcessMonitor integration      [FROZEN]
    5.4 GPU measurement                                [FROZEN]
    5.5 Inference benchmark harness                   [THIS STEP]
    5.6 Quality evaluation
    5.7 FPS / real-time feasibility analysis
    5.8 Machine-readable benchmark dataset
    5.9 Validation + reproducibility report

IMPLEMENT ONLY STEP 5.5.

DO NOT IMPLEMENT 5.6–5.9.

============================================================ 0. PURPOSE
============================================================

Step 5.5 converts the execution-control and measurement
infrastructure from Steps 5.2–5.4 into a reproducible inference
benchmark harness.

The harness must measure actual SR inference performance across
controlled configurations.

The central output of this step is trustworthy inference-performance
data.

Conceptually:

    benchmark case
        ↓
    load benchmark input
        ↓
    select SR model
        ↓
    select scale
        ↓
    select device
        ↓
    configure CPU resources if CPU
        ↓
    configure GPU measurement if CUDA
        ↓
    warm up
        ↓
    synchronize if CUDA
        ↓
    timed inference
        ↓
    synchronize if CUDA
        ↓
    collect resource observations
        ↓
    repeat
        ↓
    aggregate latency statistics
        ↓
    produce structured benchmark result

============================================================

1. # STRICT STEP BOUNDARY

IMPLEMENT:

    benchmark configuration
    benchmark case representation
    warmup handling
    repeated inference trials
    CPU timing
    CUDA timing
    CUDA synchronization
    latency aggregation
    throughput calculation where appropriate
    ProcessMonitor integration
    GPUMonitor integration
    benchmark result schema
    deterministic execution metadata
    validation of benchmark inputs
    benchmark tests

DO NOT IMPLEMENT:

    PSNR
    SSIM
    LPIPS
    VMAF
    perceptual quality
    quality comparison
    real-time feasibility
    playback deadline analysis
    chunk scheduling
    adaptive resource allocation
    online learning
    ML scheduling
    final benchmark dataset packaging
    final reproducibility report

Those belong to later steps.

============================================================ 2. IMPORTANT METHODOLOGICAL RULE
============================================================

The benchmark must distinguish:

    model loading
    initialization
    warmup
    measured inference

Do NOT include model construction or weight loading in steady-state
inference latency unless a benchmark configuration explicitly asks
for end-to-end initialization latency.

The default Step 5.5 measurement is:

    warmed model
    +
    one inference operation
    =
    measured inference latency

============================================================ 3. BENCHMARK CASE
============================================================

Create a structured benchmark configuration.

It should identify at minimum:

    model
    scale
    input sample
    device
    CPU configuration if applicable
    warmup count
    measured trial count
    GPU sampling configuration if applicable

Conceptually:

    BenchmarkConfig(
        model_id,
        scale,
        input_id,
        device,
        cpu_config,
        warmup_runs,
        measured_runs,
        gpu_sampling_interval
    )

Use existing project schemas/conventions where possible.

Do not duplicate model definitions already provided by Step 5.2.

============================================================ 4. DEVICE POLICY
============================================================

Supported execution modes:

    CPU
    CUDA GPU

CPU:

    use Step 5.3 CPUExecutionConfig
    use CPU affinity
    use num_threads

CUDA:

    explicitly select cuda:N
    use Step 5.4 GPU measurement
    do NOT silently fall back to CPU

Before constructing GPUMonitor:

    call get_cuda_availability()

Then:

    if status == AVAILABLE:
        validate requested GPU
        construct GPU measurement infrastructure

    otherwise:
        fail explicitly for a requested GPU benchmark

Do NOT detect GPU availability by catching a generic RuntimeError
from GPUMonitor.

============================================================ 5. CPU RESOURCE CONFIGURATION
============================================================

For CPU benchmarks, preserve the Step 5.3 distinction:

    CPU affinity
        =
    which logical CPUs the process may execute on

    num_threads
        =
    backend thread count

A benchmark configuration must record both.

Example:

    cpu_ids = [0,1,2,3]
    num_threads = 4

Do not assume these are interchangeable.

Allow intentional oversubscription configurations:

    cpu_ids = [0,1]
    num_threads = 4

because Step 5.3 explicitly supports studying this behavior.

Do not automatically normalize or modify the requested values.

============================================================ 6. GPU RESOURCE CONFIGURATION
============================================================

For GPU benchmarks record:

    device_id
    GPU name
    total VRAM
    CUDA runtime information where available

Do not implement GPU allocation.

Do not automatically select the least-loaded GPU.

Do not automatically distribute workloads.

The benchmark configuration must explicitly identify the GPU.

============================================================ 7. INPUT PREPARATION
============================================================

Reuse the Step 5.1 benchmark corpus.

Do not create another video/image dataset.

Reuse the existing benchmark input loading mechanism.

Each benchmark input must have a stable identifier.

Record relevant input metadata such as:

    input_id
    width
    height
    channels
    scale

Do not include video decode time in model inference latency unless
explicitly configured.

The default benchmark target is:

    prepared input
        →
    SR model inference
        →
    output

This isolates SR compute.

============================================================ 8. INPUT IMMUTABILITY
============================================================

The same logical benchmark input must be reused across comparable
trials.

Do not mutate the input between trials.

Avoid hidden preprocessing differences between:

    CPU run
    GPU run

The input representation should be equivalent across devices.

If preprocessing is device-specific, document it explicitly.

============================================================ 9. MODEL INITIALIZATION
============================================================

Use the frozen Step 5.2 adapter interface.

Conceptually:

    adapter.initialize(
        device=device,
        scale=scale,
        num_threads=num_threads
    )

Do not bypass the adapter and call model internals directly.

The benchmark harness measures the adapter's supported execution path.

============================================================ 10. MODEL LOADING TIME
============================================================

By default, exclude:

    model construction
    checkpoint loading
    CUDA model transfer
    CUDA context initialization

from steady-state inference latency.

These happen before warmup.

If the existing adapter initializes lazily, ensure the warmup phase
absorbs one-time initialization before measured trials.

Document this explicitly.

============================================================ 11. WARMUP
============================================================

Warmup is mandatory before measured GPU trials.

Warmup exists to absorb:

    CUDA context initialization
    kernel initialization
    allocator setup
    cache effects
    lazy backend initialization

Default:

    warmup_runs = 3

Make the value configurable.

Do not include warmup runs in latency statistics.

For CPU benchmarks, warmup is also recommended and should be applied
consistently.

Do not silently use warmup results as measured results.

============================================================ 12. CUDA SYNCHRONIZATION
============================================================

THIS IS CRITICAL.

CUDA execution is asynchronous.

Never measure:

    start = perf_counter()
    adapter.process(input)
    end = perf_counter()

for a CUDA workload.

That can measure CPU dispatch time rather than actual GPU execution.

For CUDA:

    torch.cuda.synchronize(device)
    start = perf_counter()

    output = adapter.process(input)

    torch.cuda.synchronize(device)
    end = perf_counter()

    latency = end - start

The synchronization after process() is mandatory.

Also synchronize before the start boundary to ensure no previous
GPU work leaks into the measured interval.

Do not use synchronization around CPU timing unless required by the
backend.

============================================================ 13. CUDA TIMING EVENTS
============================================================

For the primary benchmark latency, wall-clock timing with explicit
CUDA synchronization is acceptable.

If CUDA events are used, they must be implemented correctly and
documented.

Do not use CPU wall-clock timing without synchronization.

Do not mix timing methodologies between comparable GPU trials.

Choose one primary method and use it consistently.

Preferred default:

    torch.cuda.synchronize()
    perf_counter()
    inference
    torch.cuda.synchronize()
    perf_counter()

============================================================ 14. CPU TIMING
============================================================

For CPU:

    start = perf_counter()
    output = adapter.process(input)
    end = perf_counter()

    latency = end - start

The CPU benchmark must execute inside:

    cpu_affinity_context

and use the configured:

    num_threads

The CPU affinity must be active for the measured inference.

============================================================ 15. TIMING BOUNDARY
============================================================

The measured interval should contain only the operation being
benchmarked.

Default:

    START
        ↓
    adapter.process(input)
        ↓
    END

Do NOT include:

    file loading
    video decoding
    dataset iteration
    logging
    result serialization
    metric calculation
    model loading
    benchmark report generation

unless explicitly requested by a separate benchmark mode.

============================================================ 16. OUTPUT VALIDATION
============================================================

After every measured inference, validate the adapter output using
the Step 5.2 contract.

At minimum verify:

    output exists
    correct dimensions
    expected scale
    valid tensor/image structure

Do not calculate quality metrics here.

Output validation must not be included in the timed interval.

============================================================ 17. TRIAL COUNT
============================================================

Default measured trials:

    20

Make configurable.

Do not use only one inference.

Single-run latency is too noisy for reliable comparison.

Each trial must be independently recorded.

Example:

    trial 1
    trial 2
    ...
    trial 20

Warmup trials must remain separate.

============================================================ 18. LATENCY STATISTICS
============================================================

For each benchmark case calculate at minimum:

    count
    mean latency
    median latency
    minimum latency
    maximum latency
    standard deviation
    p95 latency

Use seconds internally or another consistent unit.

Expose milliseconds in human-readable output if useful.

Do not confuse:

    p95 latency
    with
    maximum latency

They are different statistics.

============================================================ 19. PERCENTILE DEFINITION
============================================================

Use a deterministic, documented percentile method.

Do not implement a hand-written approximation without documenting it.

Use the project's available numerical dependency if already present.

The benchmark report must record the method used for p95.

Do not silently change percentile methodology between runs.

============================================================ 20. THROUGHPUT
============================================================

Compute inference throughput only from measured inference latency.

For sequential single-sample inference:

    FPS = 1 / mean_latency_seconds

Do NOT calculate FPS from:

    wall-clock benchmark duration including setup
    monitoring time
    file I/O
    logging
    serialization

This is inference throughput, not end-to-end streaming throughput.

Document the distinction.

============================================================ 21. SEQUENTIAL EXECUTION
============================================================

Default benchmark execution is sequential:

    inference
    ↓
    record
    ↓
    next inference

Do not introduce batching.

Do not run concurrent requests.

Do not introduce asynchronous request queues.

The objective is to characterize per-inference SR compute.

============================================================ 22. BATCH SIZE
============================================================

Default:

    batch_size = 1

Do not benchmark larger batches in Step 5.5 unless explicitly
required by the existing project design.

If the adapter inherently operates on a batch dimension, preserve the
existing contract but keep the benchmark case explicit.

============================================================ 23. PROCESSMONITOR INTEGRATION
============================================================

For CPU benchmark cases, integrate the frozen Step 5.3
BenchmarkProcessMonitor.

The lifecycle must remain:

    CPU affinity
        ↓
    ProcessMonitor
        ↓
    workload
        ↓
    monitor shutdown
        ↓
    affinity restoration

Do not modify Step 5.3 semantics.

Collect resource snapshots separately from latency.

Do not include monitoring overhead in latency timing.

Do not subtract monitoring overhead mathematically.

============================================================ 24. GPUMONITOR INTEGRATION
============================================================

For GPU benchmark cases, integrate Step 5.4.

Use periodic monitoring for sustained workload characterization.

However, DO NOT assume that a 0.5-second periodic sampler captures
every individual short inference.

The benchmark must therefore also use synchronous GPU boundary
snapshots where appropriate for per-operation memory/state information.

Keep these concepts separate:

    latency
        =
    explicit timing boundary

    GPU utilization
        =
    periodic telemetry

    GPU memory boundary
        =
    before/after snapshots

============================================================ 25. GPU MEMORY ALLOCATOR CACHING
============================================================

IMPORTANT.

PyTorch's caching allocator can retain reserved memory after an
inference.

Therefore:

    memory_reserved_after
        -
    memory_reserved_before

must NOT automatically be interpreted as:

    "true fresh VRAM required by this inference."

The same applies when performing back-to-back trials.

Record:

    allocated memory
    reserved memory

as distinct quantities.

Do not automatically call:

    torch.cuda.empty_cache()

between every measured trial.

Why:

    empty_cache() changes allocator behavior and adds overhead.

The default benchmark should represent realistic repeated inference
behavior.

If an isolated allocation experiment is needed later, that should be
an explicitly separate benchmark mode.

============================================================ 26. GPU MEMORY BOUNDARY
============================================================

For each measured GPU operation, if synchronous boundary measurement
is enabled:

    snapshot_before
    ↓
    timed inference
    ↓
    snapshot_after

Do not interpret the memory delta as exact peak VRAM consumption.

It represents the change in observed allocator/device state.

============================================================ 27. GPU UTILIZATION INTERPRETATION
============================================================

Do not report:

    "SR process GPU utilization = X%"

when the measurement source is device-wide NVML utilization.

Report it as:

    device GPU utilization

and document:

    it includes all processes using the GPU.

If another unrelated process is using the same GPU during the
benchmark, the utilization value may be contaminated.

The benchmark metadata must record the utilization source.

============================================================ 28. RESOURCE CONTAMINATION
============================================================

The benchmark must record whether the selected GPU is being used
by other processes if this can be determined reliably.

Do not attempt to forcibly terminate or interfere with other
processes.

If reliable process-level GPU accounting is unavailable:

    document the limitation.

Do not fabricate isolation.

============================================================ 29. CPU MONITOR CONTENTION
============================================================

Preserve the Step 5.3 limitation:

    ProcessMonitor runs inside the same affinity-constrained process.

At low CPU counts, monitoring may consume a small amount of CPU time.

Do not subtract it mathematically.

Use the same monitor configuration across comparable CPU runs.

Record the sampling interval in benchmark metadata.

============================================================ 30. GPU MONITOR OVERHEAD
============================================================

Preserve the Step 5.4 limitation:

    GPU monitoring introduces non-zero measurement overhead.

Do not subtract it mathematically.

Keep monitoring configuration consistent across comparable GPU runs.

Record:

    GPU sampling interval
    GPU monitoring mode
    utilization source

in benchmark metadata.

============================================================ 31. BENCHMARK CASE IDENTITY
============================================================

Every benchmark result must identify the complete experimental
configuration.

At minimum:

    benchmark_id
    timestamp
    model_id
    scale
    input_id
    input_width
    input_height
    device_type
    device_id

For CPU:

    cpu_ids
    cpu_count
    num_threads

For GPU:

    gpu_name
    total_vram
    CUDA version if available

Also record:

    warmup_runs
    measured_runs
    batch_size
    benchmark software version/configuration where available

============================================================ 32. HOST METADATA
============================================================

Record enough host metadata to interpret results.

Where already available through project utilities, record:

    operating_system
    Python version
    PyTorch version

For CPU:

    logical CPU count

For GPU:

    CUDA runtime
    GPU name
    driver version where available

Do not collect unnecessary personal/system information.

============================================================ 33. REPRODUCIBILITY SEED
============================================================

If random operations exist in preprocessing or model execution,
support deterministic seeding.

If no randomness exists in the inference path, do not invent a
random seed just for the sake of metadata.

Record whether deterministic execution was enabled.

Do not silently alter model behavior solely to make the benchmark
deterministic.

============================================================ 34. RESULT SCHEMA
============================================================

Create a structured benchmark result object.

Conceptually:

    BenchmarkResult(
        benchmark_id,
        config,
        trial_latencies,
        latency_statistics,
        throughput_fps,
        resource_summary,
        metadata
    )

Do not blindly copy this exact structure if existing project schemas
provide an appropriate foundation.

Each trial should remain individually inspectable.

Do not store only:

    mean_latency

because later analysis may require the raw trial values.

============================================================ 35. RAW TRIAL PRESERVATION
============================================================

Preserve every measured trial latency.

Example:

    latencies_ms = [
        42.1,
        41.7,
        43.0,
        ...
    ]

Do not discard the raw values after calculating the mean.

Do not round raw measurements excessively.

Human-readable reporting may round them later.

============================================================ 36. FAILURE HANDLING
============================================================

If a single trial fails:

    record the failure clearly.

Do not silently treat failure as:

    latency = 0

Do not silently drop failures.

A benchmark case with failed trials must expose:

    successful trial count
    failed trial count
    failure information

If the benchmark policy requires aborting the case after a failure,
make that explicit.

Do not fabricate complete trial counts.

============================================================ 37. MODEL/DEVICE COMPATIBILITY
============================================================

Before running a benchmark case:

    verify the model supports the requested scale
    verify the adapter supports the requested device
    verify CUDA availability if GPU
    verify CPU configuration if CPU

Fail before starting the trial loop when configuration is invalid.

Do not silently switch:

    GPU → CPU
    unsupported scale → another scale
    unavailable device → another device

============================================================ 38. REAL-ESRGAN ×2 WARNING
============================================================

Step 5.2 specifically required verification that Real-ESRGAN scale 2
is a genuine ×2 inference path.

The benchmark harness must use the verified adapter capability.

Do not assume:

    configured scale = actual model compute scale

If Step 5.2 exposes verified capability metadata, preserve it in the
benchmark result.

Do not implement a fake scale correction in the benchmark harness.

============================================================ 39. MODEL LOADING REUSE
============================================================

For repeated trials of the same benchmark case:

    initialize model once
    warm up once
    run measured trials

Do NOT reload the model for every trial.

Reloading would measure model initialization repeatedly rather than
steady-state inference.

============================================================ 40. SCALE
============================================================

Benchmark scale must be explicit.

At minimum support the scales already validated by Step 5.2.

Do not invent unsupported scales.

The benchmark result must record the requested scale.

============================================================ 41. INPUT RESOLUTION
============================================================

Record the actual input dimensions:

    width
    height

Do not infer them from filenames.

This is necessary because latency depends strongly on input spatial
size.

The benchmark result must therefore be interpretable as:

    model × scale × input resolution × resource configuration

============================================================ 42. BENCHMARK MATRIX
============================================================

Do not hard-code one benchmark case.

The harness must be capable of executing multiple configurations.

Conceptually:

    models
        ×
    scales
        ×
    input resolutions
        ×
    CPU configurations
        ×
    GPU configurations

However, do NOT automatically run every possible combination on
construction.

The harness should accept an explicit list of BenchmarkConfig cases.

This prevents accidental massive benchmark runs.

============================================================ 43. FAIR COMPARISON RULE
============================================================

When comparing two models:

    same input
    same scale
    same resource configuration
    same warmup policy
    same trial count
    same monitoring configuration

must be used whenever technically applicable.

Do not compare:

    GPU model A
    against
    CPU model B

and call it a model comparison.

The benchmark harness must preserve enough metadata to detect such
mismatches later.

============================================================ 44. CPU VS GPU COMPARISON
============================================================

Step 5.5 may produce both:

    CPU inference measurements
    GPU inference measurements

But DO NOT yet make the final project decision:

    "CPU is better"
    "GPU is better"

That decision requires analysis across:

    latency
    resource usage
    quality
    real-time feasibility
    energy/cost where applicable

Those belong to later project stages.

Step 5.5 only produces trustworthy measurements.

============================================================ 45. BENCHMARK EXECUTION ORDER
============================================================

For each benchmark case:

    1. Validate configuration.

    2. Prepare input.

    3. Initialize adapter.

    4. Configure CPU affinity if CPU.

    5. Verify CUDA availability/device if GPU.

    6. Start appropriate resource monitor.

    7. Perform warmup runs.

    8. Synchronize CUDA after warmup if GPU.

    9. Begin measured trials.

    10. For GPU:
            synchronize
            capture timing start
            inference
            synchronize
            capture timing end

        For CPU:
            capture timing start
            inference
            capture timing end

    11. Validate output outside the timing boundary.

    12. Record resource observations.

    13. Repeat until measured trial count is reached.

    14. Stop monitors.

    15. Restore CPU affinity if applicable.

    16. Aggregate statistics.

    17. Create BenchmarkResult.

============================================================ 46. IMPORTANT TIMING RULE
============================================================

For every measured GPU trial:

    torch.cuda.synchronize(device)
    start = perf_counter()

    output = adapter.process(input)

    torch.cuda.synchronize(device)
    end = perf_counter()

Do not omit the synchronization before the start.

Do not omit the synchronization after inference.

This is a hard correctness requirement.

============================================================ 47. TIMING CLOCK
============================================================

Use:

    time.perf_counter()

for CPU wall-clock timing.

It is monotonic and appropriate for elapsed-time measurement.

Do not use:

    time.time()

for primary latency measurement.

============================================================ 48. WARMUP + RESOURCE MONITORING
============================================================

Warmup may occur while monitors are active.

However, distinguish warmup resource samples from measured-trial
resource samples where practical.

Do not mix warmup observations into the measured-trial latency
statistics.

If the monitor cannot cleanly separate them, document the limitation.

============================================================ 49. RESOURCE SUMMARY
============================================================

The benchmark result may contain summarized resource observations.

Examples:

CPU:

    mean_cpu_utilization
    peak_cpu_utilization
    available_cpu_count

GPU:

    mean_device_gpu_utilization
    peak_observed_device_gpu_utilization
    memory_before
    memory_after
    allocated_before
    allocated_after
    reserved_before
    reserved_after

BUT:

    distinguish observed peak from actual peak.

Periodic GPU sampling cannot guarantee true peak utilization for
short inference.

Do not call it:

    exact_peak_gpu_utilization

============================================================ 50. NO QUALITY METRICS
============================================================

Do not calculate:

    PSNR
    SSIM
    LPIPS
    VMAF

Do not compare output quality.

Output correctness only means:

    valid output
    correct dimensions
    correct scale

Quality evaluation is Step 5.6.

============================================================ 51. NO REAL-TIME ANALYSIS
============================================================

Do not calculate:

    real-time factor
    deadline misses
    playback buffer
    chunk deadline
    streaming feasibility

Those belong to Step 5.7.

FPS here means:

    model inference throughput

not:

    end-to-end streaming FPS.

Document this distinction.

============================================================ 52. TESTING STRATEGY
============================================================

Add tests covering at minimum:

1. BenchmarkConfig validation.

2. CPU benchmark execution.

3. GPU benchmark execution when CUDA is available.

4. Graceful GPU skip/error when CUDA is unavailable.

5. CUDA synchronization behavior.

6. Warmup exclusion from measured trials.

7. Exact measured trial count.

8. Raw latency preservation.

9. Mean calculation.

10. Median calculation.

11. p95 calculation.

12. Min/max calculation.

13. Standard deviation.

14. FPS calculation.

15. Output validation occurs outside timing boundary.

16. CPU affinity integration.

17. ProcessMonitor integration.

18. GPUMonitor integration.

19. GPU synchronous boundary integration.

20. Model loading occurs once per benchmark case.

21. No silent CPU fallback.

22. Invalid device rejection.

23. Invalid scale rejection.

24. Input resolution metadata.

25. Failure handling.

26. Existing Step 0–4 tests remain passing.

============================================================ 53. CUDA TESTING
============================================================

If no GPU exists:

    CPU tests must still pass.

GPU runtime tests may be skipped explicitly.

Do NOT fake CUDA.

Do NOT globally monkey-patch torch.cuda just to simulate GPU timing.

For synchronization tests without hardware, test the benchmark
control flow using mocks only at the interface boundary.

If a real CUDA GPU exists:

    run a tiny bounded smoke benchmark.

Do not make the entire test suite depend on a specific GPU model.

============================================================ 54. BENCHMARK TEST RUNTIME
============================================================

Do not make tests run 20 full SR trials on every CI/test invocation
unless the repository already uses that strategy.

Use small configurable test counts for unit/integration tests.

The production benchmark default may remain:

    warmup = 3
    measured = 20

Tests may override these values.

============================================================ 55. PERFORMANCE DATA PRECISION
============================================================

Preserve raw timing values at full Python floating-point precision.

Do not round before calculating:

    mean
    median
    p95
    standard deviation

Only round when displaying results.

============================================================ 56. NO HIDDEN SLEEP/WAIT
============================================================

Do not add arbitrary:

    sleep(1)
    sleep(0.5)
    sleep(...)

calls to make monitoring appear to work.

GPU synchronization must be used for GPU completion.

Monitoring must use its configured sampling mechanism.

Timing must measure actual inference.

============================================================ 57. BENCHMARK RESULT SERIALIZATION
============================================================

Step 5.8 will define the final machine-readable benchmark dataset.

Step 5.5 should therefore expose structured Python results that are
serializable, but DO NOT create the final dataset/export pipeline.

It is acceptable to provide:

    result.model_dump()
    or equivalent

if consistent with existing Pydantic/schema conventions.

Do not create the final CSV/JSON dataset generation workflow.

============================================================ 58. DOCUMENTATION
============================================================

Update:

    STEP5_IMPLEMENTATION.md

Add:

    Step 5.5 — Inference Benchmark Harness

Document:

    benchmark case
    device policy
    CPU configuration
    GPU configuration
    input handling
    model initialization
    warmup
    timing methodology
    CUDA synchronization
    trial count
    latency statistics
    p95 methodology
    throughput definition
    monitoring integration
    GPU allocator caveat
    GPU utilization limitation
    failure handling
    reproducibility metadata
    benchmark matrix philosophy
    fairness rules
    scope limitations

Explicitly state:

    Step 5.5 measures inference performance.

Explicitly state:

    Step 5.5 does NOT evaluate SR quality.

Explicitly state:

    Step 5.5 does NOT determine final CPU-vs-GPU deployment policy.

============================================================ 59. REGRESSION
============================================================

Run:

    python -m pytest tests/ -v

Report:

    total
    passed
    skipped
    failed

Also run a small real benchmark smoke test if the environment
supports the required model/device.

Clearly distinguish:

    unit tests
    integration tests
    actual benchmark smoke test

Do not represent a mocked GPU test as real GPU benchmarking.

============================================================ 60. FINAL REPORT
============================================================

Report:

1. Files created.

2. Files modified.

3. Frozen Step 0–4 files left untouched.

4. BenchmarkConfig implementation.

5. BenchmarkResult implementation.

6. Warmup policy.

7. Timing methodology.

8. CUDA synchronization implementation.

9. CPU affinity integration.

10. ProcessMonitor integration.

11. GPUMonitor integration.

12. Synchronous GPU boundary integration.

13. Trial/latency aggregation.

14. p95 methodology.

15. Throughput/FPS definition.

16. Failure handling.

17. Input/model reuse behavior.

18. Test results.

19. Actual hardware benchmark smoke-test status.

20. GPU availability status.

21. Any limitations.

22. Confirmation that 5.6–5.9 were NOT implemented.

============================================================
STOP CONDITION
============================================================

STOP HERE.

DO NOT begin:

    5.6 Quality evaluation
    5.7 FPS / real-time feasibility analysis
    5.8 Machine-readable benchmark dataset
    5.9 Validation + reproducibility report

# Step 5.5 must be reviewed by Claude before proceeding.
