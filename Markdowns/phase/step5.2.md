============================================================
ADAPTIVESR — STEP 5.2 IMPLEMENTATION
SR MODEL RUNNER ADAPTER INTERFACE
============================================================

IMPORTANT:

Steps 0–4 are COMPLETE and FROZEN.

Step 5 contains:

    5.1 Benchmark dataset / test-video preparation     [FROZEN]
    5.2 SR model runner adapter interface              [THIS STEP]
    5.3 CPU affinity + ProcessMonitor integration
    5.4 GPU measurement
    5.5 Inference benchmark harness
    5.6 Quality evaluation
    5.7 FPS / real-time feasibility analysis
    5.8 Machine-readable benchmark dataset
    5.9 Validation + reproducibility report

IMPLEMENT ONLY STEP 5.2.

DO NOT IMPLEMENT 5.3–5.9.

============================================================

1. # OBJECTIVE

Create a clean, model-independent adapter interface that allows
AdaptiveSR to execute different SR models through a common contract.

The purpose of Step 5.2 is NOT to benchmark models yet.

The purpose is to eliminate model-specific logic from the future
benchmark harness.

The future flow must be conceptually:

    benchmark input
          ↓
    SR adapter
          ↓
    model-specific backend
          ↓
    SR output

The benchmark harness in Step 5.5 must eventually be able to use:

    FSRCNN FP32
    FSRCNN INT8
    Real-ESRGAN
    BasicVSR++

without knowing their internal implementation details.

============================================================ 2. FIRST — INSPECT THE CURRENT REPOSITORY
============================================================

Before modifying anything:

Inspect the actual current repository.

Specifically locate:

    - Step 5.1 benchmark dataset implementation
    - Step 5.1 manifest/schema
    - existing SR model code
    - existing model registry
    - existing FSRCNN implementation
    - existing INT8/ONNX implementation
    - existing Real-ESRGAN implementation
    - existing BasicVSR++ implementation/stub
    - existing SR-related tests
    - legacy benchmark code

Do NOT assume filenames or module paths.

Reuse existing working model implementations where possible.

Do NOT rewrite the model implementations simply to fit the adapter.

============================================================ 3. FROZEN STEP 5.1 CONTRACT
============================================================

Step 5.1 is FROZEN.

Its benchmark corpus provides deterministic inputs.

The synthetic corpus is Layer A:

    latency
    throughput
    resource
    pipeline correctness
    FPS/chunk handling

It is NOT the quality corpus.

Layer B real-world reference data belongs to Step 5.6.

Layer C production-like streaming inputs belong to later
end-to-end validation.

Do not modify Step 5.1.

Do not modify its manifest schema.

Do not change:

    FPS cases
    chunk definitions
    file hashes
    metadata
    codec policy

============================================================ 4. REQUIRED ADAPTER CONCEPT
============================================================

Define one common SR adapter contract.

The interface must abstract:

    model initialization
    model metadata
    input validation
    SR execution
    output validation
    cleanup/release if needed

The benchmark harness must NOT need to know whether the backend is:

    PyTorch
    ONNX Runtime
    OpenCV
    BasicVSR++
    Real-ESRGAN
    another future backend

The adapter is responsible for hiding backend-specific details.

============================================================ 5. MODEL-AGNOSTIC INTERFACE
============================================================

Create a model-independent interface/protocol/abstract base class.

Use the project's existing Python style and architecture.

Do NOT introduce unnecessary framework dependencies.

The interface should expose concepts equivalent to:

    model_id
    backend
    scale_factor
    supports_input(...)
    initialize(...)
    process(...)
    get_metadata(...)
    close(...)

The exact method names may follow existing project conventions.

Do NOT copy this list blindly if the repository already has a better
established naming convention.

The important requirement is a stable common contract.

============================================================ 6. INPUT CONTRACT
============================================================

The adapter must accept a clearly defined SR input.

The input must contain enough information to identify:

    frames
    frame format
    width
    height
    FPS
    temporal ordering

For spatial SR models, a frame sequence of length N may be supplied.

For temporal SR models such as BasicVSR++, the adapter must be capable
of receiving a sequence rather than assuming that every model is
single-frame.

Do NOT implement temporal inference logic in this step.

The interface must simply avoid making single-frame processing a
mandatory assumption.

============================================================ 7. OUTPUT CONTRACT
============================================================

The adapter must return a clearly defined SR output.

At minimum, the output must preserve:

    output frames
    output width
    output height
    output FPS
    frame ordering

The adapter must expose the scale factor that explains the spatial
relationship:

    output_width  = input_width  × scale
    output_height = input_height × scale

where the model guarantees integer scaling.

Do not silently resize outputs to satisfy the contract.

If the model produces an unexpected output shape, raise a clear
validation error.

============================================================ 8. MODEL METADATA
============================================================

Each adapter must expose static model metadata.

At minimum:

    model_id
    display_name
    backend
    scale_factor(s)
    model_type
    temporal_or_spatial
    supported_input_formats
    supported_output_formats

Where known, also expose:

    precision
    model_file/path
    framework
    version

Do NOT record runtime measurements here.

Do NOT add:

    latency
    FPS
    CPU utilization
    GPU utilization
    VRAM
    PSNR
    SSIM
    LPIPS

Those belong to later steps.

============================================================ 9. MODEL TYPES
============================================================

The interface must support two conceptual classes:

    SPATIAL
        independent frame processing

    TEMPORAL
        sequence-aware video SR

This distinction matters because:

    FSRCNN
    Real-ESRGAN

are spatial/frame-oriented models,

while:

    BasicVSR++

is a temporal video SR model.

Do NOT force BasicVSR++ into a fake single-frame interface.

============================================================ 10. INITIAL MODEL ADAPTERS
============================================================

Implement adapters for models that are actually usable in the
current repository.

Expected candidates:

    1. FSRCNN FP32
    2. FSRCNN INT8 / ONNX Runtime
    3. Real-ESRGAN
    4. BasicVSR++

However:

DO NOT fabricate support for a model whose backend is unavailable.

If BasicVSR++ is currently environment-blocked, implement its adapter
contract/stub only if the repository architecture requires it, and
mark it explicitly as:

    unavailable
    not runnable in current environment
    reason documented

Do NOT silently substitute another model.

============================================================ 11. FSRCNN FP32 ADAPTER
============================================================

Wrap the existing FSRCNN FP32 implementation.

The adapter must:

    initialize the existing model
    validate input
    execute inference
    return normalized output
    expose metadata

Do not rewrite FSRCNN itself unless absolutely necessary.

Do not introduce benchmark timing here.

Do not measure FPS here.

Do not collect CPU/GPU metrics here.

============================================================ 12. FSRCNN INT8 ADAPTER
============================================================

Wrap the existing INT8/ONNX Runtime implementation if available.

The adapter must hide ONNX Runtime details from the future benchmark
harness.

The benchmark harness should eventually call the same method regardless
of whether the backend is:

    PyTorch
    ONNX Runtime

Do not add optimization work here.

Do not tune thread counts here.

Do not add CPU affinity here.

Those belong to Step 5.3.

============================================================ 13. REAL-ESRGAN ADAPTER
============================================================

Wrap the existing Real-ESRGAN implementation.

The adapter must expose the same common interface.

Do not redesign the Real-ESRGAN architecture.

Do not add GPU monitoring.

Do not benchmark it.

Do not alter model weights.

Do not change inference settings unless required to satisfy the
adapter contract.

============================================================ 14. BASICVSR++ ADAPTER
============================================================

BasicVSR++ is a temporal video SR model.

The adapter must therefore support sequence input.

If its current backend is unavailable on the development environment:

    - do not fake inference
    - do not create dummy output
    - do not report it as operational
    - do not install a large unrelated dependency stack merely to
      force it to run

Instead provide a clear availability state.

Example conceptual metadata:

    model_id: basicvsrpp
    status: unavailable
    reason: backend/environment unavailable

If an existing implementation already provides a clean way to detect
availability, use it.

============================================================ 15. AVAILABILITY / CAPABILITY DISCOVERY
============================================================

The system should be able to discover which adapters are currently
runnable.

For example:

    list_available_models()

could return:

    tinysr
    tinysr_int8
    real_esrgan

while marking:

    basicvsrpp

as unavailable.

Do not make unavailable models disappear silently.

The future benchmark report must be able to distinguish:

    available and benchmarkable
    registered but unavailable
    unsupported

============================================================ 16. SCALE FACTOR
============================================================

The adapter must explicitly expose the supported scale factor.

Examples:

    ×2
    ×3
    ×4

If a model supports multiple scale factors, expose the set/list.

Do NOT assume ×2 universally.

Do NOT implement automatic scale selection.

Do NOT implement adaptive scaling.

============================================================ 17. DEVICE SELECTION
============================================================

The adapter may accept a device configuration such as:

    cpu
    cuda

but it must NOT perform resource measurement.

Device selection belongs to execution configuration.

Do not hardcode:

    CUDA always
    CPU always

The future benchmark harness must be able to select the execution
device.

If a requested device is unavailable, fail clearly.

Do not silently fall back from CUDA to CPU unless the existing project
contract explicitly requires such behavior.

A silent fallback would invalidate future CPU-vs-GPU measurements.

============================================================ 18. PRECISION
============================================================

Where applicable, expose precision as metadata:

    fp32
    fp16
    int8

Do not perform automatic precision conversion in the adapter unless
the underlying model implementation already requires it.

Do not introduce quantization in Step 5.2.

============================================================ 19. PREPROCESSING / POSTPROCESSING
============================================================

Model-specific preprocessing and postprocessing belongs INSIDE the
adapter.

For example:

    color conversion
    tensor conversion
    normalization
    layout conversion
    output tensor conversion

The future benchmark harness should operate on the common adapter
contract rather than know these details.

However:

Do not include benchmark timing inside preprocessing/postprocessing.

Timing is Step 5.5.

============================================================ 20. NO BENCHMARK LOGIC
============================================================

This is critical.

DO NOT implement:

    timers
    FPS calculations
    latency calculations
    p50/p95
    throughput measurement
    CPU monitoring
    GPU monitoring
    VRAM measurement
    ProcessMonitor integration
    CPU affinity

Those belong to:

    5.3
    5.4
    5.5

The adapter simply executes inference.

============================================================ 21. NO QUALITY METRICS
============================================================

DO NOT implement:

    PSNR
    SSIM
    LPIPS
    VMAF

Those belong to Step 5.6.

============================================================ 22. NO REAL-TIME LOGIC
============================================================

DO NOT calculate:

    chunk deadline
    processing ratio
    real-time feasibility
    buffer pressure
    adaptation decision

Those belong to Step 5.7 and later.

============================================================ 23. ERROR HANDLING
============================================================

Provide explicit errors for:

    unsupported input format
    invalid frame shape
    invalid frame count
    unsupported scale factor
    unavailable device
    missing model weights
    unavailable backend
    invalid output shape
    inference failure

Errors must identify:

    model_id
    operation
    reason

Do not swallow exceptions.

Do not return fake frames after inference failure.

============================================================ 24. REGISTRY INTEGRATION
============================================================

If an existing model registry exists:

    integrate the new adapter layer with it.

Do not create a second competing registry.

The registry should allow the system to resolve:

    model_id
        ↓
    adapter implementation

Do not put runtime benchmark results in the registry.

The registry is configuration/capability information only.

============================================================ 25. TESTABILITY
============================================================

The adapter interface must be testable without requiring every
real SR dependency to be installed.

Create a minimal fake/mock adapter for interface tests if appropriate.

Tests should verify:

    adapter contract
    metadata contract
    input validation
    output validation
    scale validation
    device validation
    availability discovery
    registry lookup
    error handling

For actual installed models, add smoke tests only when their
dependencies are available.

Do not make the entire test suite fail simply because BasicVSR++ is
not available in the current environment.

Instead:

    skip with explicit reason

or use the project's established optional-dependency mechanism.

============================================================ 26. STEP 5.1 INTEGRATION TEST
============================================================

Use an existing Step 5.1 benchmark input to verify that an available
adapter can accept the prepared input.

This is ONLY an adapter smoke test.

It should prove:

    dataset input
        ↓
    adapter
        ↓
    valid SR output

Do not record benchmark measurements.

Do not calculate quality metrics.

Do not generate the Step 5.8 benchmark dataset.

============================================================ 27. FILE ORGANIZATION
============================================================

Follow the existing repository structure.

A reasonable conceptual structure is:

    adaptive_sr/
        benchmarking/
            adapters/
                base.py
                fsrcnn.py
                real_esrgan.py
                basicvsrpp.py

But DO NOT blindly create this exact structure if the repository
already has a better organization.

Keep adapter code isolated from:

    Edge service
    Client service
    network emulator
    resource monitor

This is a benchmark/model execution layer.

============================================================ 28. DOCUMENTATION
============================================================

Create/update:

    STEP5_IMPLEMENTATION.md

Add:

    Step 5.2 — SR Model Runner Adapter Interface

Document:

    purpose
    adapter contract
    input contract
    output contract
    model metadata
    spatial vs temporal distinction
    supported adapters
    unavailable adapters
    device selection
    precision
    registry integration
    error handling
    tests
    limitations

Explicitly state:

    Step 5.2 provides model execution abstraction.

Explicitly state:

    Step 5.2 does NOT benchmark models.

============================================================ 29. BACKWARD COMPATIBILITY
============================================================

Do not break existing Step 0–4 functionality.

Do not change:

    Step 1 profiler
    Step 2 representation/chunk contracts
    Step 3 network emulation
    Step 4 ProcessMonitor/resource telemetry

Do not alter frozen tests merely to accommodate Step 5.2.

============================================================ 30. TEST EXECUTION
============================================================

Run:

    python -m pytest tests/ -v

All existing tests must remain passing.

Also run the new Step 5.2 adapter tests.

Report:

    total tests
    passed
    skipped
    failed

If an optional dependency is unavailable, clearly report it.

============================================================ 31. FINAL REPORT
============================================================

When finished, report ONLY:

1. Files created.
2. Files modified.
3. Existing frozen Step 0–4 files left untouched.
4. Adapter interface created.
5. Adapter implementations created.
6. Models currently runnable.
7. Models registered but unavailable.
8. Device support.
9. Scale-factor support.
10. Precision support.
11. Registry integration.
12. Tests added.
13. Full test results.
14. Optional dependencies unavailable.
15. Confirmation that no benchmarking was implemented.
16. Confirmation that 5.3–5.9 were NOT implemented.

============================================================
STOP CONDITION
============================================================

STOP HERE.

Do NOT begin:

    5.3 CPU affinity + ProcessMonitor integration
    5.4 GPU measurement
    5.5 inference benchmark harness
    5.6 quality evaluation
    5.7 FPS/real-time feasibility
    5.8 machine-readable benchmark dataset
    5.9 validation/reproducibility report

# Step 5.2 must be reviewed before proceeding.
