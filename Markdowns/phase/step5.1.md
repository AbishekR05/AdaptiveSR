============================================================
ADAPTIVESR — STEP 5.1 IMPLEMENTATION
BENCHMARK DATASET / TEST-VIDEO PREPARATION
============================================================

IMPORTANT:

Steps 0–4 are COMPLETE and FROZEN.

Step 5 contains:

    5.1 Benchmark dataset / test-video preparation
    5.2 SR model adapter interface
    5.3 CPU affinity + ProcessMonitor integration
    5.4 GPU measurement
    5.5 Inference benchmark harness
    5.6 Quality evaluation
    5.7 FPS / real-time feasibility analysis
    5.8 Machine-readable benchmark dataset
    5.9 Validation + reproducibility report

IMPLEMENT ONLY STEP 5.1.

DO NOT IMPLEMENT 5.2 OR ANY LATER STEP.

============================================================

1. # OBJECTIVE

Prepare a deterministic, reproducible benchmark corpus for the
future AdaptiveSR SR-model benchmarking pipeline.

Step 5.1 must establish the INPUT DATASET and its metadata.

The purpose is to ensure that when Steps 5.2–5.9 benchmark SR
models, every model is evaluated on the same:

    videos
    source FPS
    temporal content
    chunks
    frame counts
    input resolutions
    benchmark cases

This step must NOT perform SR inference.

The output of Step 5.1 will later be consumed by the SR benchmark
harness.

============================================================ 2. ARCHITECTURAL CONTEXT
============================================================

AdaptiveSR is a cloud/edge/client video streaming system.

The frozen architecture is:

    CLIENT
       │
       │ client_edge
       ▼
     EDGE
       │
       │ edge_cloud
       ▼
     CLOUD / ORIGIN

Step 1 already established deterministic source-side profiling.

Step 2 established representations and authoritative logical
chunk-to-representation mapping.

Step 3 established the two independent network paths and controlled
network emulation.

Step 4 established Edge resource monitoring.

Step 5 is now concerned ONLY with empirical SR benchmarking.

The benchmark dataset must therefore be independent of:

    network conditions
    Edge state
    scheduler decisions
    ML predictions
    SR output
    future playback results

The dataset is an INPUT to benchmarking, not an output of the
adaptive system.

============================================================ 3. FIRST — INSPECT THE FROZEN IMPLEMENTATION
============================================================

Before changing anything:

Inspect the current repository and identify the actual implemented
outputs from:

    Step 1 — video/content profiling
    Step 2.1 — representation schema
    Step 2.2 — chunk-to-representation mapping
    Step 3 — network measurement/emulation
    Step 4 — Edge resource monitoring

Do NOT assume paths, filenames, or schema names from this prompt.

Reuse the existing Step 1/2 artifacts where they already provide
the required information.

Do NOT duplicate:

    video metadata
    logical chunk definitions
    representation definitions

if they already exist in shared schemas/artifacts.

The authoritative temporal timeline remains the Step 1 timeline.

Step 5.1 must consume that timeline rather than creating a second,
conflicting chunk timeline.

============================================================ 4. LEGACY CODE SAFETY
============================================================

Do NOT modify the frozen Step 0–4 service architecture.

Do NOT modify:

    Cloud service
    Edge service
    Client service
    Edge cache
    network telemetry semantics
    resource telemetry semantics

Do NOT rewrite the legacy src/ implementation.

The legacy implementation may be inspected for reusable benchmark
assets/utilities, but it must remain available for reference.

If an existing utility can be reused safely, prefer a wrapper or
Step-5-specific integration rather than modifying frozen code.

============================================================ 5. BENCHMARK CORPUS DESIGN
============================================================

The benchmark corpus must explicitly cover:

    30 FPS
    60 FPS
    120 FPS

These are mandatory source-FPS cases.

Do NOT convert the source videos to a common FPS.

Do NOT silently downsample:

    60 → 30
    120 → 30

during dataset preparation.

FPS adaptation belongs to Step 7.

For Step 5 benchmarking, the original source FPS must remain
available.

============================================================ 6. SYNTHETIC TEST VIDEOS
============================================================

Create a small deterministic synthetic test-video corpus.

The synthetic videos must be reproducible and should cover:

    30 FPS
    60 FPS
    120 FPS

Use the same logical visual content across the FPS variants as far
as technically practical.

The purpose is to isolate FPS-related benchmarking behavior.

The synthetic footage should contain enough visual variation to
exercise SR processing rather than being a completely static frame.

Prefer a deterministic generated pattern/sequence containing
multiple spatial frequencies and motion.

Do NOT use random uncontrolled video generation.

If randomness is used internally, use a fixed seed.

The generated videos should be short enough that tests remain fast.

Suggested starting duration:

    approximately 4–6 seconds

but choose the smallest duration that still gives meaningful
multiple-chunk benchmark input.

Do not make the dataset unnecessarily large.

============================================================ 7. REAL-WORLD TEST VIDEO
============================================================

If an existing real-world benchmark/test video is already present
in the repository or existing project data, inspect it and reuse it
where appropriate.

Do NOT download arbitrary internet videos merely to increase the
dataset size.

If there is no suitable existing real-world video, do NOT block
Step 5.1 waiting for one.

Instead:

    document that the initial benchmark corpus is synthetic and
    deterministic, and leave the dataset structure ready for a
    later real-world clip.

The benchmark architecture must support real-world videos without
requiring a code redesign.

============================================================ 8. CONTENT DIVERSITY
============================================================

The benchmark corpus should not consist exclusively of one trivial
visual pattern.

Where practical, the deterministic synthetic corpus should exercise
different spatial/temporal characteristics, such as:

    low-motion content
    moderate-motion content
    higher-motion content

Do not invent complicated semantic categories or claim that these
constitute a statistically representative video dataset.

These are controlled benchmark cases only.

The purpose is to make model performance measurable under different
visual workloads.

============================================================ 9. SOURCE METADATA
============================================================

For every benchmark video, record at minimum:

    benchmark_video_id
    filename
    source_fps
    width
    height
    duration_seconds
    frame_count
    codec
    pixel_format if available
    source_bitrate if available
    audio_presence

Do not fabricate unavailable metadata.

Use:

    null

when metadata cannot be reliably determined.

Metadata should come from actual video inspection rather than
hardcoded assumptions.

============================================================ 10. FPS VALIDATION
============================================================

The corpus must verify that the generated/selected videos actually
have:

    30 FPS
    60 FPS
    120 FPS

Do not simply label a file as 120 FPS because it was intended to be
120 FPS.

Inspect the resulting file/container metadata.

The validation should detect:

    incorrect FPS
    unexpected frame count
    unexpected duration
    corrupted/incomplete video

and fail clearly.

For a known duration D:

    expected_frame_count ≈ D × FPS

allowing for legitimate container/timestamp behavior.

Do not require mathematically exact equality if the container's
timing model makes that inappropriate.

============================================================ 11. CHUNK ASSOCIATION
============================================================

Step 1's logical chunk timeline is authoritative.

Step 5.1 must associate benchmark videos with their logical chunks
where Step 1 profiling artifacts are available.

Do NOT independently invent another chunking algorithm.

For the default Step 1 configuration, chunks are approximately:

    2 seconds

but do NOT hardcode 2 seconds into Step 5.1.

Read the actual chunk metadata/configuration.

For each benchmark case, preserve:

    chunk_id
    start_time_seconds
    end_time_seconds
    duration_seconds
    start_frame
    end_frame
    frame_count

where those fields are available from the frozen Step 1 output.

============================================================ 12. REPRESENTATION AWARENESS
============================================================

Step 2 established that a logical chunk can exist across multiple
representations.

Step 5.1 must not create a new representation schema.

Where representation files already exist, the benchmark dataset may
reference them.

For example:

    360p / chunk_000
    480p / chunk_000
    720p / chunk_000

refer to the same logical temporal interval.

However:

DO NOT implement representation generation here.

DO NOT implement FFmpeg encoding ladders here.

DO NOT implement ABR here.

If representation files are unavailable, the benchmark corpus
manifest should record the logical/source input and leave
representation preparation to the appropriate later workflow.

============================================================ 13. BENCHMARK CASE IDENTITY
============================================================

Create deterministic identifiers for benchmark cases.

A benchmark case should be uniquely attributable to:

    video
    source FPS
    content case
    chunk

Do not include SR model identifiers yet.

SR model identity belongs to Step 5.2+.

Conceptually:

    synthetic_lowmotion_30fps
    synthetic_lowmotion_60fps
    synthetic_lowmotion_120fps

and corresponding chunk identifiers.

The exact naming convention may follow existing project conventions.

============================================================ 14. DATASET DIRECTORY STRUCTURE
============================================================

Create a clean Step-5 benchmark-data area.

Do not mix benchmark inputs with:

    Edge cache
    runtime telemetry
    Step 1 profile output
    network-emulation state

A reasonable structure is:

    data/
        benchmarks/
            sr/
                videos/
                metadata/
                manifests/
                chunks/

Adjust this if the existing repository already has a better
convention.

The important requirement is clear separation from runtime state.

============================================================ 15. MANIFEST
============================================================

Create a deterministic benchmark-corpus manifest.

The manifest should identify:

    schema_version
    dataset_id
    creation/configuration information
    benchmark videos
    source metadata
    FPS
    chunk references
    content-case labels
    file paths
    file hashes

Hashes are for dataset integrity/reproducibility.

Do NOT put SR quality metrics into this manifest.

Do NOT put:

    PSNR
    SSIM
    LPIPS
    VMAF

into Step 5.1.

Those belong to Step 5.6.

Do NOT put:

    inference FPS
    latency
    CPU usage
    GPU usage
    VRAM

into Step 5.1.

Those belong to later benchmark stages.

============================================================ 16. DATASET HASHING
============================================================

For each generated/selected benchmark video, calculate a stable
cryptographic hash such as SHA-256.

The manifest should allow us to verify that the benchmark input has
not changed.

Do not use timestamps or filesystem modification times as the
dataset identity.

The dataset must remain reproducible if copied to another machine.

============================================================ 17. REPRODUCIBILITY
============================================================

The synthetic corpus generation must be deterministic.

Document:

    generation parameters
    FPS
    resolution
    duration
    random seed if applicable
    codec/container
    generation tool/version if available

Running the same preparation process with the same configuration
must produce equivalent benchmark inputs.

If exact binary reproducibility cannot be guaranteed because of
codec/container metadata, document that limitation and use the
content/configuration metadata plus hashes appropriately.

Do NOT falsely claim bit-for-bit reproducibility if the encoder
introduces nondeterminism.

============================================================ 18. RESOLUTION

The benchmark dataset must support the spatial workloads needed by
the later SR benchmark.

Do not hardcode the dataset to only one resolution if the existing
Step 2 representation configuration already provides multiple
resolutions.

At minimum, ensure the dataset can represent:

    source resolution
    input representation resolution
    source FPS

without losing the relationship between them.

Do NOT implement SR scale selection yet.

Do NOT introduce target/base representation decisions.

============================================================ 19. TEST DATA SHOULD BE SMALL

This is a benchmark preparation stage, not the final dataset.

Keep generated test videos small enough for:

    unit tests
    CI
    local development
    repeated benchmarking

Do not generate gigabytes of video.

The corpus should be sufficient to validate:

    FPS handling
    chunk association
    deterministic metadata
    benchmark case identity
    file integrity

============================================================ 20. CLI

Provide a reproducible CLI for preparing the benchmark corpus.

For example:

    python -m adaptive_sr.benchmarking.prepare_dataset \
        --output <directory>

The exact module path must follow the actual project structure.

The CLI should support at least:

    output directory

and, if useful:

    duration
    resolution
    seed
    overwrite/force

Do not expose unnecessary SR-specific parameters.

The CLI should print a concise summary such as:

    Dataset ID
    Number of videos
    FPS coverage
    Total duration
    Total frames
    Manifest path

============================================================ 21. VALIDATION COMMAND

Provide a way to validate an existing benchmark corpus.

For example:

    python -m adaptive_sr.benchmarking.prepare_dataset \
        --validate <manifest>

or a separate validator if that is cleaner.

Validation must check:

    files exist
    hashes match
    metadata is readable
    FPS matches manifest
    frame count is plausible
    chunk references are valid
    benchmark IDs are unique
    no duplicate dataset entries exist

Do not perform SR inference during validation.

============================================================ 22. TESTS

Add automated tests for:

1. Dataset preparation succeeds.

2. 30 FPS synthetic video is generated/validated.

3. 60 FPS synthetic video is generated/validated.

4. 120 FPS synthetic video is generated/validated.

5. FPS is detected from actual video metadata.

6. Frame counts are plausible for the declared duration/FPS.

7. Benchmark video IDs are unique.

8. Dataset manifest is valid.

9. SHA-256 hashes are generated.

10. Hash validation detects a modified/corrupted file.

11. Re-running preparation with the same configuration produces
    equivalent dataset metadata.

12. Logical chunk references are consistent with Step 1 artifacts
    when those artifacts are available.

13. Chunk IDs are not duplicated.

14. 30/60/120 FPS cases do not accidentally share incorrect frame
    counts.

15. Missing/corrupt video files are detected.

16. The validator rejects a manifest whose file hash does not match.

Tests must remain fast.

Do not make the test suite depend on external internet downloads.

============================================================ 23. REGRESSION

Run:

    python -m pytest tests/ -v

All existing Step 0–4 tests must continue to pass.

Do NOT modify frozen tests merely to make Step 5.1 pass.

If an interface incompatibility is genuinely discovered, stop and
report it rather than silently changing a frozen contract.

============================================================ 24. DOCUMENTATION

Create:

    STEP5_IMPLEMENTATION.md

but ONLY document Step 5.1 in this step.

Clearly label:

    Step 5.1 — Benchmark Dataset / Test-Video Preparation

Document:

    purpose
    benchmark corpus design
    30/60/120 FPS coverage
    synthetic video generation
    real-world video policy
    metadata
    chunk association
    manifest
    hashing
    reproducibility
    directory structure
    CLI
    validation
    tests
    limitations

Explicitly state:

    Step 5.1 prepares benchmark INPUTS.

Explicitly state:

    Step 5.1 does NOT benchmark SR models.

============================================================ 25. STRICT NON-GOALS
============================================================

DO NOT implement:

- SR model adapters
- FSRCNN inference
- FSRCNN INT8 inference
- Real-ESRGAN inference
- BasicVSR++ inference
- SR model loading
- inference benchmarking
- CPU affinity
- ProcessMonitor integration
- GPU monitoring
- VRAM monitoring
- PSNR
- SSIM
- LPIPS
- VMAF
- inference FPS measurement
- latency measurement
- real-time feasibility
- SR quality scoring
- SR model selection
- scheduler logic
- ABR
- adaptive FPS
- ML
- online learning
- resource allocation
- Azure deployment
- remote SR inference

Those belong to later Step 5 substeps.

============================================================ 26. IMPORTANT: DO NOT PRE-IMPLEMENT 5.2
============================================================

Do not create the SR model adapter abstraction yet.

Do not create:

    SRModelAdapter
    ModelRunner
    inference interface
    model registry redesign

unless an existing frozen/shared structure is strictly required for
dataset preparation.

Step 5.1 must remain usable even if no SR model is installed.

============================================================ 27. FINAL REPORT

When Step 5.1 is complete, report ONLY:

1. Files created.
2. Files modified.
3. Existing Step 0–4 files left untouched.
4. Benchmark dataset directory structure.
5. Synthetic videos generated.
6. FPS coverage:
   30 FPS
   60 FPS
   120 FPS
7. Metadata captured.
8. Manifest structure.
9. Hashing/integrity mechanism.
10. Chunk association mechanism.
11. CLI command.
12. Validation command.
13. Tests added.
14. Full pytest result.
15. Any issues or limitations.
16. Confirmation that no SR inference was implemented.
17. Confirmation that 5.2–5.9 were NOT implemented.

============================================================
STOP CONDITION
============================================================

STOP HERE.

Do NOT automatically begin:

    5.2 SR model adapter interface

Do NOT modify the implementation after Step 5.1 succeeds.

# The next step will be reviewed separately before implementation.
