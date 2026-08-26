STEP 1 — VIDEO & CONTENT PROFILING
IMPLEMENTATION

Step 0 and Step 0.1 are COMPLETE and FROZEN.

Step 0:

- Cloud → Edge → Client service boundaries
- Cloud origin
- Edge cache
- Client player/buffer emulator
- telemetry
- RTT measurement
- edge identity

Step 0.1:

- independent RTT measurements
- stall duration
- dual-edge identity validation
- per-edge cache isolation
- target/base representation schema groundwork

The foundation has passed 13/13 tests.

DO NOT modify the Step 0 architecture unless an absolutely necessary
interface compatibility change is required. Preserve all existing
Step 0 tests.

============================================================
OBJECTIVE
============================================================

Implement ONLY:

    STEP 1 — VIDEO & CONTENT PROFILING

The purpose of Step 1 is to create a deterministic, reproducible,
SOURCE-SIDE profiling pipeline that analyzes an input video and
produces:

    1. video-level metadata
    2. deterministic chunk boundaries
    3. chunk-level source-content features
    4. a persisted profiling dataset
    5. a manifest/integrity record

This profiling occurs BEFORE live adaptive streaming decisions.

The output of Step 1 will later be consumed by the chunk,
network, SR, and adaptive scheduling stages.

============================================================
SUPPORTED INPUT
============================================================

The profiler MUST support source videos with:

    30 FPS
    60 FPS
    120 FPS

Do NOT hardcode 30 FPS assumptions.

All frame/time calculations must derive from the actual FPS parsed
from the source video.

For a default 2-second chunk:

    30 FPS  → approximately 60 frames
    60 FPS  → approximately 120 frames
    120 FPS → approximately 240 frames

Use actual parsed FPS and timestamps rather than hardcoded frame
counts.

============================================================
ARCHITECTURAL PRINCIPLE
============================================================

The profiler is an OFFLINE / SOURCE-SIDE preprocessing stage.

It does NOT have a real-time 8.3 ms/frame deadline.

Do NOT optimize away accurate profiling merely to satisfy a
runtime streaming FPS budget.

The 8.3 ms/frame constraint applies to future runtime processing
of 120 FPS content, particularly SR/inference, not to this
offline profiling pass.

Accuracy and reproducibility are more important than profiler
throughput at this stage.

============================================================
REUSE LEGACY COMPONENTS
============================================================

Reuse the legacy components identified by the Step 1 audit where
appropriate:

    src/modules/video_loader.py
        VideoLoader

    src/modules/frame_extractor.py
        FrameExtractor

    src/modules/scene_analyzer.py
        analyze_frame

    src/modules/complexity_estimator.py
        estimate_complexity

Do NOT blindly copy their old behavior.

Modify/adapt them only where required to satisfy the Step 1
contract below.

Do NOT reuse the old runtime DecisionEngine, EnhancementEngine,
DeviceMonitor, or VideoEncoder as part of Step 1.

Do NOT modify the legacy components unless necessary for the new
Step 1 implementation. Prefer creating/adapting Step 1-specific
wrappers/modules where that keeps the Step 0 architecture clean.

============================================================
STEP 1 PIPELINE
============================================================

Implement:

    SOURCE VIDEO
        ↓
    Video Metadata Extraction
        ↓
    Deterministic Chunking
        ↓
    Frame/Temporal Analysis
        ↓
    Spatial/Content Analysis
        ↓
    Chunk Aggregation
        ↓
    Profile Dataset Export
        ↓
    Manifest / Integrity Metadata

============================================================

1. # VIDEO METADATA

Extract and persist at minimum:

    video_id
    source filename
    duration_seconds
    source_fps
    width
    height
    frame_count
    codec
    pixel format, if available
    source bitrate, if available
    audio presence

Do not fabricate values when FFprobe/OpenCV cannot provide them.

Use null/None where appropriate.

============================================================ 2. DETERMINISTIC CHUNKING
============================================================

Implement a deterministic source-side chunker.

Default:

    chunk_duration_seconds = 2.0

Make chunk duration configurable.

Example:

    --chunk-duration 2.0

Do not hardcode the chunk duration internally.

For every chunk, persist:

    chunk_id
    start_time_seconds
    end_time_seconds
    duration_seconds
    start_frame
    end_frame
    frame_count

Frame boundaries MUST be derived from actual source FPS/timestamps.

Do not assume:

    60 frames = one chunk

because that fails for 60/120 FPS content.

The chunking process must be deterministic:

    same source
    same chunk duration
        ↓
    same chunk boundaries

If FFmpeg is used for physical segment generation, ensure the
metadata index and generated files correspond deterministically.

============================================================ 3. MOTION FEATURE
============================================================

The legacy implementation calculates motion from frame-to-frame
pixel differences.

Do NOT use the legacy static:

    MOTION_SCALE_FACTOR = 4.0

as the primary correction for higher FPS.

Do NOT simply multiply motion by:

    FPS / 30

as the primary implementation.

Instead, implement a constant temporal comparison interval.

Conceptually:

    30 FPS:
        compare approximately t with t - 1/30 s

    60 FPS:
        compare approximately t with t - 1/30 s

    120 FPS:
        compare approximately t with t - 1/30 s

Therefore the comparison frame offset should be derived from the
actual FPS.

For example:

    temporal_window_frames ≈ round(source_fps / 30)

with appropriate handling of boundaries.

The purpose is to compare frames separated by approximately the
same amount of real time across 30/60/120 FPS sources.

Document this assumption.

IMPORTANT:

This is a practical normalization strategy, not a claim that
motion is perfectly FPS-invariant.

The implementation should make the temporal comparison window
configurable so it can be empirically evaluated later.

============================================================ 4. SPATIAL / CONTENT FEATURES
============================================================

Carry forward the following source-side features identified in
the audit:

    motion
    texture density
    edge density
    blur / clarity
    spatial complexity

Use the existing scene-analysis logic where appropriate.

Definitions should remain consistent with the legacy implementation
unless modification is necessary for FPS/chunk compatibility.

The features MUST be computed only from raw source frames.

============================================================ 5. CHUNK AGGREGATION
============================================================

Do NOT store only one mean value per feature.

At minimum implement the following aggregation policy:

    Motion:
        mean
        p95
        max

    Texture density:
        mean
        p95

    Edge density:
        mean
        p95

    Blur / clarity:
        mean
        p95

    Spatial complexity:
        mean
        p95
        max

Use a numerically correct percentile implementation.

Document the aggregation policy.

The purpose of retaining percentile/max statistics is to preserve
short high-complexity or high-motion events that could disappear
when only a mean is retained.

Do NOT invent additional features merely to make the dataset larger.

============================================================ 6. COMPLEXITY
============================================================

Reuse the legacy complexity calculation where appropriate.

The existing complexity estimator produces a weighted visual
complexity score.

Adapt it so that frame-level scores can be aggregated into the
chunk-level statistics specified above.

Do NOT make the complexity score depend on:

    network state
    edge state
    SR output
    bitrate decisions
    future playback outcomes

============================================================ 7. SCENE CHANGE
============================================================

If the existing scene-analysis implementation already provides a
valid scene-change signal, audit whether it can be safely included.

Do NOT invent a new scene-change algorithm unless the existing
implementation requires it for the Step 1 contract.

If a reliable scene-change score cannot be implemented without
introducing unnecessary complexity, leave it out and explicitly
document that it is deferred.

Do not fabricate scene-change values.

============================================================ 8. FPS HANDLING
============================================================

Explicitly test the profiler with:

    30 FPS input
    60 FPS input
    120 FPS input

Verify:

    source_fps is detected correctly
    chunk duration remains approximately constant in seconds
    frame counts scale appropriately
    temporal comparison offsets scale appropriately
    no hardcoded 30-FPS frame assumptions remain

For a 2-second chunk, the profiler should approximately observe:

    30 FPS → 60 frames
    60 FPS → 120 frames
    120 FPS → 240 frames

allowing for actual container/timestamp behavior.

============================================================ 9. PROFILE DATASET FORMAT
============================================================

Create a clean persisted profiling dataset.

Use JSON unless the existing project structure provides a strong
reason to use another format.

Separate:

    CONTENT PROFILE

from:
FILE/INTEGRITY MANIFEST

Do NOT put file hashes into the content feature object itself.

Conceptual profile structure:

{
"schema_version": "...",
"video_id": "...",
"source": {
"filename": "...",
"duration_seconds": ...,
"fps": ...,
"width": ...,
"height": ...,
"frame_count": ...,
"codec": "...",
"pixel_format": "...",
"bitrate": ...,
"has_audio": ...
},
"profiling_config": {
"chunk_duration_seconds": 2.0,
"motion_temporal_window_seconds": ...,
"aggregation": {
"motion": ["mean", "p95", "max"],
"texture": ["mean", "p95"],
"edge_density": ["mean", "p95"],
"blur": ["mean", "p95"],
"complexity": ["mean", "p95", "max"]
}
},
"chunks": [
{
"chunk_id": "0000",
"start_time_seconds": ...,
"end_time_seconds": ...,
"duration_seconds": ...,
"start_frame": ...,
"end_frame": ...,
"frame_count": ...,

            "motion": {
                "mean": ...,
                "p95": ...,
                "max": ...
            },

            "texture_density": {
                "mean": ...,
                "p95": ...
            },

            "edge_density": {
                "mean": ...,
                "p95": ...
            },

            "blur": {
                "mean": ...,
                "p95": ...
            },

            "spatial_complexity": {
                "mean": ...,
                "p95": ...,
                "max": ...
            }
        }
    ]

}

This is the conceptual contract. Adjust field naming to match
existing project conventions where appropriate, but preserve the
semantics.

============================================================ 10. MANIFEST / INTEGRITY DATA
============================================================

Create a separate manifest containing reproducibility/integrity
information such as:

    video_id
    source file hash
    generated profile path
    generated chunk paths
    chunk hashes, if physical chunk files are generated
    profiler version/schema version

Hashes belong here, NOT in the content feature structure.

============================================================ 11. DATA LEAKAGE RULE
============================================================

THIS IS A HARD CONSTRAINT.

Step 1 source profiling MUST NOT use:

    PSNR
    SSIM
    LPIPS
    VMAF
    SR output quality
    post-SR frames
    edge processing time
    network throughput
    RTT
    cache state
    future playback outcome
    future stall information
    future scheduler decisions

These may be used in later evaluation/runtime stages.

In particular:

    VMAF is a valid later QoE/output-quality metric,
    but MUST NOT become a source-side profiling feature.

============================================================ 12. OUTPUT LOCATION
============================================================

Keep Step 1 data separate from Step 0 runtime state.

Use a clear structure such as:

    data/
        profiles/
        chunks/
        manifests/

or an equivalent clean project structure.

Do not mix profiling artifacts with the Edge cache.

Do not place profiling metadata inside runtime telemetry.

============================================================ 13. CLI
============================================================

Provide a reproducible CLI entry point.

Example:

    python -m adaptive_sr.profiling.profile_video \
        --input <video> \
        --output <output-directory> \
        --chunk-duration 2.0

If a different module path is more consistent with the current
project structure, use that instead.

The CLI must expose at least:

    input video
    output directory
    chunk duration

If useful, also expose:

    motion temporal window

Do not expose unnecessary tuning parameters yet.

============================================================ 14. TESTING
============================================================

Add automated tests for:

1. Video metadata extraction.

2. Deterministic chunk boundaries.

3. 30 FPS chunk frame count/timing.

4. 60 FPS chunk frame count/timing.

5. 120 FPS chunk frame count/timing.

6. Motion temporal comparison offset:
   30 FPS → approximately 1 frame
   60 FPS → approximately 2 frames
   120 FPS → approximately 4 frames

7. Chunk aggregation:
   mean
   p95
   max

8. Profile JSON schema/required fields.

9. Repeatability:
   same input + same configuration
   → same profile metadata/chunk boundaries.

10. Data-leakage protection:
    profile generation must not depend on
    PSNR/SSIM/LPIPS/VMAF/network/runtime telemetry.

11. Existing Step 0/0.1 tests must continue to pass.

Use synthetic test videos where practical.

For FPS tests, create or use deterministic test footage at:

    30 FPS
    60 FPS
    120 FPS

Do not rely solely on one arbitrary real-world video.

============================================================ 15. EMPIRICAL MOTION VALIDATION
============================================================

Do NOT claim that the constant temporal-window normalization
makes motion perfectly comparable across FPS.

Add a small validation report/test fixture demonstrating the
behavior across 30/60/120 FPS synthetic footage.

The objective is to confirm that the implementation behaves
sensibly, not to prove a universal motion-invariance theorem.

If results reveal unexpected behavior, report it instead of
silently compensating with another arbitrary multiplier.

============================================================ 16. DOCUMENTATION
============================================================

Create:

    STEP1_IMPLEMENTATION.md

Document:

    purpose
    architecture
    reused legacy components
    modified components
    chunking strategy
    FPS handling
    motion temporal-window strategy
    feature definitions
    aggregation rules
    dataset schema
    manifest structure
    data-leakage rules
    CLI usage
    tests
    known limitations
    future dependencies

Explicitly state:

    Step 1 is offline/source-side profiling.

Explicitly state:

    Step 1 does not perform SR, ABR, adaptive FPS,
    resource allocation, ML, online learning, or scheduling.

============================================================ 17. STRICT NON-GOALS
============================================================

DO NOT implement:

    Super-resolution
    FSRCNN
    Real-ESRGAN
    BasicVSR++
    SR model benchmarking
    ABR
    adaptive bitrate
    adaptive FPS
    CPU allocation
    GPU allocation
    VRAM allocation
    online learning
    ML decision engine
    multi-edge scheduling
    multi-cluster scheduling
    network emulation
    Azure deployment
    concurrent streaming prefetch
    QoE optimization
    VMAF-based decisions

Those belong to later steps.

============================================================ 18. LEGACY CODE SAFETY
============================================================

The Step 0/0.1 service architecture is frozen.

Do not rewrite it.

Do not move Cloud/Edge/Client services.

Do not alter the Edge cache implementation.

Do not alter telemetry semantics unless Step 1 genuinely requires
a new schema field, and if it does, document the reason.

The legacy `src/` implementation should remain usable for reference
and should not be destructively rewritten.

============================================================ 19. FINAL REPORT
============================================================

When implementation is complete, report:

1. Files created.
2. Files modified.
3. Legacy components reused.
4. Legacy components left untouched.
5. CLI command.
6. Example output profile.
7. Example manifest.
8. 30 FPS test result.
9. 60 FPS test result.
10. 120 FPS test result.
11. Motion temporal-window validation result.
12. Aggregation test result.
13. Full test-suite result.
14. Confirmation that Step 0/0.1 tests still pass.
15. Known limitations.

STOP after Step 1.

Do NOT automatically begin Step 2.
