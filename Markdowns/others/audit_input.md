# AdaptiveSR — Previous Implementation Audit

We are restarting the implementation of our AdaptiveSR project from an architectural standpoint.

IMPORTANT:
Do NOT modify, delete, refactor, rename, or rewrite any code yet.
Do NOT start implementing the new architecture yet.

Your current task is ONLY to audit the existing repository and determine what can be reused, what needs modification, and what should be discarded.

---

## PROJECT CONTEXT

The final project is a real-time Super-Resolution-assisted adaptive video streaming system.

The previous implementation was primarily LOCAL.

The new implementation must instead follow a distributed streaming architecture inspired by the Rosevin paper:

    Cloud / Video Origin
            ↓
       Edge Cluster
            ↓
       Edge Server(s)
            ↓
          Client

The client should NOT upload the raw source video to the cloud for every chunk.

The cloud/origin stores encoded video representations/chunks.

The edge cluster:

- receives/obtains chunks from the cloud,
- maintains a cache,
- performs SR when selected,
- allocates compute resources,
- serves processed chunks to clients.

The client:

- requests/receives chunks,
- maintains a playback buffer,
- plays the resulting video,
- measures network conditions,
- reports telemetry.

The project must eventually support:

- 30 FPS input
- 60 FPS input
- 120 FPS input

Initial FPS adaptation will primarily consider:

- 30 → 30
- 60 → 60 or 30
- 120 → 60 or 30

For 60→30 and 120→30, deterministic temporal downsampling can initially be used.

Do NOT introduce motion-aware frame selection unless you find that the existing code already implements it.

---

# BASE-PAPER ARCHITECTURE

The implementation is inspired by:

"Joint Bitrate and Resource Adaptation for Super-Resolution Video Streaming in Multi-Cluster Edge Networks: A New Online Learning Approach"

However, we have clarified an important point:

The Rosevin implementation/system model we are using as our paper-faithful baseline is:

    Cloud
       ↓
    ONE edge cluster
       ↓
    Multiple edge servers
       ↓
    Geographically distributed users

The multi-cluster version is a FUTURE EXTENSION for our project, not something we need to implement immediately.

Therefore:

- single edge cluster = initial paper-faithful implementation
- multiple servers inside that cluster = required architectural concept
- cluster_id should still exist in our data model
- genuine multi-cluster deployment = later extension/stretch goal

Do not silently describe the old implementation as multi-cluster.

---

# IMPORTANT RESOURCE-ALLOCATION DECISION

The Rosevin paper uses CPU-core allocation as its main SR resource-allocation variable.

Therefore the initial architecture should preserve CPU-core allocation as the paper-faithful baseline.

However, we have access to an Azure for Students subscription and may eventually test GPU-based SR.

Therefore:

- CPU allocation is the initial baseline.
- GPU/VRAM may become an extension.
- Do NOT redesign the entire system around GPU allocation now.
- Do NOT assume an Azure GPU is required.
- We will benchmark compute requirements before choosing the final deployment configuration.

---

# IMPORTANT STREAMING CONCEPTS

The new architecture must distinguish:

1. Target bitrate
2. Base bitrate

The client can have a target bitrate/quality requirement.

The adaptive system can select a lower base bitrate and use SR to recover quality.

Therefore the scheduler should eventually reason about something like:

    target bitrate
    base bitrate
    SR configuration
    CPU allocation
    cache state

Do NOT assume "bitrate" is a single variable.

---

# CACHING IS CORE

Caching is NOT something we want to bolt on later.

The edge needs to know whether a requested chunk/representation is already cached.

Conceptually the edge may encounter:

    Edge cache + no SR
    Edge cache + SR
    Cloud fetch + no SR
    Cloud fetch + SR

The existing implementation should therefore be audited for:

- caching
- cache state
- chunk lookup
- LRU or other eviction
- representation awareness
- cloud-origin fetching

If none exists, simply report that.

Do not implement it yet.

---

# REAL-TIME REQUIREMENT

Real-time means end-to-end streaming behavior, not merely model inference FPS.

Eventually we will consider:

    network transfer
    +
    decoding
    +
    SR
    +
    encoding
    +
    scheduling
    +
    delivery
    +
    client buffer

The client buffer is important.

A chunk taking 3 seconds does not necessarily cause a stall if the player has sufficient buffered video.

Therefore the future system will maintain an explicit buffer state.

Again: audit only. Do not implement this yet.

---

# NETWORK REQUIREMENT

The previous implementation did not properly implement the real network path.

The new system MUST eventually have:

    Cloud → Edge
    Edge → Client

as genuine communication paths.

Network characteristics will eventually include:

- RTT
- throughput
- jitter
- packet loss
- transfer time

Do not replace these with hardcoded network-quality labels unless the old code is explicitly a test/emulation utility.

During development we may use controlled network emulation, but the architecture must preserve the network boundary.

---

# WHAT TO AUDIT

Perform a systematic audit of the entire repository.

Inspect:

- directory structure
- Python files
- configuration files
- scripts
- model wrappers
- video utilities
- FFmpeg/OpenCV usage
- chunking
- encoding/decoding
- frame extraction
- FPS handling
- preprocessing
- scene/content analysis
- SR inference
- model benchmarking
- metrics
- logging
- experiment scripts
- datasets
- APIs
- networking
- caching
- resource monitoring
- GPU handling
- CPU handling
- Docker/container files if present
- cloud/deployment files if present
- requirements/environment files

Do not only inspect filenames.

Trace the actual execution flow of the main entry points.

Identify which components are genuinely functional and which are placeholders.

---

# CLASSIFICATION SYSTEM

Every relevant component should be placed into exactly one of these categories:

## A. REUSE AS-IS

Use this only if the component is architecturally independent from the old local-only design.

Examples:

- FFmpeg video metadata extraction
- generic frame extraction
- generic PSNR/SSIM calculation
- video validation
- model benchmarking utilities
- generic logging

## B. REUSE WITH MODIFICATION

Use this when the underlying implementation is valuable but its integration assumes local processing.

Examples:

    old:
        Python function directly calls SR model

    future:
        same SR inference logic runs inside an Edge Service

Another example:

    old:
        local video processing loop

    future:
        same processing function operates on an independently received chunk

Explain exactly what must change.

## C. REPLACE

Use this when the component fundamentally conflicts with the new architecture.

Examples:

- local-only orchestration
- local device deciding all SR execution
- assumptions that GPU is always on the client
- pipelines where video never crosses a network boundary
- timing logic that assumes zero transmission latency

## D. UNKNOWN / NEEDS TESTING

Use this when the code looks reusable but you cannot confidently establish correctness without running it.

---

# FOR EACH COMPONENT REPORT

For each significant component, provide:

1. File path
2. Component/function/class
3. Purpose
4. Current behavior
5. Dependencies
6. Classification:
   - REUSE AS-IS
   - REUSE WITH MODIFICATION
   - REPLACE
   - UNKNOWN / NEEDS TESTING
7. Reason
8. What future AdaptiveSR component it could become
9. Any architectural risks

Example:

    File:
        src/video/video_utils.py

    Component:
        extract_video_metadata()

    Purpose:
        Extract FPS, resolution, duration and codec.

    Classification:
        REUSE AS-IS

    Reason:
        Independent of local/cloud execution.

    Future role:
        Client-side video profiling.

---

# TRACE THE OLD EXECUTION PIPELINE

Find the main entry point(s) and construct the actual current pipeline.

For example, determine whether the current system is effectively:

    Video
      ↓
    preprocessing
      ↓
    local SR
      ↓
    metrics
      ↓
    output

or something else.

Do not assume.

Trace imports and function calls to determine the real execution flow.

Represent the result as a diagram.

Then identify every point where the old architecture assumes:

- local execution
- local GPU
- zero network latency
- direct function calls
- local filesystem access
- client-side SR

---

# IDENTIFY REUSABLE ASSETS

Create a separate list of reusable assets.

For example:

    Video ingestion
    Metadata extraction
    FFmpeg utilities
    Frame extraction
    SR model wrappers
    Metrics
    Benchmark scripts
    Content analysis
    Dataset handling

For each, give its file path and classification.

---

# IDENTIFY MISSING COMPONENTS

Do NOT implement them.

Only identify what the new architecture will need that does not exist.

Expected categories include:

    Cloud/origin service
    Edge service
    Client service
    Controller/scheduler
    Network communication
    Chunk transfer
    Cache manager
    Cache state
    Resource monitoring
    CPU allocation
    Client buffer
    Telemetry
    Request/response protocol
    Cloud → edge path
    Edge → client path

Only report what is actually missing after auditing the repository.

---

# IDENTIFY ARCHITECTURAL COUPLING

Pay particular attention to code that assumes:

    "the SR model is called directly from the client"

or:

    "the video file is always locally accessible"

or:

    "all processing happens inside one Python process"

or:

    "GPU = local machine"

or:

    "network = zero"

These are likely the most important pieces that need redesign.

---

# DO NOT IMPLEMENT ANYTHING

This is an audit-only task.

Do NOT:

- create new architecture
- rewrite code
- move files
- delete files
- rename files
- install packages
- create cloud resources
- create Azure VMs
- modify requirements
- modify configuration
- refactor existing code

The only output should be an audit report.

---

# OUTPUT

Create:

    OLD_IMPLEMENTATION_AUDIT.md

at the repository root.

The report must contain:

1. Executive summary
2. Repository structure
3. Current execution pipeline
4. Reuse-as-is components
5. Reuse-with-modification components
6. Components to replace
7. Components requiring testing
8. Existing SR/model infrastructure
9. Existing video/chunk infrastructure
10. Existing metrics/benchmark infrastructure
11. Existing network/cloud infrastructure
12. Existing caching infrastructure
13. Existing resource-monitoring infrastructure
14. Architectural coupling to local execution
15. Missing components for AdaptiveSR
16. Recommended migration path
17. Risks / technical debt
18. Suggested order for extracting reusable code

At the end, include this summary table:

| Component           | Existing? | Classification | Future Role |
| ------------------- | --------- | -------------- | ----------- |
| Video ingestion     |           |                |             |
| Metadata extraction |           |                |             |
| Chunking            |           |                |             |
| Frame processing    |           |                |             |
| SR inference        |           |                |             |
| Model benchmarking  |           |                |             |
| Metrics             |           |                |             |
| Content analysis    |           |                |             |
| Networking          |           |                |             |
| Cloud/origin        |           |                |             |
| Edge service        |           |                |             |
| Cache               |           |                |             |
| Resource monitoring |           |                |             |
| Client buffer       |           |                |             |
| Scheduler           |           |                |             |
| Telemetry           |           |                |             |

Finally provide:

## Recommended next action

Do NOT begin Step 0 implementation yet.

Instead, conclude with exactly what should be done after the audit, based on what you actually found in the repository.

The goal is to let us make an informed decision about the new Step 0 architecture without throwing away useful previous work or accidentally carrying local-only assumptions into the new system.

Again:

**AUDIT ONLY. DO NOT MODIFY CODE.**
