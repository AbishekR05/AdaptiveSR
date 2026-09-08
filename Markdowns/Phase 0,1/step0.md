We have completed the OLD_IMPLEMENTATION_AUDIT.md audit.

IMPORTANT:
We are now beginning STEP 0 of the new AdaptiveSR implementation.

Do NOT modify or migrate the old local pipeline yet.
Do NOT refactor src/main.py.
Do NOT implement SR.
Do NOT implement ML.
Do NOT provision Azure resources.
Do NOT implement BasicVSR++.
Do NOT implement GPU scheduling.
Do NOT implement multi-cluster deployment.
Do NOT implement full MPEG-DASH/HLS.

We first need a clean distributed foundation.

==================================================
STEP 0 — DISTRIBUTED FOUNDATION
==================================================

Goal:

Build the minimum working distributed prototype:

    Cloud/Origin
          ↓ HTTP
       Edge Server
          ↓ HTTP
        Client

The system must use actual network communication between independently running processes/services.

Initially all three services may run on the same development machine using different localhost ports.

This is DEVELOPMENT EMULATION ONLY.

Do not describe localhost services as geographically distributed edge/cloud infrastructure.

The architecture must however preserve the boundaries required for eventual Azure deployment.

--------------------------------------------------
1. NEW SERVICE STRUCTURE
--------------------------------------------------

Create a clean service-oriented structure.

Suggested structure:

    adaptive_sr/
    │
    ├── services/
    │   ├── cloud/
    │   │   ├── app.py
    │   │   ├── storage/
    │   │   └── ...
    │   │
    │   ├── edge/
    │   │   ├── app.py
    │   │   ├── cache/
    │   │   └── ...
    │   │
    │   └── client/
    │       ├── app.py
    │       ├── player/
    │       └── ...
    │
    ├── shared/
    │   ├── schemas/
    │   ├── config/
    │   └── utils/
    │
    ├── tests/
    │
    └── README.md

You may adjust the exact structure if there is a strong technical reason.

Do NOT move/delete the old src/ tree yet.

The old implementation must remain available for reference and benchmarking.

--------------------------------------------------
2. CLOUD / ORIGIN SERVICE
--------------------------------------------------

Implement a minimal FastAPI Cloud Origin service.

Responsibilities at Step 0:

- expose available videos
- expose a simple manifest
- serve stored video chunks
- identify video_id
- identify chunk_id
- identify representation

Do NOT implement adaptive decisions here.

Do NOT perform SR here.

Use a simple local storage directory for now.

Example:

    storage/
      videos/
        sample/
          manifest.json
          chunks/
            0000.mp4
            0001.mp4
            0002.mp4

Required endpoints should include something equivalent to:

    GET /health

    GET /videos

    GET /videos/{video_id}/manifest

    GET /videos/{video_id}/chunks/{chunk_id}

Return appropriate HTTP errors for missing videos/chunks.

--------------------------------------------------
3. EDGE SERVICE
--------------------------------------------------

Implement a separate FastAPI Edge service.

Responsibilities at Step 0:

- receive a chunk request from the client
- check whether the requested chunk exists in edge cache
- if cache miss:
      fetch the chunk from Cloud Origin over HTTP
- store the fetched chunk in edge cache
- return the chunk to the client

This is the first place where we establish the actual:

    Cloud → Edge

network boundary.

The edge must NOT import the Cloud application's Python functions.

The edge must communicate with Cloud using HTTP.

That requirement is critical.

--------------------------------------------------
4. EDGE CACHE
--------------------------------------------------

Implement the simplest useful cache first.

Use disk-backed caching.

Cache key should contain enough information to distinguish:

    video_id
    chunk_id
    representation_id

For example:

    sample__chunk_0001__360p_1000.mp4

The cache must support:

    get()
    put()
    contains()

Also expose cache hit/miss information for telemetry.

Do NOT implement sophisticated LRU eviction yet.

--------------------------------------------------
5. CLIENT SERVICE / PLAYER
--------------------------------------------------

Implement a minimal client program.

The client should:

1. Request the manifest.
2. Select a known representation manually for Step 0.
3. Request chunk 0 from the Edge.
4. Receive the chunk.
5. Verify its integrity.
6. Record download time.
7. Add the chunk duration to a simple playback buffer.
8. Request the next chunk.

Do NOT implement ABR yet.

Do NOT implement ML yet.

Do NOT implement adaptive FPS selection yet.

Do NOT implement SR yet.

The purpose is simply to prove that the distributed data path works.

--------------------------------------------------
6. CLIENT BUFFER
--------------------------------------------------

Create a simple mathematical playback buffer.

State should include at minimum:

    buffer_seconds
    chunk_duration
    playback_rate
    stall_count

When a chunk arrives:

    buffer += chunk_duration

As simulated playback progresses:

    buffer -= elapsed_time

If:

    buffer <= 0

record a stall event.

Do NOT build a real video player GUI.

This is a research client emulator, not VLC.

--------------------------------------------------
7. NETWORK TELEMETRY
--------------------------------------------------

The client should record at minimum:

    request_start_time
    response_time
    download_duration
    bytes_received
    measured_throughput_mbps
    RTT if measurable
    buffer_before
    buffer_after
    cache_hit/miss if returned by Edge

Use timestamps based on a monotonic clock for duration measurements.

Do NOT hardcode network latency.

--------------------------------------------------
8. EDGE TELEMETRY
--------------------------------------------------

The Edge should record:

    request_id
    video_id
    chunk_id
    representation_id
    cache_hit
    cloud_fetch_time
    edge_processing_time
    response_time
    bytes_sent

SR processing does not exist yet.

Leave a placeholder for future:

    sr_processing_time

--------------------------------------------------
9. SHARED SCHEMAS
--------------------------------------------------

Define typed request/response models.

At minimum define concepts for:

    VideoManifest
    Representation
    ChunkMetadata
    ChunkRequest
    ClientTelemetry
    EdgeTelemetry

Every chunk must be identifiable by:

    video_id
    chunk_id
    representation_id

Every edge must have:

    cluster_id
    edge_id

For Step 0 use:

    cluster_id = "cluster_01"
    edge_id = "edge_01"

These identifiers must exist even though only one logical cluster is deployed.

--------------------------------------------------
10. NO LOCAL FUNCTION SHORTCUTS
--------------------------------------------------

This is a HARD REQUIREMENT.

Do NOT do:

    client → Python function → edge logic

or:

    edge → import cloud.storage → read file

The services must communicate through HTTP.

The reason is that we are specifically correcting the old architecture's local-only design.

--------------------------------------------------
11. SAMPLE DATA
--------------------------------------------------

Do not require a large video dataset yet.

Use one small test video/chunk set.

If existing benchmark videos are suitable, reference/copy them into a dedicated Step-0 test-data directory without modifying the old benchmark data.

The test should be deterministic and fast.

--------------------------------------------------
12. TESTS
--------------------------------------------------

Implement tests for:

    Cloud health endpoint
    Manifest retrieval
    Chunk retrieval
    Edge cache miss → Cloud fetch
    Edge cache hit → no Cloud fetch
    Client → Edge request
    Buffer update
    Throughput calculation
    Missing chunk handling

Also implement ONE end-to-end integration test:

    Client
      ↓ HTTP
    Edge
      ↓ HTTP
    Cloud
      ↓
    Edge
      ↓ HTTP
    Client

The test must prove that the chunk actually crossed both HTTP boundaries.

--------------------------------------------------
13. RUNNABLE DEMO
--------------------------------------------------

Provide a simple way to start:

    Cloud
    Edge
    Client

Do not require Docker unless it is genuinely necessary.

Prefer plain Python processes initially.

For example:

    python -m services.cloud.app
    python -m services.edge.app
    python -m services.client.app

Use configurable ports.

Suggested development ports:

    Cloud: 8000
    Edge: 8001
    Client: 8002

These are suggestions, not hard requirements.

--------------------------------------------------
14. CONFIGURATION
--------------------------------------------------

Do not hardcode:

    cloud URL
    edge URL
    ports
    storage paths

Create a small configuration mechanism using environment variables or a config file.

Example:

    CLOUD_URL=http://localhost:8000
    EDGE_URL=http://localhost:8001
    CLUSTER_ID=cluster_01
    EDGE_ID=edge_01

--------------------------------------------------
15. LOGGING
--------------------------------------------------

Use structured logs.

Every request should be traceable using:

    request_id

The logs should make it possible to determine:

    Client requested chunk
    Edge received request
    Edge cache hit/miss
    Edge fetched from Cloud if necessary
    Edge returned chunk
    Client received chunk

--------------------------------------------------
16. SUCCESS CRITERIA
--------------------------------------------------

Step 0 is complete ONLY when we can demonstrate:

    1. Cloud starts independently.
    2. Edge starts independently.
    3. Client starts independently.
    4. Client requests a chunk from Edge.
    5. Edge checks cache.
    6. On cache miss, Edge requests chunk from Cloud using HTTP.
    7. Cloud returns chunk.
    8. Edge caches chunk.
    9. Edge returns chunk to Client.
    10. Client verifies chunk.
    11. Client updates playback buffer.
    12. Second request demonstrates an Edge cache hit.
    13. Telemetry records the complete path.

The resulting architecture should visibly demonstrate:

    CLIENT
       │
       │ HTTP
       ▼
    EDGE
       │
       │ HTTP
       ▼
    CLOUD

--------------------------------------------------
17. IMPORTANT NON-GOALS
--------------------------------------------------

Do NOT implement:

- SR inference
- FSRCNN integration
- Real-ESRGAN integration
- BasicVSR++
- ML
- online learning
- adaptive bitrate
- adaptive FPS
- CPU core optimization
- GPU allocation
- multi-cluster deployment
- Azure deployment
- sophisticated DASH/HLS
- motion-aware frame selection
- QoE optimization

These belong to later steps.

--------------------------------------------------
18. DOCUMENTATION
--------------------------------------------------

Create/update:

    STEP0_IMPLEMENTATION.md

Document:

- architecture
- services
- endpoints
- schemas
- data flow
- how to run
- how to test
- known limitations
- what Step 1 will add

Do not claim that the local three-process setup is cloud deployment.

Call it:

    "local distributed emulation of the Cloud → Edge → Client architecture."

--------------------------------------------------
FINAL INSTRUCTION
--------------------------------------------------

Implement ONLY Step 0.

Do not proceed automatically to Step 1 after Step 0 succeeds.

At the end, report:

1. Files created
2. Files modified
3. Tests added
4. Test results
5. How to run the three services
6. Example end-to-end request flow
7. Any issues encountered
8. Confirmation that the old local implementation remains untouched

STOP after Step 0.