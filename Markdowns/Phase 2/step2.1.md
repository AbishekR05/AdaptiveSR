STEP 2.1 — REPRESENTATION SCHEMA AND CONTRACT

Step 1 is FROZEN.

Begin Step 2.

Do NOT implement the entire Step 2.
Implement ONLY Step 2.1: the representation schema and
configuration contract.

============================================================
OBJECTIVE
============================================================

Establish the data model for multiple encoded video
representations that will later be chunked, served by Cloud
Origin, cached by Edge, and eventually used by the SR scheduler.

A representation is a pre-encoded version of the same source video
with a specific spatial resolution/quality and bitrate.

============================================================
REQUIRED REPRESENTATION FIELDS
============================================================

Each representation must define:

    representation_id
    width
    height
    resolution_label
    bitrate_kbps
    codec
    fps

Do not hardcode the project to only 360p/480p/720p.
The schema must support arbitrary representation entries.

============================================================
FPS REQUIREMENT
============================================================

Input source videos may be:

    30 FPS
    60 FPS
    120 FPS

Do NOT automatically reduce FPS.

Do NOT implement FPS adaptation.

Do NOT introduce a 120→30 FPS rule.

FPS is simply metadata of the representation at this stage.

============================================================
REPRESENTATION CONFIGURATION
============================================================

Create a configuration/schema capable of expressing something
like:

representations:

    - id: 360p
      width: 640
      height: 360
      bitrate_kbps: 800
      codec: h264
      fps: source

    - id: 480p
      width: 854
      height: 480
      bitrate_kbps: 1400
      codec: h264
      fps: source

    - id: 720p
      width: 1280
      height: 720
      bitrate_kbps: 2500
      codec: h264
      fps: source

The exact configuration format may follow the existing project's
conventions.

If "fps: source" is used, resolve it to the actual source FPS when
the representation metadata is materialized.

============================================================
IMPORTANT DISTINCTION
============================================================

Do NOT introduce:

    target_representation_id
    base_representation_id

as adaptive decisions here.

Those fields already exist as telemetry groundwork from Step 0.1.

Step 2.1 only defines what representations EXIST.

The future scheduler will decide:

    target representation
    base representation

later.

============================================================
VALIDATION
============================================================

Implement schema validation for:

- unique representation IDs
- positive width
- positive height
- positive bitrate
- supported codec identifier
- valid FPS
- no duplicate resolution/representation conflicts unless
  explicitly allowed by the schema

Do not impose unnecessary assumptions such as:

    bitrate must increase monotonically with resolution

That is normally expected but is not a schema requirement.

============================================================
TESTS
============================================================

Add tests verifying:

1. Valid representation configuration is accepted.
2. Duplicate representation IDs are rejected.
3. Invalid resolution is rejected.
4. Invalid bitrate is rejected.
5. Invalid FPS is rejected.
6. Multiple FPS values (30/60/120) are accepted.
7. Multiple representations can coexist.
8. Representation schema does not require a target/base decision.

Run the existing Step 0/0.1/1 regression suite as well.

============================================================
NON-GOALS

Do NOT implement:

- video encoding
- FFmpeg representation generation
- chunk generation
- manifest generation
- SR
- ABR
- ML
- network simulation
- Azure deployment
- resource allocation
- scheduler logic

============================================================
OUTPUT

Report:

- files created/modified
- representation schema
- validation rules
- tests added
- full regression results

STOP after Step 2.1.

Do not proceed to Step 2.2 automatically.
