STEP 2.2 — CHUNK-TO-REPRESENTATION MAPPING

Step 0, Step 0.1, Step 1, and Step 2.1 are FROZEN.

Begin ONLY Step 2.2.

Do not implement Step 2.3 or later.

============================================================
OBJECTIVE
============================================================

Establish the mapping between the authoritative logical chunk
timeline produced by Step 1 and every configured video
representation.

The central invariant is:

    logical chunk N

must refer to the same temporal source interval across every
representation.

Example:

    360p/chunk_007
    480p/chunk_007
    720p/chunk_007

must correspond to the same logical source chunk.

============================================================
AUTHORITATIVE TIMELINE
============================================================

Step 1's chunk boundaries are authoritative.

Do NOT independently invent a different chunk timeline for each
representation.

Do NOT allow each representation encoder to independently choose
arbitrary segment boundaries.

The mapping must preserve:

    chunk_id
    logical start frame
    logical end frame
    logical start timestamp
    logical end timestamp

across representations.

============================================================
REPRESENTATION-CHUNK MODEL
============================================================

Create a model representing a chunk within a representation.

It should contain at minimum:

    chunk_id
    representation_id
    frame_start
    frame_end
    start_time_seconds
    end_time_seconds
    duration_seconds
    file_path
    size_bytes

Use the existing project's schema conventions.

Do not duplicate information unnecessarily if the project already
has an appropriate shared chunk model.

============================================================
FRAME RANGE
============================================================

For each logical chunk, frame ranges must come from Step 1.

For example:

    chunk_000:
        frame_start = 0
        frame_end   = 59

    chunk_001:
        frame_start = 60
        frame_end   = 119

The exact values depend on source FPS and actual Step 1 output.

Do NOT hardcode 60 frames per chunk.

30/60/120 FPS sources must all work.

============================================================
REPRESENTATION MAPPING
============================================================

For every configured representation:

    representation × logical_chunk

must produce exactly one mapping entry.

If there are:

    3 representations
    10 logical chunks

the mapping must contain:

    30 representation-chunk entries

No missing combinations.
No duplicate combinations.

============================================================
TIMESTAMP CONTRACT
============================================================

The logical chunk's temporal interval is authoritative.

Representation-specific metadata may contain tiny timestamp
differences caused by encoding/container behavior.

Do not silently overwrite the authoritative logical interval.

Instead, retain the logical interval and, if useful, store
representation/container timing separately.

Do not assume every chunk is exactly 2.0 seconds.

Always use:

    duration_seconds

from the actual chunk/timeline metadata.

============================================================
FILE EXISTENCE
============================================================

At this stage, the implementation may operate on representation
metadata/mapping without performing full video encoding if the
project architecture requires encoding to be implemented in the
next step.

However, if existing code already generates representation files,
the mapping layer should validate that the expected files exist.

Do not introduce a full encoding pipeline unless necessary for
Step 2.2.

============================================================
VALIDATION INVARIANTS
============================================================

Implement validation for:

1.  Every logical chunk maps to every configured representation.

2.  No duplicate:

        (representation_id, chunk_id)

    pairs.

3.  Logical frame ranges are identical across representations.

4.  Logical start/end timestamps are identical across
    representations.

5.  frame_start <= frame_end.

6.  duration_seconds > 0.

7.  Chunk ordering is monotonic.

8.  No logical chunk gaps.

9.  No logical chunk overlaps.

10. The first logical chunk begins at the source's first frame.

11. The final logical chunk ends at the source's final frame.

12. Representation IDs referenced by mappings must exist in the
    representation configuration.

============================================================
FPS HANDLING
============================================================

Do not convert FPS.

Do not implement FPS adaptation.

If the source is:

    30 FPS

the representation mapping must preserve that source timeline.

If the source is:

    60 FPS

preserve the 60 FPS timeline.

If the source is:

    120 FPS

preserve the 120 FPS timeline.

FPS adaptation is a later system capability.

============================================================
IMPORTANT DISTINCTION

Do not confuse:

    logical chunk identity

with:

    physical encoded file identity.

The logical chunk is:

    chunk_007

A representation-specific physical object is:

    360p/chunk_007
    480p/chunk_007
    720p/chunk_007

They are different files representing the same logical interval.

============================================================
TESTS

Add tests for:

1. One representation × multiple chunks.

2. Multiple representations × multiple chunks.

3. Every representation receives every logical chunk.

4. No duplicate representation/chunk pairs.

5. Frame ranges remain identical across representations.

6. Timestamp ranges remain identical across representations.

7. Missing representation mapping is rejected.

8. Unknown representation ID is rejected.

9. Chunk gaps are rejected.

10. Chunk overlaps are rejected.

11. Non-monotonic chunk ordering is rejected.

12. 30 FPS source mapping.

13. 60 FPS source mapping.

14. 120 FPS source mapping.

15. Variable final chunk duration is accepted.

16. No hardcoded 2-second duration assumption exists.

============================================================
REGRESSION

Run the complete existing suite:

    python -m pytest tests/ -v

Do not break Step 0/0.1 or Step 1.

============================================================
NON-GOALS

Do NOT implement:

- FFmpeg encoding ladder generation
- manifest generation
- Cloud deployment
- Edge scheduling
- ABR
- SR
- ML
- network emulation
- GPU allocation
- Azure deployment

============================================================
DOCUMENTATION

Update STEP2_IMPLEMENTATION.md with:

    Step 2.2 — Chunk-to-Representation Mapping

Clearly document:

    logical chunk
        =
    authoritative temporal unit

and:

    representation chunk
        =
    encoded representation of that logical unit.

Explicitly state that all representations inherit the Step 1
logical timeline.

============================================================

STOP after Step 2.2.

Report:

- files changed
- models added/modified
- mapping structure
- invariants
- tests added
- complete test results
- any concerns discovered

Do NOT begin Step 2.3 automatically.
