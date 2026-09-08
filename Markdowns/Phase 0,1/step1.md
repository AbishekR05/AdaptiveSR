STEP 1 PRE-IMPLEMENTATION AUDIT — DO NOT IMPLEMENT ANYTHING

Step 0 and Step 0.1 are now FROZEN.

The foundation has passed 13/13 tests.

We are now preparing Step 1: VIDEO & CONTENT PROFILING.

IMPORTANT:
This is an AUDIT ONLY.
Do not modify source code.
Do not create the Step 1 implementation.
Do not introduce SR, ABR, ML, adaptive FPS, resource allocation, Azure deployment, or network emulation.

==================================================
OBJECTIVE
==================================================

Inspect the existing/legacy implementation and identify which
components can be reused for Step 1's video/content profiling.

We specifically want to investigate components related to:

- video loading
- video metadata extraction
- frame extraction
- FPS handling
- scene analysis
- motion estimation
- complexity estimation
- spatial/content analysis
- temporal analysis
- chunking
- bitrate estimation
- quality metrics

Likely locations include the legacy `src/` directory and any
associated utilities/tests/scripts.

==================================================
STEP 1 TARGET
==================================================

Step 1 will eventually produce source-side video/chunk profiles.

Conceptually:

INPUT VIDEO
↓
VIDEO PROFILER
↓
video-level metadata +
chunk-level content characteristics
↓
reproducible profiling dataset

Supported source FPS:

    30 FPS
    60 FPS
    120 FPS

Do NOT assume that all videos are 30 FPS.

==================================================
IMPORTANT ARCHITECTURAL CONSTRAINT
==================================================

Step 1 features must be computable from information available
from the SOURCE VIDEO before the adaptive SR decision is made.

Do NOT recommend features that depend on:

- SR output
- post-SR quality
- future playback outcomes
- future network conditions
- future edge processing
- future ML predictions

Those would create information leakage.

==================================================
AUDIT EACH COMPONENT
==================================================

For every relevant legacy component, report:

1. File/path
2. Component/class/function name
3. What it currently does
4. Inputs
5. Outputs
6. Metrics/features it calculates
7. Whether it supports 30/60/120 FPS
8. Whether it operates at video-level or chunk-level
9. Whether it is source-side and available before transmission
10. Whether it can be reused unchanged
11. Whether it requires modification
12. Whether it should be discarded
13. Why

Use these classifications:

    REUSE AS-IS
    REUSE WITH MODIFICATION
    NEW IMPLEMENTATION REQUIRED
    NOT NEEDED
    POTENTIAL DATA LEAKAGE

==================================================
FEATURE AUDIT
==================================================

Create a table of every potentially useful feature currently
available in the legacy code.

For each feature:

    Feature
    Existing implementation
    Computation cost
    Video/chunk level
    Source-side available?
    Relevant to later AdaptiveSR?
    Recommendation

Do not invent features merely because they sound useful.

==================================================
CHUNKING AUDIT
==================================================

Determine whether the old implementation already has a reliable
chunking mechanism.

Report:

- chunk duration
- frame boundaries
- handling of 30 FPS
- handling of 60 FPS
- handling of 120 FPS
- whether chunk boundaries are deterministic
- whether audio is considered
- whether chunk metadata is persisted

Do not modify the chunking implementation.

==================================================
FPS AUDIT
==================================================

Explicitly inspect how the legacy system handles:

    30 FPS
    60 FPS
    120 FPS

Identify:

- assumptions that break at higher FPS
- hardcoded FPS values
- frame skipping/downsampling
- timing calculations
- frame-count assumptions

Do not fix anything yet.

==================================================
OUTPUT
==================================================

Produce:

1. STEP1_LEGACY_AUDIT.md

2. A concise recommendation containing:

   A. Components to reuse
   B. Components requiring modification
   C. Components to discard
   D. Features worth carrying into Step 1
   E. Features that should NOT be used
   F. Missing capabilities that Step 1 must implement
   G. Potential data-leakage risks
   H. 30/60/120 FPS compatibility risks

3. Do NOT modify existing code.

4. Do NOT modify Step 0 or Step 0.1.

STOP after producing the audit.

==================================================
SUCCESS CONDITION
==================================================

We should finish with enough information to design the exact
Step 1 implementation contract before writing any new code.

Do not proceed beyond the audit.
