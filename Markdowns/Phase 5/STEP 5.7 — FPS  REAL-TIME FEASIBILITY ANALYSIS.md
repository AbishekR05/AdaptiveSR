============================================================
STEP 5.7 — FPS / REAL-TIME FEASIBILITY ANALYSIS
IMPLEMENTATION SPECIFICATION
============================================================

STEP 0–4 ARE FROZEN.

STEP 5.1–5.6 ARE FROZEN.

Implement ONLY:

    STEP 5.7 — FPS / REAL-TIME FEASIBILITY ANALYSIS

DO NOT implement Step 5.8 or any later step.

============================================================
1. OBJECTIVE
============================================================

Step 5.7 must determine whether an SR model configuration can
keep up with the source video's frame rate based on the measured
inference latency produced by Step 5.5.

The core question is:

    "Given the measured SR inference latency and the source FPS,
     can this configuration process frames within the available
     per-frame time budget?"

This is an ANALYSIS layer.

It does NOT perform adaptive decisions.

It does NOT select SR models.

It does NOT perform scheduling.

It does NOT modify network behavior.

It does NOT implement streaming.

============================================================
2. IMPORTANT SCOPE DISTINCTION
============================================================

Step 5.7 measures:

    SR inference-only real-time feasibility.

It does NOT measure end-to-end streaming feasibility.

The following are OUTSIDE Step 5.7:

    video decoding
    preprocessing
    video encoding
    network transfer
    network latency
    network bandwidth
    client playback
    buffering
    HTTP overhead
    edge scheduling
    multi-edge selection

Therefore DO NOT claim:

    "the system is end-to-end real-time"

based only on Step 5.7.

Use wording such as:

    "SR inference-only real-time feasible"

when appropriate.

============================================================
3. INPUT
============================================================

Read benchmark results produced by Step 5.5.

Use the existing Step 5.5 benchmark result schema and existing
project interfaces.

DO NOT create a second incompatible benchmark schema.

Relevant information includes, where available:

    model_id
    scale
    device
    latency_ms
    p95_latency_ms
    input_id
    cpu_affinity_config
    session_count
    decision_eligible
    eligibility_reason
    p95_confidence

Do not assume that every possible model/device/scale combination
exists.

If a configuration is absent from Step 5.5 results:

    represent it as unavailable / not measured.

DO NOT fabricate benchmark values.

============================================================
4. SOURCE FPS
============================================================

Source FPS must come from authoritative project metadata.

Use the existing Step 5.1 benchmark manifest / input metadata
and the Step 5.5 input identifier as the join key.

Do NOT hardcode:

    30 FPS

as the only supported source FPS.

The implementation must support:

    30 FPS
    60 FPS
    120 FPS

and should work with arbitrary positive FPS values when the
existing schema permits them.

Source FPS provenance must be retained in the analysis output.

Example:

    input_id
        ↓
    benchmark manifest
        ↓
    source_fps

Document the exact provenance.

============================================================
5. FRAME BUDGET
============================================================

For source FPS:

    source_fps

calculate:

    frame_budget_ms = 1000 / source_fps

Examples:

    30 FPS  → 33.333... ms
    60 FPS  → 16.666... ms
    120 FPS → 8.333... ms

Do NOT round the budget prematurely.

Use sufficient numerical precision internally.

============================================================
6. ESTIMATED PROCESSING FPS
============================================================

Given measured inference latency:

    latency_ms

calculate:

    estimated_fps = 1000 / latency_ms

Reject invalid/non-positive latency values rather than producing
nonsensical results.

============================================================
7. REAL-TIME RATIO
============================================================

Calculate:

    real_time_ratio = frame_budget_ms / latency_ms

Interpretation:

    ratio >= 1.0
        inference latency fits within the frame budget.

    ratio < 1.0
        inference latency exceeds the frame budget.

Equivalent feasibility condition:

    latency_ms <= frame_budget_ms

Use an explicit boolean:

    realtime_measured

Do not infer feasibility from estimated FPS alone when a clearer
latency-vs-budget comparison is available.

============================================================
8. P95 ANALYSIS
============================================================

Where Step 5.5 provides p95 latency:

    p95_latency_ms

calculate the corresponding p95 budget comparison.

Do NOT present p95 as a production guarantee.

Respect Step 5.5's existing:

    p95_confidence

field.

If p95 confidence is:

    exploratory

label the result:

    non-decisive / exploratory

Do not use exploratory p95 as the headline production-feasibility
claim.

If Step 5.5 already provides the p95 confidence and eligibility
semantics, reuse them rather than redefining them.

============================================================
9. DECISION ELIGIBILITY
============================================================

IMPORTANT:

    MEASURED FEASIBILITY
and
    DECISION ELIGIBILITY

are different concepts.

A configuration may be:

    measured feasible
but
    not decision-eligible.

Respect Step 5.5's existing multi-session eligibility rules.

The current frozen methodology requires sufficient repeated
sessions before a configuration is eligible for decision use.

Do NOT mark a configuration decision-eligible merely because:

    latency < frame budget.

Carry through:

    decision_eligible
    eligibility_reason
    session_count

from the Step 5.5 benchmark evidence where applicable.

============================================================
10. REQUIRED CLASSIFICATIONS
============================================================

Every available configuration should be classified into one of
these categories:

A. MEASURED + DECISION-ELIGIBLE

    latency fits frame budget
    AND
    Step 5.5 eligibility requirements are satisfied.

B. MEASURED FEASIBLE BUT NOT DECISION-ELIGIBLE

    latency fits frame budget
    BUT
    insufficient evidence / variance / eligibility requirements.

C. MEASURED UNFEASIBLE

    latency exceeds frame budget.

D. NOT MEASURED

    required benchmark data is unavailable.

Do NOT merge these categories.

============================================================
11. SCALE ANALYSIS
============================================================

If multiple scales for the SAME model are actually present in
the benchmark dataset, report their measured configurations.

For example:

    x2
    x4

However:

DO NOT create a "scale degradation trend" unless multiple scales
are actually measured.

If only x2 exists, explicitly state:

    "No scale-degradation trend can be established from the
     currently available benchmark data because only the x2
     configuration has been measured."

Do not infer x4 performance.

============================================================
12. CPU vs GPU ANALYSIS
============================================================

If matching CPU and GPU measurements exist for the same:

    model
    scale
    input configuration

a comparison may be reported.

If no matching GPU result exists:

    report GPU comparison as unavailable / N/A.

Do NOT estimate GPU latency from CPU latency.

Do NOT invent GPU benchmark results.

Preserve an explicit indicator such as:

    gpu_benchmark_gap = true

when appropriate.

============================================================
13. CURRENT DATA COVERAGE
============================================================

The existing Step 5.7 evidence may contain only:

    tinysr
    x2
    CPU

with a single benchmark record.

This is a COVERAGE LIMITATION.

Do NOT compensate by:

    running unrequested broad benchmark campaigns
    fabricating missing configurations
    assuming results from another machine
    copying values from previous experiments
    extrapolating GPU performance

Step 5.7 must analyze whatever valid Step 5.5 data currently
exists.

Broader benchmark coverage belongs to additional Step 5.5
execution, not to invented Step 5.7 data.

============================================================
14. OUTPUT DATASET
============================================================

Create a machine-readable Step 5.7 analysis output.

Use the existing project data conventions.

Each analysis record should retain enough information to trace the
result back to Step 5.5.

At minimum include:

    model_id
    scale
    device
    input_id
    source_fps
    latency_ms
    p95_latency_ms
    frame_budget_ms
    estimated_fps
    real_time_ratio
    realtime_measured
    p95_realtime_measured
    p95_confidence
    session_count
    decision_eligible
    eligibility_reason
    interpretation
    gpu_benchmark_gap where applicable

Do not duplicate fields unnecessarily if an existing schema can be
extended cleanly.

============================================================
15. REPORT
============================================================

Generate a human-readable report.

The report should contain:

------------------------------------------------------------
A. Executive Summary
------------------------------------------------------------

Report:

    fastest measured configuration

if such a configuration exists.

If none is decision-eligible, explicitly say:

    "Fastest Real-Time Eligible Configuration: None found."

Do not confuse fastest measured with decision-eligible.

------------------------------------------------------------
B. Quantitative Benchmark Comparison
------------------------------------------------------------

Include a table containing:

    Model
    Scale
    Device
    Median Latency
    p95 Latency
    p95 Feasibility
    Estimated FPS
    Source FPS
    Frame Budget
    Real-Time Ratio
    Real-Time Measured
    Decision Eligible
    Session Count
    p95 Confidence
    Interpretation
    Eligibility/Caveat

------------------------------------------------------------
C. Feasibility Classification
------------------------------------------------------------

Separate:

    Measured + Decision-Eligible
    Measured Feasible Only
    Failing Configurations
    Not Measured

------------------------------------------------------------
D. Scale Analysis
------------------------------------------------------------

Only report trends supported by actual data.

------------------------------------------------------------
E. CPU vs GPU
------------------------------------------------------------

Only report comparisons where matching measurements exist.

Otherwise clearly report:

    GPU benchmark gap / unavailable.

------------------------------------------------------------
F. Limitations
------------------------------------------------------------

Explicitly document:

    current benchmark coverage
    session count limitations
    p95 confidence
    inference-only scope
    absence of network/encoding/decode measurements
    GPU coverage gaps
    scale coverage gaps where applicable

============================================================
16. TESTING
============================================================

Add automated tests for:

1. Correct 30 FPS frame budget.

2. Correct 60 FPS frame budget.

3. Correct 120 FPS frame budget.

4. Correct estimated FPS calculation.

5. Correct real-time ratio calculation.

6. Latency exactly equal to frame budget
   → feasible.

7. Latency below frame budget
   → feasible.

8. Latency above frame budget
   → infeasible.

9. Invalid zero latency rejected.

10. Invalid negative latency rejected.

11. Missing source FPS handled correctly.

12. Missing benchmark record represented as unavailable.

13. Decision eligibility remains separate from measured
    feasibility.

14. Exploratory p95 is not presented as a decisive headline
    result.

15. Multiple scales do not produce a trend unless actually
    measured.

16. Missing GPU data does not fabricate a CPU/GPU comparison.

17. Source FPS is obtained from authoritative metadata rather
    than hardcoded.

18. Existing Step 5.5 result schema remains compatible.

19. Existing Steps 0–5.6 tests continue to pass.

============================================================
17. DATA INTEGRITY
============================================================

Do not modify the underlying Step 5.5 benchmark measurements.

Step 5.7 is an analysis layer.

Do not overwrite raw benchmark results.

Do not silently alter latency values.

Do not silently alter session counts.

Do not silently alter eligibility status.

Derived values must be reproducible from the raw Step 5.5 data.

============================================================
18. NON-GOALS
============================================================

DO NOT implement:

    adaptive FPS
    adaptive bitrate
    ABR
    SR model selection
    edge selection
    resource allocation
    scheduling
    network adaptation
    network emulation
    online learning
    ML decision engine
    cloud deployment
    multi-edge scheduling
    end-to-end streaming
    client-side adaptive playback
    model retraining

Those belong to later steps.

============================================================
19. DOCUMENTATION
============================================================

Create/update:

    STEP5.7_IMPLEMENTATION.md

Document:

    objective
    inputs
    Step 5.5 dependency
    source FPS provenance
    frame-budget formula
    estimated-FPS formula
    real-time-ratio formula
    feasibility rule
    p95 semantics
    decision eligibility semantics
    classification rules
    output schema
    report format
    limitations
    known benchmark coverage gaps
    non-goals

Explicitly state:

    Step 5.7 evaluates SR inference-only real-time feasibility.

Explicitly state:

    Step 5.7 does not establish end-to-end streaming
    real-time feasibility.

============================================================
20. REGRESSION
============================================================

Run:

    python -m pytest tests/ -v

The complete existing suite must remain passing.

Also run the Step 5.7-specific tests.

============================================================
21. FINAL REPORT TO ME
============================================================

When complete, return:

1. Files created.
2. Files modified.
3. Step 5.5 input/result source used.
4. Exact source-FPS provenance.
5. Frame-budget calculation.
6. Estimated-FPS calculation.
7. Real-time-ratio calculation.
8. Feasibility classification logic.
9. p95 handling.
10. Decision-eligibility handling.
11. Current benchmark coverage.
12. GPU coverage status.
13. Scale coverage status.
14. Example machine-readable output.
15. Example human-readable report.
16. Tests added.
17. Full pytest result.
18. Confirmation that Steps 0–5.6 remain unchanged/frozen.
19. Known limitations.

============================================================
STOP CONDITION
============================================================

STOP after Step 5.7.

Do NOT begin Step 5.8.

Do NOT run a broad new benchmark campaign unless explicitly
requested.

Do NOT modify frozen Step 0–5.6 implementation merely to make
Step 5.7 easier.

The goal is a clean, reproducible feasibility-analysis layer
built on top of the existing Step 5.5 empirical benchmark data.
============================================================