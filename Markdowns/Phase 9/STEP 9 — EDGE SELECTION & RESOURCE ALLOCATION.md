Implement STEP 9 — EDGE SELECTION & RESOURCE ALLOCATION.

IMPORTANT:
- Steps 0–8 are FROZEN.
- Do NOT modify their contracts or methodology.
- Do NOT implement Step 10's final Adaptive Decision Engine.
- Do NOT implement global representation/model selection.
- Do NOT invent unsupported resource measurements.
- Reuse existing Step 5 instrumentation and Step 6 runtime telemetry wherever applicable.

OBJECTIVE:

Build the Edge Resource Evaluation layer that determines whether available Edge nodes can support a requested SR workload under their current compute and network conditions, and produces machine-readable candidate/resource signals for Step 10.

STEP 9 SUBTOPICS:

9.1 EDGE RESOURCE DISCOVERY

Represent available Edge resources using existing project abstractions where possible:

- edge_id
- CPU availability/capacity
- GPU availability
- GPU memory
- CPU utilization
- GPU utilization
- supported device
- relevant runtime capability

Do not fabricate resource values.

9.2 CURRENT RESOURCE STATE

Represent current Edge load separately from hardware capability.

A capable GPU under heavy utilization must not be treated as equivalent to an idle GPU.

Use actual measurements when available.
Missing values must remain null/unknown.

9.3 NETWORK CONDITIONS

Consume existing network telemetry where available, including:
- Cloud → Edge RTT
- available transfer/bandwidth measurements

Do not treat RTT alone as complete end-to-end streaming latency.

Keep network telemetry separate from SR inference latency.

9.4 RESOURCE FEASIBILITY

For a requested workload:

- model_id
- scale
- device
- base representation
- target resolution
- source FPS

evaluate whether an Edge candidate has the required capabilities/resources.

Produce explicit states such as:
- feasible
- infeasible
- unknown

Do not silently fall back to another device.

Do not make final adaptation decisions.

9.5 CANDIDATE EVALUATION / RANKING

For each Edge candidate expose measurable comparison attributes such as:

- SR processing capability where measured
- CPU/GPU load
- available GPU memory
- network RTT
- transfer/network cost where measurable
- resource feasibility

Do NOT invent arbitrary weighted utility coefficients.

Do NOT create the final multi-objective decision score.
Step 10 will combine the signals and make the final adaptive decision.

If a deterministic ranking is implemented, document its exact rule and ensure it only uses supported measurements.

9.6 MACHINE-READABLE SIGNAL

Create an Edge/Resource adaptation signal containing at minimum:

- edge_id
- model_id
- scale
- device
- base_representation_id
- target_resolution
- resource_feasible
- network_feasible
- CPU/resource telemetry
- GPU/resource telemetry
- network telemetry
- measurement provenance
- warnings

Preserve unknown/missing values as null rather than fabricating them.

Clearly distinguish:
- hardware capability
- current resource availability
- measured SR performance
- network condition
- feasibility

9.7 VALIDATION

Add tests for:

- GPU-capable feasible Edge
- CPU-only Edge
- unavailable requested GPU
- insufficient GPU memory
- overloaded Edge
- missing resource telemetry
- missing network telemetry
- multiple Edge candidates
- feasible vs infeasible candidates
- RTT not being treated as end-to-end latency
- no silent CPU fallback
- provenance preservation

If practical, include ONE real local integration test using the existing Step 6 Edge runtime and actual TinySR inference/resource telemetry.

8. SCOPE BOUNDARY

Step 9 MUST NOT:
- choose the final representation
- choose the final SR model
- implement bitrate adaptation
- implement FPS adaptation
- implement the final Adaptive Decision Engine
- create an arbitrary global utility function
- modify Steps 0–8

Step 9 produces candidate/resource evidence for Step 10.

DOCUMENTATION:

Document:
- resource-state schema
- capability vs availability
- feasibility rules
- network metric semantics
- missing-data behavior
- provenance
- candidate comparison semantics
- relationship to Steps 7 and 8
- explicit limitations

STOP after Step 9.

Return:
1. files changed
2. implementation summary
3. exact signal schema
4. feasibility/ranking rules
5. tests executed + actual terminal output/results
6. sample machine-readable output
7. BLOCKER / IMPORTANT / NON-BLOCKING findings
8. whether Step 9 is safe to freeze