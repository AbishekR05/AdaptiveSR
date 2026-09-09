STEP 11 — END-TO-END REAL-TIME ADAPTIVESR IMPLEMENTATION

Implement Step 11 only.

IMPORTANT:
- Steps 0–10 are FROZEN. Do not modify their contracts, formulas, schemas, or behavior.
- Read the existing repository before coding. Reuse existing Step 6–10 interfaces.
- Do not redesign previous steps.
- Do not fabricate measurements.
- Do not silently fall back from CUDA to CPU.
- Do not implement Step 12 evaluation/baselines yet.

OBJECTIVE
Build the end-to-end closed-loop AdaptiveSR streaming runtime that operates chunk-by-chunk:

observe → generate adaptation signals → fuzzy decision → execute selected configuration → update buffer/state → next chunk.

1. END-TO-END ORCHESTRATOR

Create an appropriate Step 11 runtime/orchestrator module using the existing project architecture.

For every logical chunk:

1. Identify video_id + chunk_id.
2. Obtain current network/resource/SR measurements using existing infrastructure.
3. Generate/consume the existing Step 7 FPSAdaptationSignal.
4. Generate/consume the existing Step 8 BitrateAdaptationSignal.
5. Generate/consume the existing Step 9 EdgeResourceSignal.
6. Pass those signals to Step 10 FuzzyAdaptiveDecisionEngine.
7. If Step 10 returns no suitable candidate, record the decision failure explicitly.
8. If selected, execute the exact selected:
   - edge_id
   - representation
   - model_id
   - scale
   - device
9. Use the existing Step 6 remote SR execution path.
10. Record execution timing/result.
11. Update client playback-buffer state.
12. Move to the next chunk.

Do NOT duplicate Step 6 SR inference logic.

2. CHUNK-LEVEL STATE

Maintain explicit runtime state containing at minimum:

- video_id
- current_chunk_id
- buffer_seconds
- stall_count
- total_stall_duration
- decision_count
- configuration_switch_count
- previous_selected_configuration
- current selected edge/representation/model/scale/device

Do not hardcode chunk duration. Use existing chunk metadata.

3. BUFFER / PLAYBACK MODEL

Reuse the existing project's buffer mathematics.

For every chunk record:

- buffer_before
- download/execution elapsed time
- chunk_duration
- buffer_after
- stall occurrence
- stall duration

If playback buffer reaches/exhausts zero during processing, record a stall.

Do not implement a real browser/VLC player.

4. CONFIGURATION SWITCH TRACKING

Compare the current selected configuration with the previous chunk.

A configuration change must be recorded when any of these change:

- edge_id
- representation_id
- model_id
- scale
- device

Maintain configuration_switch_count.

Do NOT add hysteresis, cooldown, smoothing, or arbitrary stability logic yet. First measure raw decision behavior.

5. PER-CHUNK TELEMETRY

Produce machine-readable records containing:

IDENTITY:
- video_id
- chunk_id
- request_id
- edge_id
- cluster_id

SELECTED CONFIGURATION:
- representation_id
- target_resolution
- model_id
- scale
- device

DECISION:
- decision
- fuzzy_suitability
- suitability_tier
- decision/rejection reason where applicable

TIMING:
- request/start time
- completion time
- download/transfer timing
- SR processing timing
- total chunk completion time

NETWORK:
- measured bandwidth/throughput
- RTT
- bytes received

RESOURCE:
- CPU utilization where applicable
- GPU utilization where applicable
- GPU memory where applicable

BUFFER:
- buffer_before
- buffer_after
- stall_count
- stall_duration

PROVENANCE:
- Step 7 signal provenance
- Step 8 signal provenance
- Step 9 signal provenance
- Step 10 decision provenance

Reuse existing field semantics wherever they already exist. Do not create conflicting duplicate definitions.

6. FAILURE HANDLING

Handle explicitly:

- no suitable fuzzy candidate
- missing required telemetry
- missing chunk
- invalid representation
- invalid SR model
- unavailable requested GPU
- SR execution failure
- edge/request failure

Failures must be recorded in telemetry.

Never silently select another model/device/edge when the selected configuration cannot execute.

7. DYNAMIC CONDITIONS

Provide a controlled test mechanism for changing conditions between chunks.

At minimum support scenarios where:

- network changes from good → degraded/poor → recovery
- edge resource load changes
- selected configuration can change when conditions justify it

Do not fabricate production measurements. Controlled test conditions must be clearly marked as test/injected conditions.

8. TESTS

Add Step 11 tests covering:

A. Single-chunk end-to-end execution.
B. Multi-chunk sequential execution.
C. Step 7/8/9 → Step 10 integration.
D. Selected Step 10 configuration is actually honored.
E. Buffer update correctness.
F. Stall detection.
G. Configuration-switch tracking.
H. Dynamic-condition adaptation.
I. No-suitable-candidate handling.
J. CUDA-unavailable failure handling.
K. SR execution failure handling.
L. Missing telemetry handling.
M. Deterministic/reproducible telemetry structure.
N. Existing Steps 0–10 regression compatibility.

The tests must distinguish mocked/injected conditions from genuine runtime measurements.

9. DOCUMENTATION

Create/update Step 11 implementation documentation containing:

- architecture
- closed-loop sequence
- chunk lifecycle
- runtime state
- telemetry schema
- buffer model
- configuration switching
- failure handling
- test scenarios
- limitations
- exact commands executed
- actual test results

Do not claim tests passed unless the command was actually executed.

10. STRICT SCOPE BOUNDARY

Step 11 does NOT implement:

- final baseline comparison
- heuristic-vs-fuzzy evaluation
- ablation studies
- sensitivity analysis
- statistical significance
- final QoE claims
- final energy-efficiency claims
- Step 12 experimental campaign

Those belong to Step 12.

11. VALIDATION / COMPLETION

Run:

- all new Step 11 tests
- the complete existing regression suite

Report:
- exact command
- exact number of tests passed/failed
- warnings
- files changed
- known limitations

Do not tag or claim Step 11 is frozen automatically.

STOP after implementation + testing + documentation.

Return a concise implementation report with:
1. files changed
2. architecture implemented
3. tests executed + actual results
4. sample machine-readable chunk telemetry
5. known limitations
6. anything requiring audit before freeze