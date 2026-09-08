Implement STEP 7 — FPS ADAPTATION.

IMPORTANT:
- Steps 0–6 are FROZEN.
- Do NOT modify their methodology or contracts.
- Do NOT implement Step 8+ behavior.
- Do NOT build bitrate adaptation, edge selection, resource allocation, or the final decision engine.

Objective:
Create the FPS adaptation/feasibility layer that compares video playback FPS requirements against measured SR processing capability and exposes a machine-readable adaptation signal.

1. INPUTS
Use existing frozen data/interfaces wherever possible:
- source FPS from the existing video/benchmark metadata
- SR latency/throughput from Step 5 benchmark outputs
- actual Edge SR telemetry from Step 6 where appropriate

Do not invent measurements or fabricate missing values.

2. CORE METRICS
Implement/document:
- frame_budget_ms = 1000 / source_fps
- estimated_processing_fps = 1000 / latency_ms
- real_time_ratio = frame_budget_ms / latency_ms

Feasibility:
- feasible if measured SR latency <= frame_budget_ms
- alternatively equivalent to real_time_ratio >= 1

Handle invalid/missing FPS or latency explicitly.

3. FPS ADAPTATION SIGNAL
Create a machine-readable result containing at minimum:
- source_fps
- frame_budget_ms
- measured_latency_ms
- estimated_processing_fps
- real_time_ratio
- realtime_feasible
- model_id
- scale
- device
- input/base representation identity
- measurement provenance

Clearly distinguish:
- measured SR feasibility
- decision eligibility
- end-to-end streaming feasibility

Do NOT treat one benchmark run as statistically decision-eligible if Step 5's eligibility rules reject it.

4. ADAPTATION CLASSIFICATION
Define a small, deterministic classification based on the real-time ratio, for example:
- realtime
- near_realtime
- below_realtime
- severely_below_realtime

Document the exact thresholds used.
Do not choose thresholds arbitrarily without documenting their purpose.

5. RUNTIME INTEGRATION
If Step 6 already exposes SR processing timing, integrate with that telemetry without changing Step 6 contracts.

Do not duplicate SR inference.
Do not create a second inference pipeline.

6. OUTPUT
Produce a machine-readable FPS adaptation result that later Step 8/10 can consume.

Keep the layer decision-oriented but NOT a full decision engine.

7. TESTING
Add tests for:
- 30 FPS with latency below budget
- 30 FPS with latency above budget
- exact budget boundary
- invalid/zero FPS
- invalid/zero latency
- missing measurements
- multiple models/scales/devices
- decision eligibility distinction

If practical, include one REAL local integration test using the existing Step 6 Edge runtime and actual TinySR inference.

8. DOCUMENTATION
Document:
- formulas
- classification thresholds
- input/output schema
- provenance
- relationship to Step 5 measurements
- relationship to Step 6 runtime telemetry
- explicit limitations

STOP after Step 7.
Return:
1. files changed
2. implementation summary
3. tests executed + actual results
4. sample output
5. any BLOCKER / IMPORTANT / NON-BLOCKING findings
6. whether Step 7 is safe to freeze