Implement STEP 8 — BITRATE / QUALITY ADAPTATION.

IMPORTANT:
- Steps 0–7 are FROZEN.
- Do NOT modify their contracts or methodology.
- Do NOT implement Step 9+ behavior.
- Do NOT implement edge selection, resource allocation, or the final Adaptive Decision Engine.
- Do NOT fabricate quality/bitrate measurements.

OBJECTIVE:
Build the Bitrate/Quality Adaptation Layer that measures the bandwidth-vs-quality tradeoff of available video representations and SR-enhanced representations, producing a machine-readable signal for later decision-making.

SUBTOPICS TO COVER:

8.1 REPRESENTATION / BITRATE MODEL
Use the existing frozen representation/manifest schema.
Capture where available:
- representation_id
- base resolution
- bitrate
- chunk size
- source FPS
- video/chunk identity
- provenance

Do not alter existing representation definitions.

8.2 BANDWIDTH SAVINGS
Implement:
bandwidth_saving_percent =
((reference_bitrate - candidate_bitrate) / reference_bitrate) * 100

Handle zero/missing bitrate explicitly.
Do not assume lower resolution automatically means lower bitrate.

8.3 QUALITY INTEGRATION
Consume existing Step 5.6 quality outputs:
- PSNR
- SSIM
- VMAF only when genuinely measured

Preserve:
- model inference vs bicubic simulation
- metric availability/unavailability
- measurement provenance

Never fabricate missing metrics.

8.4 NATIVE vs SR QUALITY
Represent the relationship:

base representation
→ SR model + scale
→ enhanced output
→ measured quality

Preserve:
- base_representation_id
- target/output resolution
- model_id
- scale
- device
- quality metrics
- quality provenance

Do not claim SR output is equivalent to a native higher-resolution representation unless measurements support it.

8.5 QUALITY CONSTRAINT
Expose whether a candidate has sufficient quality evidence for later decision-making.

Do NOT invent a universal quality threshold unless justified by existing project methodology.
If a threshold is needed for implementation, document it clearly as a configurable policy parameter rather than a scientific conclusion.

8.6 BANDWIDTH–QUALITY TRADEOFF SIGNAL
Create a machine-readable result containing at minimum:
- reference_representation_id
- candidate/base_representation_id
- candidate_bitrate
- bitrate_saving_percent
- base_resolution
- target_resolution
- model_id
- scale
- PSNR/SSIM/VMAF when available
- quality_evaluable
- measurement_provenance
- decision_eligible
- warnings

Do NOT create a final global utility score yet.

8.7 ADAPTATION OUTPUT
Expose the Step 8 signal for later Steps 9/10.

Clearly distinguish:
- measured bandwidth saving
- measured visual quality
- quality availability
- decision eligibility

Step 8 must NOT select the final representation/model/edge.

8.8 VALIDATION
Add tests for:
- bitrate saving calculation
- equal bitrate
- candidate bitrate > reference
- zero/missing bitrate
- available quality metrics
- unavailable metrics
- genuine model-inference provenance
- bicubic simulation provenance
- base/target representation identity
- multiple representations
- missing quality measurements
- decision eligibility distinction

If practical, add ONE real local integration test using an existing video representation/chunk and the Step 6 SR runtime.

DOCUMENTATION:
Document formulas, schemas, provenance, missing-data behavior, eligibility semantics, and relationship to Step 7.

Explicitly document:
LOWER BITRATE + SR DOES NOT AUTOMATICALLY MEAN EQUIVALENT QUALITY.

Do not modify Steps 0–7.

STOP after Step 8.

Return:
1. files changed
2. implementation summary
3. exact formulas/schema
4. tests executed + actual terminal output/results
5. sample machine-readable output
6. BLOCKER / IMPORTANT / NON-BLOCKING findings
7. whether Step 8 is safe to freeze