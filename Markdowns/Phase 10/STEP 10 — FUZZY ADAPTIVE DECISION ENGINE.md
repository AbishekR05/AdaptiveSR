Implement STEP 10 — FUZZY ADAPTIVE DECISION ENGINE.

IMPORTANT:
- Steps 0–9 are FROZEN.
- Do NOT modify their contracts or methodology.
- Do NOT implement Step 11+ behavior.
- This is the first step allowed to make final adaptive candidate decisions.
- Do not fabricate measurements.
- Do not introduce arbitrary weighted utility coefficients.

OBJECTIVE:

Build a Mamdani-style fuzzy decision engine that consumes the validated signals from Steps 7–9 and selects the most suitable feasible SR configuration.

10.1 CANDIDATE CONSTRUCTION

Consume existing signals from:
- Step 7 FPSAdaptationSignal
- Step 8 BitrateAdaptationSignal
- Step 9 EdgeResourceSignal

Construct candidate configurations preserving:
- base representation
- target resolution
- model_id
- scale
- device
- edge_id

Do not invent candidates outside the supplied configuration space.

10.2 HARD FEASIBILITY GATE

Before fuzzy inference:
- reject candidates marked resource-infeasible
- reject unsupported model/device combinations
- reject candidates with explicit hard safety violations

Hard infeasibility must NOT be overridden by fuzzy scoring.

Preserve reasons for rejection.

10.3 FUZZY INPUTS

Implement fuzzy variables for:

A. real_time_ratio
  - poor
  - moderate
  - good

B. bandwidth_saving_percent
  - low
  - medium
  - high

C. quality suitability
  - poor
  - acceptable
  - good

D. Edge resource condition
  - constrained
  - moderate
  - available

E. network condition
  - poor
  - moderate
  - good

Do not invent scientific thresholds silently.

Membership-function boundaries must be explicit, configurable, documented, and justified from existing project measurements or clearly identified engineering assumptions.

If a quality metric is unavailable, do not fabricate a quality value.

10.4 QUALITY INPUT

Do not blindly combine PSNR, SSIM and VMAF using arbitrary weights.

Define a transparent quality-suitability policy based on available valid metrics and preserve:
- metric availability
- evaluation mode
- provenance
- decision eligibility

If sufficient quality evidence is unavailable, represent that explicitly rather than inventing a score.

10.5 FUZZY OUTPUT

Create:
adaptation_suitability

Use linguistic output levels:
- very_low
- low
- medium
- high
- very_high

Use Mamdani inference and centroid/center-of-area defuzzification.

10.6 RULE BASE

Implement a compact, interpretable rule base covering important tradeoffs.

Examples:

IF real_time_ratio IS good
AND bandwidth_saving IS high
AND quality IS good
AND resource_condition IS available
AND network_condition IS good
THEN suitability IS very_high

IF real_time_ratio IS poor
AND resource_condition IS constrained
THEN suitability IS very_low

IF bandwidth_saving IS high
AND quality IS acceptable
AND real_time_ratio IS good
THEN suitability IS high

Document every rule and its rationale.

Do NOT generate a large arbitrary rule table.

10.7 FINAL DECISION

Evaluate all feasible candidates independently.

Select the candidate with the highest defuzzified suitability.

Add a configurable minimum suitability threshold.

If no candidate reaches the threshold:
return:
decision = "no_suitable_candidate"

Do not force a selection.

Use deterministic tie-breaking and document it.

10.8 MACHINE-READABLE OUTPUT

Create a decision signal containing:
- decision
- selected edge
- selected base representation
- target resolution
- model_id
- scale
- device
- fuzzy suitability
- candidate evaluations
- rejected candidates + reasons
- input signal references/provenance
- rule/inference metadata
- warnings

10.9 MISSING DATA

Do not fabricate missing values.

If a required fuzzy input is unavailable:
- either mark the candidate unevaluable
- or use an explicitly documented missing-data policy

Do not silently treat missing quality/network/resource measurements as good.

10.10 VALIDATION

Add tests for:
- single feasible candidate
- multiple feasible candidates
- hard-infeasible candidate rejected
- highest fuzzy suitability selected
- tie-breaking
- minimum suitability threshold
- no suitable candidate
- missing quality metrics
- missing network telemetry
- missing resource telemetry
- real-time feasible vs decision eligible distinction
- provenance preservation
- deterministic repeated decisions

Include one integration test combining real Step 7/8/9 signals if practical.

Do NOT require a full live streaming system yet.

10.11 BASELINE SUPPORT

Structure the decision output so later Step 12 evaluation can compare AdaptiveSR decisions against non-fuzzy baselines.

Do not implement the final evaluation campaign now.

SCOPE:
Step 10 owns the adaptive decision policy.

It may select:
- representation
- SR model
- scale
- device
- Edge

It must NOT implement:
- end-to-end streaming
- client playback
- final evaluation campaign
- deployment to Azure/Kaggle

DOCUMENTATION:
Document:
- fuzzy variables
- membership functions
- thresholds and justification
- rule base
- Mamdani inference
- centroid defuzzification
- hard feasibility gates
- missing-data policy
- tie-breaking
- minimum suitability threshold
- decision output schema
- limitations
- relationship to Steps 7–9

STOP after Step 10.

Return:
1. files changed
2. implementation summary
3. fuzzy variables/membership functions
4. complete rule base
5. decision algorithm
6. tests executed + actual terminal results
7. sample decision output
8. BLOCKER / IMPORTANT / NON-BLOCKING findings
9. whether Step 10 is safe to freeze