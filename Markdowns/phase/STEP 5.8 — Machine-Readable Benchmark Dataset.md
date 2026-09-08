Implement STEP 5.8 — Machine-Readable Benchmark Dataset.

Step 0–5.7 are frozen. Do not modify their behavior or start Step 5.9.

Goal:
Create a unified, machine-readable benchmark dataset from the existing empirical outputs of:

- Step 5.5 — benchmark/inference measurements
- Step 5.6 — PSNR/SSIM/VMAF quality evaluation
- Step 5.7 — FPS/real-time feasibility analysis

Instructions:

1. Inspect the existing Step 5.5–5.7 schemas, result files, identifiers, and project conventions before implementing anything.
2. Create ONE canonical Step 5.8 dataset/schema that preserves the original measurements and their provenance.
3. Join records using the existing authoritative identifiers. Do not invent new join semantics if existing identifiers already support the relationship.
4. Preserve important fields such as:
   - model_id
   - scale
   - device
   - input/video/chunk identifiers where available
   - latency
   - FPS
   - source_fps
   - frame_budget
   - real-time feasibility
   - PSNR
   - SSIM
   - VMAF / vmaf_unavailable
   - CPU/GPU/resource measurements where available
   - decision eligibility
   - session information
5. Missing measurements must remain explicitly missing/null/unavailable. NEVER fabricate or estimate values.
6. Preserve provenance so every unified record can be traced back to its source benchmark/quality/feasibility result.
7. Do not overwrite raw Step 5.5, 5.6, or 5.7 results.
8. Produce the machine-readable dataset using the project's existing preferred format/conventions (JSON/CSV as appropriate).
9. Add focused validation/tests for:
   - schema validity
   - correct joins
   - duplicate/conflicting records
   - missing data handling
   - provenance preservation
   - compatibility with existing Step 5 outputs
10. Run the relevant Step 5.8 tests and the full regression suite.

Important:
Step 5.8 is ONLY a data consolidation/representation layer.

Do NOT implement:
- adaptive model selection
- bitrate adaptation
- FPS adaptation
- edge selection
- resource allocation
- scheduling
- online learning
- decision engine
- Step 5.9

Create/update concise documentation for Step 5.8 explaining the schema, provenance, joins, missing-data policy, outputs, and validation.

STOP after Step 5.8.

Final response:
- files created/modified
- canonical dataset location
- schema summary
- tests run + results
- regression result
- any data coverage limitations
- confirmation that Steps 0–5.7 remain unchanged