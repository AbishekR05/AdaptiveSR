# Step 5.6 Hardening — Implementation Specification

**Status:** Required fix pass before Step 5.6 can be frozen (verdict: C — FIX BEFORE FREEZE)
**Scope:** Step 5.6 only. Does not modify Steps 0–4 or Step 5.5 (frozen).
**Note on this spec:** The source repository (`adaptive_sr/...`, `tests/...`) is not available in this review environment — only the Step 0–5 implementation docs and the Step 5.6 results report were inspected. Every item below that depends on exact existing field/function names is marked **[VERIFY IN REPO]** and must be confirmed against the actual code before implementation. Nothing here invents a schema that isn't already evidenced in `STEP5_IMPLEMENTATION.md` or the Step 5.6 report.

---

## 1. Bicubic Substitution

**Problem:** `cv2.resize` bicubic output is currently recorded and reported under real model names (`tinysr`, `tinysr_int8`, `real_esrgan`), including scale-specific rows. This is a mislabeled independent variable, not a documented limitation.

**Required changes:**

- Add a mandatory `evaluation_mode` field to every quality record (frame/chunk/clip), with allowed values `"model_inference"` | `"bicubic_simulation"`. No record may omit this field.
- `model_id` may only be set to a value returned by the Step 5.2 adapter registry (`get_adapter`/`list_available_models()`) **when** `evaluation_mode == "model_inference"`. When `evaluation_mode == "bicubic_simulation"`, `model_id` MUST be `"bicubic_baseline"` (or equivalent explicit non-model identifier) — never `tinysr`, `tinysr_int8`, or `real_esrgan`.
- Bicubic-mode runs and model-inference-mode runs are separate output artifacts (or clearly partitioned within one artifact by `evaluation_mode`) — never merged into a single comparison table without the mode column visible.
- **[VERIFY IN REPO]** Locate the current call site that invokes `cv2.resize` in place of adapter `.process()` and confirm it is the only substitution path (report §6 mentions GTX 1650 WDDM paging + tile=400 workaround for `real_esrgan` — confirm whether that workaround was itself sufficient to avoid substitution for at least one model/scale, since if so §4 minimum-inference may already be partially satisfiable).

**Report disclosure:** See §6 (Reporting) — disclosure must appear in the document's opening section, not buried in an infrastructure notes table.

---

## 2. Reference / Ground Truth

**Problem:** The Step 5.6 report never states what HR reference frame PSNR/SSIM was computed against. This must be resolved by inspection, not assumption.

**Required investigation (do before any code change):**

- Inspect the Step 5.1 corpus generator and manifest (`data/benchmarks/sr/videos/`, `_profile.json`, `_manifest.json` per STEP5*IMPLEMENTATION.md §5.1 §8–§11) to determine: does the corpus store a source/HR frame independent of any downscaled LR input, or is the corpus single-resolution (i.e. the "source" video \_is* the only resolution generated)?
- **[VERIFY IN REPO]** Check whether Step 5.6's implementation currently derives its own LR input via downsampling the Layer B clip, or whether it reads an LR representation that doesn't exist yet.

**If the corpus is single-resolution (no LR/HR pair exists):**

- Minimum required fix: generate the LR input for evaluation deterministically from the existing HR clip (e.g. `LR = downsample(HR, factor=1/scale)` using a documented, fixed, non-model method — e.g. `cv2.INTER_AREA`), then the HR clip frame is the reference and `adapter.process(LR, scale)` output is compared against it.
- This LR-generation step must be logged in the record (`lr_generation_method` field) so it is never confused with a "real degraded source."
- Do not invent an alternate reference methodology (e.g. downloading external HR footage) — Layer B corpus as already generated (Step 5.6 report §1) remains the reference source; only the LR/HR pairing mechanism is added if missing.

**Alignment requirement:** Reference frame and model/bicubic output frame must be the same frame index, same chunk, same clip — verified explicitly (§3 below), never assumed by array position.

---

## 3. PSNR / SSIM

**Preserve as-is (already valid per report §2, §7):**

- Y-channel PSNR via BGR→YCbCr conversion, float64 precision.
- Y-channel SSIM.
- Divisibility cropping (`apply_divisibility_crop`, already tested per report §7).

**Required additions:**

- **Explicit color space / alignment contract:** Document (in code docstring + this spec) that both PSNR and SSIM operate on the Y channel only, post-crop, at matching `(H, W)`. Add an assertion that raises (not silently resizes) if reference and output dimensions differ after cropping — no implicit `cv2.resize` inside the metric functions to force a match.
- **No silent mismatch handling:** If frame counts differ between reference clip and model/bicubic output for a chunk, raise an explicit error identifying `chunk_id` and the count mismatch — do not truncate or pad silently.
- **SSIM downsampling validation (required, not optional):**
  - Select a representative sample: minimum 1 clip × 1 model (or bicubic) × 1 scale × all frames in 1 chunk (report already computes at 192×108; add a parallel full-resolution run for this sample only — full corpus does not need to be reprocessed at full-res).
  - Compute the deviation: `|SSIM_fullres − SSIM_192x108|` per frame in the sample, report mean/max deviation.
  - Do not retain the current claim ("100x speedup, no meaningful metric distortion") unless this measured deviation is below a stated, justified threshold (e.g. document the actual number — do not pre-select a threshold to force a pass).
  - If deviation exceeds what the report can defend, keep 192×108 only for pipeline-smoke-testing (§6) and use full-resolution SSIM for any record labeled `evaluation_mode = "model_inference"`.

---

## 4. Actual Model Inference (Minimum Requirement)

**Problem:** Zero genuine SR adapter inference occurred in the reviewed report. §9 conclusions about model quality are entirely unsupported.

**Minimum representative inference required before any model-quality claim is permitted:**

- At least **one spatial adapter** already registered in Step 5.2 (`tinysr` is the natural choice — always available, no CUDA dependency per STEP5_IMPLEMENTATION.md §5.2 item 1) run via its adapter `.process()` method — not `cv2.resize` — on:
  - At least 1 clip from the existing Layer B corpus (any motion profile),
  - At least 1 full chunk (all frames in that chunk, not a single sampled frame),
  - At least 1 supported scale for that adapter.
- This satisfies "genuine model-quality evaluation" as a floor. Full cross-model, cross-scale, cross-clip coverage is **optional future work** (§Optional below), not a blocking requirement.
- **Reuse constraint:** Call the existing `BaseSRAdapter` interface (`initialize`, `process`, `close`) from Step 5.2 exactly as Step 5.5's harness does. Do not write a second/parallel inference path. If GPU paging is the blocker for `real_esrgan`, restricting the minimum-inference requirement to CPU-executable `tinysr` avoids re-encountering the GTX 1650 WDDM issue entirely — this is an acceptable scoping choice, not a workaround that needs its own justification beyond "avoids re-triggering the known hardware constraint."
- Do not attempt to fix the GTX 1650 WDDM paging issue itself as part of this hardening pass — that is out of scope (infrastructure/hardware concern, not a Step 5.6 methodology defect).

---

## 5. Step 5.5 Integration — Join Keys

**Required investigation (do before finalizing schema):**

- **[VERIFY IN REPO]** Inspect the actual Step 5.5 result schema (`metadata["trials"]` structure referenced in STEP5_IMPLEMENTATION.md §5.5 §5) and the benchmark manifest schema from Step 5.1 (§8–§9 of STEP5_IMPLEMENTATION.md, which explicitly lists `benchmark_video_id`, `chunk_id`/chunk timeline fields, and separately the model adapter's `model_id`/`scale`/`device` are documented as harness inputs in §5.5 §1–§2).
- Confirm exactly which of the following are present as literal field names in Step 5.5 output vs. which are configuration inputs only:
  - `benchmark_video_id` — documented in Step 5.1 manifest (§5.1 §8); confirm it also appears in Step 5.5 trial records.
  - `chunk_id` — documented in Step 5.1/5.2 chunk association (§5.1 §9); confirm presence in Step 5.5 trial records.
  - `model_id` — Step 5.5 harness selects adapters by ID (§5.5 §1 item 3); confirm this exact string is persisted per trial, not just used at call time.
  - `scale` — confirm persisted per trial (used in output validation, §5.5 §1 item 7).
  - `device` — Step 5.5 distinguishes CPU/CUDA execution paths (§5.5 §2); confirm the literal device identifier persisted (e.g. `"cpu"` / `"cuda:0"`) matches Step 5.4's `device_id` convention.

**Required change:**

- Step 5.6 quality records (frame/chunk/clip) MUST carry the same five identifiers using identical field names and identical value formats as confirmed present in Step 5.5's schema. If any of the five do not exist in Step 5.5's actual output, do not fabricate them in Step 5.6 — instead add them to Step 5.6's own records only, and document in this spec's implementation notes that Step 5.5 does not currently expose that key (this is a Step 5.5 gap to flag, not silently patch into Step 5.5 itself — Step 5.5 remains frozen and unmodified).
- Add `evaluation_mode` (§1) as a sixth Step-5.6-only key; it has no Step 5.5 equivalent and is not required to join, only to disambiguate within Step 5.6's own output.

---

## 6. Reporting

**Structural separation required:**

- The report must contain two clearly headed sections:
  1. **"Pipeline / Infrastructure Validation"** — covers all `evaluation_mode = "bicubic_simulation"` results. May discuss cropping correctness, PSNR/SSIM computation correctness, VMAF fallback behavior, schema completeness. **Must not** contain any sentence comparing named SR models' quality.
  2. **"Model Quality Results"** — covers only `evaluation_mode = "model_inference"` results (§4 minimum, or more if performed). Only this section may make claims about a named model's PSNR/SSIM.
- Move the bicubic-substitution disclosure to the top of the document (executive summary / abstract level), stating plainly: _"Sections below labeled 'Pipeline / Infrastructure Validation' use bicubic interpolation, not SR model inference, and must not be read as model quality results."_

**Claims to remove (not reword, remove) from any section not backed by §4 real inference:**

- Model quality rankings (§9.1 of the reviewed report).
- INT8 vs. FP32 quality parity claims (§9.2).
- Real-ESRGAN suitability, hallucination-behavior, or motion-sensitivity claims (§9.3, §5.3 of the reviewed report).
- Any cross-model comparison table that mixes bicubic and inference results without the `evaluation_mode` column visibly present in the same table.

**Preserve unchanged:**

- VMAF `null` / `vmaf_unavailable: true` handling — this was correctly implemented (no fabricated fallback) and must not be altered by this hardening pass.

---

## 7. Tests

Add to `tests/test_quality_evaluation.py` (extending, not replacing, the existing 5 tests):

1. **`test_bicubic_cannot_be_labeled_as_model`** — asserts that calling the bicubic-simulation code path with `model_id` set to a registered adapter ID (e.g. `"tinysr"`) raises/rejects; only `"bicubic_baseline"` (or equivalent) is accepted when `evaluation_mode="bicubic_simulation"`.
2. **`test_evaluation_mode_field_required`** — asserts every emitted frame/chunk/clip record contains a valid `evaluation_mode` value; schema validation fails if absent.
3. **`test_reference_output_dimension_mismatch_raises`** — feeds a reference frame and output frame with mismatched post-crop dimensions; asserts an explicit error is raised, not a silent resize.
4. **`test_frame_count_mismatch_raises`** — reference clip and output have different frame counts for a chunk; asserts explicit error identifying `chunk_id`.
5. **`test_join_keys_present`** — asserts every quality record contains all confirmed-present join keys from §5 (exact key names per repo verification) with non-null values.
6. **`test_ssim_downsampling_validation_sample`** — runs the full-res vs. 192×108 SSIM comparison on the representative sample (§3) and asserts the measured deviation is captured/logged as a numeric result (not that it passes an arbitrary threshold — the test validates the _measurement exists and is recorded_, not a target it must hit).
7. **`test_no_unsupported_conclusions_in_bicubic_section_metadata`** — if report generation is code-driven (templated), assert that no record with `evaluation_mode="bicubic_simulation"` is included in any aggregate/table tagged as a "model quality" output artifact.
8. **`test_minimum_real_inference_smoke`** — integration test running the real `tinysr` adapter (§4) on one chunk of one Layer B clip, confirming `evaluation_mode="model_inference"`, `model_id="tinysr"`, and valid PSNR/SSIM values are produced and persisted with all join keys.

---

## 8. Acceptance Criteria (for re-freeze review)

Step 5.6 may be resubmitted for freeze review only when **all** of the following hold:

1. No quality record anywhere in the output artifacts has `model_id` set to a registered adapter name while `evaluation_mode="bicubic_simulation"` (or lacks `evaluation_mode` entirely).
2. At least one genuine adapter-inference run (§4 minimum) exists, with results clearly separated in the report under "Model Quality Results."
3. The reference/ground-truth mechanism is explicitly documented (§2), including whether LR was generated from HR and by what fixed method.
4. PSNR/SSIM alignment is enforced by explicit errors on mismatch, not silent resize/crop (§3).
5. The SSIM downsampling deviation has been measured on a representative sample and the measured number (not an assumed one) appears in the report; the "no meaningful distortion" claim is either substantiated by this number or removed.
6. Step 5.6 records carry the verified-present Step 5.5 join keys (§5) under identical field names; any key confirmed absent from Step 5.5 is documented as a known gap, not fabricated.
7. The report separates infrastructure/pipeline validation from model-quality results in distinct sections, with the bicubic-limitation disclosure in the executive summary.
8. All removed claims (§6 list) are absent from the resubmitted report — not reworded/hedged, fully removed unless backed by §4 real inference.
9. All 8 tests in §7 exist and pass, in addition to the original 5 passing tests from the reviewed report.
10. VMAF null-handling behavior is unchanged and still verified by its existing test.

---

## Optional / Future Work (explicitly NOT required for this freeze)

- Full cross-model × cross-scale × cross-clip real inference coverage (beyond the §4 minimum).
- Resolving the GTX 1650 WDDM paging issue to enable `real_esrgan` at 4K natively.
- Installing `libvmaf`-enabled `ffmpeg` for real VMAF scoring.
- LPIPS or other perceptual metrics.
- Real (non-synthetic) Layer B reference clips beyond the current procedural-noise corpus.

These are not blocking for Step 5.6 freeze and must not be pulled into this hardening pass.
