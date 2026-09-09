Build the AdaptiveSR local demonstration prototype using the attached `code.html` interface as the PRIMARY UI/visual reference.

IMPORTANT:
- Do NOT redesign the interface from scratch.
- Preserve the overall visual style, dark theme, layout, typography, cards, sidebar, video workspace, and comparison concept from `code.html`.
- Adapt the existing HTML design into the project's actual application.
- This is a LOCAL DEMONSTRATION PROTOTYPE, not a production deployment.
- Reuse the existing AdaptiveSR Steps 6–11 implementation wherever possible. Do not create a second/fake SR pipeline.

GOAL
The prototype must visibly demonstrate:

User selects/uploads a low-resolution video
        ↓
AdaptiveSR local runtime processes it
        ↓
actual SR model inference
        ↓
actual upscaled output video
        ↓
browser displays Original vs AdaptiveSR output
        ↓
actual runtime/decision information is shown in the sidebar

The most important requirement is that the AFTER video is genuinely produced by the registered AdaptiveSR SR model. Do NOT use ordinary OpenCV resize/bicubic interpolation and label it as AI super-resolution.

==================================================
1. VIDEO INPUT
==================================================

Add a functional local video upload/select control.

After selecting a video:
- read actual video metadata
- display filename
- input resolution
- source FPS
- duration
- input bitrate if available
- generate/use the appropriate local processing endpoint

Do not hardcode values such as 1920x1080, 4K, 59.94 FPS, etc. from the reference HTML.

==================================================
2. ACTUAL ADAPTIVESR PROCESSING
==================================================

Connect the UI to the existing AdaptiveSR runtime.

Use the existing pipeline:

observe
→ Steps 7–9 adaptation signals
→ Step 10 Mamdani fuzzy decision
→ Step 6 SR inference
→ output/delivery
→ telemetry

Do NOT duplicate the decision engine or SR implementation inside the frontend.

The prototype should invoke the existing backend/runtime APIs.

For the first working demo, the system must support the actual registered TinySR model and CUDA when available.

The selected configuration shown in the UI must come from the actual AdaptiveSR decision/runtime result.

==================================================
3. ACTUAL UPSCALED OUTPUT
==================================================

This is a HARD REQUIREMENT.

The backend must:
- decode the input video/chunks
- run the selected SR model on the actual frames
- produce the higher-resolution frames
- encode the resulting frames into an output video
- return a playable local video/output URL

Example:

320×240 input
→ TinySR ×2
→ 640×480 actual output

Do NOT simply resize the original video for the prototype.

The output resolution displayed in the UI must be derived from the actual generated output.

==================================================
4. VIDEO COMPARISON UI
==================================================

Use the attached interface's central comparison workspace as the design reference.

Implement:

ORIGINAL (BEFORE)        |        ADAPTIVESR (AFTER)

The original panel plays the uploaded/source video.

The AdaptiveSR panel plays the ACTUAL generated SR output.

Keep the synchronized comparison concept from the reference.

Implement a functional comparison slider/divider if practical.

Both videos should remain synchronized as closely as possible for the local demo.

If synchronized playback becomes unreliable, prioritize correct actual video playback over visual polish.

==================================================
5. RIGHT SIDEBAR
==================================================

Keep the reference sidebar structure, but replace fictional/demo values with actual AdaptiveSR values.

Sections:

A. VIDEO INFO
- filename
- input resolution
- source FPS
- duration
- output resolution
- output FPS where applicable

B. ADAPTIVE DECISION
Show the actual selected configuration:

- Delivery mode: SR / Native / Execution Failed
- Model
- Scale
- Device
- Edge
- Decision status
- Decision eligibility

Do NOT present manually selected model cards as though they are the adaptive decision.

The UI should communicate:

"AdaptiveSR selected this configuration"

rather than:

"User selected this configuration"

C. ENGINE TELEMETRY

Display only telemetry actually available from the runtime.

Useful fields:
- processing latency
- estimated/inference FPS where valid
- GPU utilization
- VRAM usage
- CPU utilization
- buffer state if available
- network RTT/bandwidth where available

If a metric is unavailable, show `N/A` rather than inventing a value.

Do not use fake values such as:
"36.7 FPS", "42% GPU", "2.6s buffer", etc.

==================================================
6. PROCESSING UX
==================================================

Before processing:
- show an idle/upload state

During processing:
- show processing status/progress
- clearly indicate that AdaptiveSR is processing the video
- disable conflicting actions where necessary

After successful processing:
- automatically load/play the generated SR output
- populate actual metadata and telemetry
- show the selected AdaptiveSR configuration

On failure:
- show a clear error state
- do not silently fall back to another SR model/device
- preserve the distinction between execution failure and native fallback according to Step 11 semantics

==================================================
7. CONTROLS

Make the important controls functional:

- Upload/select video
- Start / Apply AdaptiveSR
- Play/pause
- Timeline/scrubbing
- Comparison slider
- Reset/re-run

Export can initially download the actual generated SR output.

Do not implement decorative controls that appear functional but do nothing.

==================================================
8. LOCAL-ONLY SCOPE

This prototype is intended to run on the development machine.

Do NOT add:
- authentication
- user accounts
- database
- cloud deployment
- CDN
- unnecessary production infrastructure

Use the existing local AdaptiveSR backend/runtime.

The browser frontend may run on localhost and communicate with the existing local backend.

==================================================
9. RESEARCH INTEGRITY

Do not fabricate:
- PSNR
- SSIM
- VMAF
- FPS
- latency
- GPU utilization
- bitrate
- network measurements
- fuzzy scores
- edge information

If the existing runtime does not provide a value during the demo, display `N/A` or `Not available`.

Do not claim TensorRT, 4K, 60 FPS, HDR, motion interpolation, or other capabilities unless they are actually implemented in AdaptiveSR.

The reference HTML contains illustrative UI values/features. Treat those as visual inspiration only, NOT as factual system capabilities.

==================================================
10. DEMO ACCEPTANCE TEST

The prototype is considered functional only when this complete flow works locally:

1. Launch local AdaptiveSR application.
2. Upload/select an actual low-resolution video.
3. UI displays actual input metadata.
4. Start AdaptiveSR.
5. Existing AdaptiveSR runtime performs actual inference.
6. A genuine SR model upscales the video frames.
7. Generated output video is saved/served locally.
8. Browser displays original video.
9. Browser displays the genuinely upscaled AdaptiveSR output.
10. Input/output resolutions are visibly different and correct.
11. Sidebar displays the actual selected model/scale/device.
12. Runtime telemetry shown in UI corresponds to actual execution.
13. No fabricated metrics are displayed.
14. Re-running the same demo does not create conflicting/stale output state.

PRIORITY ORDER:

1. Actual SR upscaling working
2. Actual output video playback
3. Correct AdaptiveSR integration
4. Correct metadata/telemetry
5. Original-vs-output comparison
6. UI polish

Do not sacrifice the actual SR pipeline just to make the interface look complete.

After implementation, run the relevant tests and provide the actual terminal output and a concise summary of:
- files changed
- backend/API used
- how the SR output is generated
- how the frontend loads the output
- tests executed
- any remaining limitations