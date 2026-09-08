IMPLEMENT STEP 6 — REMOTE SR INFERENCE

Steps 0–5.9 are FROZEN. Do not modify their behavior.
Do not implement Steps 7+.

Goal:
Integrate the existing SR inference subsystem into the existing
Cloud → Edge → Client runtime so that SR can actually execute at
the Edge on a requested video chunk and return the enhanced result.

First inspect the existing Step 0–4 service architecture, chunk/
representation APIs, and Step 5 SR adapter/benchmark interfaces.
Reuse them instead of creating parallel implementations.

Required flow:

CLIENT
  ↓ SR request / chunk request
EDGE
  ↓ obtain requested source representation/chunk
SR INFERENCE
  ↓
enhanced chunk
  ↓
CLIENT

Requirements:

1. SR execution must occur at the Edge, not the Client.
2. Reuse the existing Step 5 SR adapter registry/interface.
3. Support the currently registered SR models without creating
   new models.
4. Preserve:
   - request_id
   - chunk_id
   - representation_id
   - model_id
   - scale
   - source/output resolution metadata
5. The Edge must explicitly report whether SR was:
   - requested
   - successfully executed
   - failed
6. Do NOT silently fall back from GPU to CPU.
7. Do NOT implement model selection yet.
   The requested model/configuration may be supplied explicitly.
8. Do NOT implement adaptive bitrate, adaptive FPS, scheduling,
   edge selection, resource allocation, or the decision engine.
9. Preserve existing cache HIT/MISS and Cloud ↔ Edge network
   semantics.
10. Do not fake network transfer or SR processing.
    The Edge must actually execute the existing SR adapter.
11. Handle inference failures cleanly without corrupting the
    existing streaming/chunk pipeline.
12. Keep the implementation compatible with the current Windows
    development environment.

Add focused tests covering:

- SR request reaches Edge.
- Correct model adapter is invoked.
- Actual SR output is produced.
- Input/output dimensions are correct.
- Request/chunk/representation identity is preserved.
- GPU/CPU device configuration is respected.
- SR failure is reported correctly.
- Existing non-SR behavior still works.
- Existing full regression suite remains passing.

Create concise Step 6 implementation documentation.

Do NOT benchmark large model matrices here.
Do NOT redesign the Step 5 benchmark system.
Do NOT start Step 7.

Run the focused Step 6 tests and full regression suite.

Final response:
- files created/modified
- runtime flow implemented
- SR models supported
- example request/response
- tests + actual results
- regression result
- limitations
- confirmation that Steps 0–5.9 remain frozen

STOP after Step 6.