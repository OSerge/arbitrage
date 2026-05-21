---
name: alor-adapter-replay-validation
description: Validate Alor adapter work, mapper outputs, and replay parity in the statarb MVP. Use when the task mentions Alor, apidev, oauthdev, ws, cws, token rotation, mapper, order or trade events, positions, account state, replay, smoke-checks, or test-contour validation.
---

# Alor Adapter Replay Validation

## Safety Defaults

- Default to the `Alor` test contour.
- Default to read-only validation.
- Do not run `cws` command-path checks or anything live-like without explicit approval.

## Workflow

1. Confirm the intended contour and approval level.
2. Gather the validation source:
   - fixture;
   - recorded payload;
   - safe read-only smoke-check output.
3. Validate transport assumptions:
   - separate auth handling for HTTP, `ws`, and `cws`;
   - token lifetime and refresh behavior;
   - reconnect and resubscribe expectations;
   - unique `guid` discipline for command flows.
4. Validate mapper output against canonical internal events:
   - order event;
   - fill event;
   - position snapshot;
   - account summary.
5. Check replay compatibility:
   - event shape is stable enough for replay;
   - required fields are present;
   - broker payload details are not leaking into the canonical contract unnecessarily.
6. Record caveats and anything intentionally not executed.

## Handoff Checklist

- contour used;
- read-only versus command path;
- payload source;
- mapping verdict;
- replay parity verdict;
- open approval gates.

Stop immediately if validation would place an order, needs live credentials, or weakens the safe/test-only posture.
