# Domain Ops Contracts

Статус: foundation

## Contract: ReplaySession

- layer: ops
- purpose: control-plane record for replay execution over canonical events.
- required-fields:
  - `session_id`
  - `started_at`
  - `status`
  - `source_contour`
  - `source_event_count`
- optional-fields:
  - `ended_at`
  - `replayed_event_count`
  - `notes`
- invariants:
  - `source_event_count >= 0`
  - `replayed_event_count >= 0`
  - `ended_at >= started_at` when present.
  - replay session owns replay process metadata, not business payloads.

## Contract: OperatorApproval

- layer: ops
- purpose: explicit founder/operator approval record for risk-sensitive actions.
- required-fields:
  - `approval_id`
  - `action_type`
  - `status`
  - `requested_at`
- invariants:
  - approval records are append-only.
  - execution contracts may reference approvals, but approval state is not embedded into order payload shape by default.

## Contract: KillSwitchState

- layer: ops
- purpose: canonical kill-switch state visible to runtime and UI.
- required-fields:
  - `state_id`
  - `enabled`
  - `changed_at`
- invariants:
  - kill switch is operator state, not broker state.
  - toggling semantics stay outside current executable domain shell until runtime workstream.

## Compatibility notes

- this workstream only requires an executable `ReplaySession` shell.
- `OperatorApproval` and `KillSwitchState` are documented now so later workstreams do not invent incompatible shapes.
