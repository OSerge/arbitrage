# Agent Operating System Contracts

Статус: foundation

## Canonical conventions

- id-format: stable string identifiers owned by the producing layer.
- time-format: UTC timestamps; Python surface uses timezone-aware `datetime`.
- numeric-format: price, quantity, pnl, and conviction are represented as decimal-compatible values.
- contour-enum: `paper | broker-test | broker-live | replay`.
- source-enum: `research | portfolio | execution | adapter | replay | ops`.
- serialization-rule: every executable contract must round-trip to plain Python primitives suitable for JSON encoding.

## Lineage fields

- `research_run_id`: links a research output to the run that produced it.
- `signal_intent_id`: links a portfolio target back to research intent.
- `target_position_id`: links execution lifecycle back to portfolio target.
- `order_id`: canonical order lifecycle identifier, stable across paper/replay/adapter mapping.
- `session_id`: operator or replay session identifier.

## Distinction rules

- `SignalIntent` expresses research conviction and direction only.
- `TargetPosition` expresses desired portfolio state only.
- `OrderEvent` and `FillEvent` express execution lifecycle only.
- `PositionSnapshot` and `AccountSummary` express observed state, not desired state.
- `ReplaySession` describes replay control-plane state, not broker state.

## Envelope semantics

Canonical execution-event records should be preservable with the following top-level fields:

- `event_id`
- `event_type`
- `occurred_at`
- `source`
- `contour`
- `payload`

The payload may evolve additively, but the top-level meaning of these fields is frozen for MVP.

## Compatibility

- allowed-by-default: additive fields, additive enum values, doc clarifications.
- approval-required: field removal, field rename, semantic reinterpretation, boundary movement, hidden live-only branches.
