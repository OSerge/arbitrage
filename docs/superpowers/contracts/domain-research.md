# Domain Research Contracts

Статус: foundation

## Contract: ResearchRun

- layer: research
- purpose: reproducible record of a research execution that can emit downstream intents.
- required-fields:
  - `run_id`
  - `strategy_id`
  - `dataset_id`
  - `status`
  - `started_at`
- optional-fields:
  - `completed_at`
  - `parameters`
  - `artifact_uris`
  - `signal_count`
- invariants:
  - `run_id`, `strategy_id`, and `dataset_id` are non-empty.
  - `completed_at >= started_at` when present.
  - `signal_count >= 0`.

## Contract: SignalIntent

- layer: research
- purpose: canonical research output before portfolio sizing and execution.
- required-fields:
  - `intent_id`
  - `research_run_id`
  - `instrument`
  - `direction`
  - `conviction`
  - `generated_at`
- optional-fields:
  - `horizon`
  - `thesis`
- invariants:
  - `conviction` is within `[0, 1]`.
  - contract does not contain target quantity, broker ids, or execution status.
  - `direction=flat` is allowed only as an explicit neutral research conclusion.

## Compatibility notes

- `SignalIntent` is upstream of `TargetPosition`; downstream layers may reference it, but must not mutate its meaning.
- portfolio sizing, constraints, and risk decisions are intentionally excluded from this layer.
