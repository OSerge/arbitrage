# Ops Read-Model Contract

- Статус: additive foundation
- Дата: 2026-05-23

## Назначение

File-backed ops read-model для локального operator workflow поверх canonical JSONL event journal. Не требует UI, HTTP API или Postgres; служит backend contract для future internal UI и post-factum inspection.

## Change type

- additive

## Affected artifacts

- `docs/superpowers/contracts/read-model-ops.md`
- `docs/superpowers/contracts/README.md`
- `statarb/runtime/read_model.py`
- `docs/superpowers/runbooks/paper-trading-day.md`

## Compatibility

- не меняет domain event shape;
- читает только canonical journal records (`order_event`, `fill_event`, `position_snapshot`, `account_summary`);
- UI/API layers могут строиться поверх `OpsReadModel.to_dict()` без доступа к paper engine.

## Required approval

- none

## Contract: OpsReadModel

- layer: ops/read-model
- purpose: aggregated operator-facing snapshot из одного JSONL journal.
- required-fields:
  - `journal_path`
  - `total_events`
  - `event_counts`
  - `source_contours`
  - `position_symbols`
  - `open_order_ids`
- optional-fields:
  - `latest_account_summary`
- invariants:
  - `total_events == sum(event_counts.values())`;
  - `position_symbols` derived from latest `position_snapshot` per instrument symbol;
  - `open_order_ids` содержит order ids, чей последний `order_event.status` не terminal (`filled`, `canceled`, `rejected`);
  - read-model не мутирует journal и не выполняет replay.

## Operator command

```bash
uv run python -m statarb.runtime.read_model inspect --journal-path artifacts/historical-smoke/<day>.jsonl
```

## Non-goals (MVP)

- multi-journal aggregation;
- risk guardrails state;
- live broker subscriptions;
- UI rendering.
