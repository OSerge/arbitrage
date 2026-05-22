# Read-Model API Contract (Backend Freeze)

- Статус: frozen for UI handoff
- Дата: 2026-05-23
- `schema_version`: `1`

## Назначение

Зафиксировать file-backed read-model gateway между backend runtime и будущим internal UI **без** добавления frontend toolchain. UI позже может читать JSON artifacts напрямую или через thin static file server.

## Change type

- additive freeze

## Affected artifacts

- `docs/superpowers/contracts/read-model-api.md`
- `statarb/runtime/read_model_gateway.py`
- `docs/superpowers/runbooks/paper-trading-day.md`

## Compatibility

- не меняет canonical domain events;
- не требует HTTP server для MVP operator workflow;
- breaking changes требуют increment `schema_version` и отдельного approval.

## Required approval

- none for backend freeze;
- frontend toolchain остаётся отдельным gate.

## Export command

```bash
uv run python -m statarb.runtime.read_model_gateway export \
  --journal-path artifacts/historical-smoke/<day>.jsonl \
  --output-dir var/read-models/<day>/
```

## Output files

| file | slice | purpose |
| --- | --- | --- |
| `manifest.json` | bundle | schema version + all slices |
| `ops.json` | ops | journal aggregates, open orders, latest account summary |
| `research.json` | research | lineage ids and source contours |
| `risk.json` | risk | exposure snapshot, open orders, latest equity |

## Manifest shape

Required top-level fields:

- `schema_version`
- `generated_at`
- `ops`
- `research`
- `risk`

## UI consumption rules

- UI must treat files as read-only snapshots;
- UI must not mutate journals or trigger broker connections;
- stale snapshot is acceptable; refresh is operator-driven export;
- degradation of UI/read-model export must not affect paper/replay runtime.

## Non-goals

- REST/WebSocket server in MVP backend freeze;
- live broker read-model;
- Postgres-backed read-model (future prod-like extension).
