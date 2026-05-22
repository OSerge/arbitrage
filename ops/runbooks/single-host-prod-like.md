# Single-host prod-like deployment runbook

- Статус: operational
- Profile: `single-host-prod-like`

## Назначение

Описать wiring одного хоста для MVP с теми же contracts/env names, что и `local-dev`, но с prod-like observability и optional Postgres для control-plane state.

## Prerequisites

- один Linux host с Python 3.12+ и `uv`
- env из `ops/env/single-host-prod-like.env.example`
- optional Postgres для control-plane (не обязателен для paper/replay-only day)

## Bootstrap

```bash
cp ops/env/single-host-prod-like.env.example /etc/statarb/statarb.env
# заполнить секреты и DATABASE_URL при использовании Postgres
uv sync
```

## Optional Postgres (control-plane)

Postgres **не обязателен** для paper/replay и file-backed read-models. Подключайте только если нужен control-plane state beyond JSONL artifacts.

Минимальный compose skeleton:

```bash
docker compose -f ops/compose/single-host-prod-like.yml up -d postgres
```

## Core operator flows

Идентичны `local-dev`, но artifacts пишутся в profile-specific dirs:

```bash
export STATARB_PROFILE=single-host-prod-like
uv run python -m statarb.runtime.paper_trading_day
uv run python -m statarb.runtime.read_model_gateway export \
  --journal-path artifacts/historical-smoke/<day>.jsonl \
  --output-dir var/single-host-prod-like/read-models/<day>/
uv run python -m statarb.runtime.alor_test_read_only smoke --env-file /etc/statarb/statarb.env
```

## Artifact locations

- `STATARB_ARTIFACTS_DIR` -> `var/single-host-prod-like`
- `STATARB_EVENT_LOG_DIR` -> `var/single-host-prod-like/events`
- structured logs: `STATARB_STRUCTURED_LOGS=true`

## Recovery

| symptom | action |
| --- | --- |
| process crash mid paper day | rerun `paper_trading_day` with new journal path; inspect prior JSONL read-only |
| Postgres unavailable | continue file-backed ops; defer control-plane writes |
| token rotation needed | update env secret, restart operator job, rerun validate/list-jwt |
| disk growth | archive `artifacts/historical-smoke/` and `var/single-host-prod-like/read-models/` |

## Explicit gates

- live trading remains disabled by default
- frontend UI toolchain requires separate approval
- `cws` command smoke requires separate approval

## Related docs

- `ops/runbooks/local-dev.md`
- `docs/superpowers/plans/2026-05-21-agent-operated-mvp-implementation-plan.md`
