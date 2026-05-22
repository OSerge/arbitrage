# Local-dev deployment runbook

- Статус: operational
- Profile: `local-dev`

## Назначение

Запуск и recovery MVP runtime на локальной машине разработчика без live enablement и без обязательного Postgres.

## Prerequisites

- Python 3.12+, `uv`
- `.env` из `.env.example` или `ops/env/local-dev.env.example`
- секреты только локально: `ALOR_TEST_REFRESH_TOKEN`, опционально `ALOR_TEST_PORTFOLIO`

## Bootstrap

```bash
cp .env.example .env
# заполнить ALOR_TEST_REFRESH_TOKEN и при необходимости ALOR_TEST_PORTFOLIO
uv sync
```

## Core operator flows

### Paper/replay trading day

```bash
uv run python -m statarb.runtime.paper_trading_day
uv run python -m statarb.runtime.read_model inspect --journal-path artifacts/historical-smoke/<day>.jsonl
uv run python -m statarb.runtime.read_model_gateway export \
  --journal-path artifacts/historical-smoke/<day>.jsonl \
  --output-dir var/local-dev/read-models/<day>/
```

### Alor test contour (read-only)

```bash
uv run python -m statarb.runtime.alor_portfolio_resolve list-jwt --env-file .env
uv run python -m statarb.runtime.alor_portfolio_resolve validate --env-file .env
uv run python -m statarb.runtime.alor_test_read_only smoke --env-file .env --with-canonical-preview
uv run python -m statarb.runtime.alor_test_read_only map-preview --env-file .env
```

## Artifact locations

- `STATARB_ARTIFACTS_DIR` (default `var/local-dev`)
- `STATARB_EVENT_LOG_DIR` (default `var/local-dev/events`)
- historical smoke journals: `artifacts/historical-smoke/`
- exported read-models: `var/local-dev/read-models/`

## Recovery

| symptom | action |
| --- | --- |
| auth refresh fails | проверить `ALOR_TEST_REFRESH_TOKEN`, перевыпустить token в Alor dev cabinet |
| portfolio-scoped smoke fails validation | `list-jwt` -> обновить `ALOR_TEST_PORTFOLIO` -> `validate` |
| replay parity false | пересоздать journal через `paper_trading_day`, не редактировать JSONL вручную |
| partial public-only smoke only | задать `ALOR_TEST_PORTFOLIO` для portfolio-scoped surfaces |

## Explicit non-goals

- live trading (`STATARB_ENABLE_LIVE_TRADING` must stay `false` without approval)
- `cws` command-path smoke
- treating test contour results as live-quality proof

## Related docs

- `docs/superpowers/runbooks/alor-test-contour.md`
- `docs/superpowers/runbooks/paper-trading-day.md`
- `docs/superpowers/contracts/read-model-api.md`
