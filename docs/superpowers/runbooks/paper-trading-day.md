# Paper trading day runbook

- Статус: рабочая опорная инструкция для MVP
- Дата: 2026-05-22

## Назначение

Этот runbook описывает безопасный operator workflow для локального paper/replay trading day без брокерских соединений и без live enablement. Он покрывает default historical smoke path и служит базой для post-factum разбора торгового дня через canonical event journal.

## Scope и границы

- Разрешен только `paper`/`replay` contour; `Alor test/live` adapter surfaces здесь не вызываются.
- Workflow read-only относительно брокера: никаких order placement, cancel или `cws` command path.
- Результат smoke не означает готовность к live trading и не заменяет отдельный Alor test contour smoke.

## Быстрый старт

Запустить default historical paper/replay day и получить redacted summary:

```bash
uv run python -m statarb.runtime.paper_trading_day
```

Явно указать journal path:

```bash
uv run python -m statarb.runtime.paper_trading_day --journal-path artifacts/historical-smoke/manual-day.jsonl
```

## Что делает helper

1. Собирает default `HistoricalSmokeBundle` из legacy historical bridge (`SRM5`/`SPM5`).
2. Прогоняет target positions через `PaperExecutionEngine` с risk checks и `JsonlEventSink`.
3. Загружает journal обратно и выполняет `RuntimeReplayer`.
4. Печатает summary с:
   - `research_run_id`;
   - `source_pair`;
   - `journal_path`;
   - counts paper/journal/replay events;
   - `replay_status`;
   - `replay_parity` между paper snapshots и replayed snapshots.

Exit code:

- `0` — replay succeeded и parity подтверждена;
- `1` — replay failed или parity не совпала.

## Рекомендуемый operator checklist

1. Убедиться, что `STATARB_PROFILE=local-dev` и live trading не включен.
2. Запустить `python -m statarb.runtime.paper_trading_day`.
3. Сохранить printed summary и путь к JSONL journal как audit artifact.
4. При необходимости повторно прочитать journal через replay loader/tests, не редактируя исходный файл.
5. Для aggregated ops snapshot без replay выполнить:

```bash
uv run python -m statarb.runtime.read_model inspect --journal-path artifacts/historical-smoke/manual-day.jsonl
```

6. Не смешивать paper journal с broker raw payloads: canonical shape должен оставаться source-of-truth для replay.

## Расширенный contour: pair-bootstrap

Если нужен paper/replay day поверх локальных Alor history datasets, используйте enrichment-aware pair bootstrap:

```bash
uv run python -m statarb.data.alor_fetch pair-bootstrap \
  --left-symbol SRM5 \
  --right-symbol SPM5 \
  --exchange MOEX \
  --timeframe 60
```

Перед этим при отсутствии local reference snapshots может понадобиться:

```bash
uv run python -m statarb.data.alor_fetch securities --exchange MOEX --limit 25
uv run python -m statarb.data.alor_fetch available-boards --symbol SRM5 --exchange MOEX
```

Pair-bootstrap по умолчанию также запускает historical smoke и печатает summary с `replay_status`.

## Что проверять в summary

- `replay_status == SUCCEEDED`
- `replay_parity == true`
- `source_event_count == replayed_event_count`
- `journal_path` указывает на созданный JSONL artifact
- `position_symbols` соответствуют ожидаемой паре инструментов сценария

## Связанные документы

- `docs/superpowers/contracts/domain-execution.md`
- `docs/superpowers/contracts/domain-ops.md`
- `docs/superpowers/runbooks/alor-test-contour.md`
- `docs/superpowers/plans/2026-05-21-agent-operated-mvp-implementation-plan.md`

## Источники

- `statarb/runtime/paper_trading_day.py`
- `statarb/runtime/historical.py`
- `statarb/runtime/read_model.py`
- `statarb/runtime/alor_pair_bootstrap.py`
- `tests/runtime/test_paper_trading_day.py`
