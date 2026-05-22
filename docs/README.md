# Документация проекта

Этот каталог содержит архитектурные, governance- и операционные артефакты вокруг текущего узкого `agent-operated` MVP для платформы статистического арбитража.

## Актуальный рабочий контур

Сейчас активным контуром проекта считается:

- узкий `agent-operated` MVP;
- `paper-first` runtime;
- controlled `Alor` `test/live` contour, где практический текущий фокус ограничен `test` и read-only шагами;
- `core/` как legacy research/prototype contour;
- `statarb/` как активный platform shell для нового MVP-кода, контрактов-ориентированного runtime и адаптеров.

`Phase 1`–`4` и backend slices `Phase 5`/`Phase 6` в implementation plan в основном закрыты. Явный оставшийся gate — **frontend UI toolchain approval** для `apps/internal-ui`.

## Что читать в первую очередь

- `docs/superpowers/specs/2026-05-21-agent-operated-mvp-design.md` — утвержденный design spec для узкого `agent-operated` MVP: `paper-first platform + controlled Alor test/live contour`.
- `docs/superpowers/plans/2026-05-21-agent-operated-mvp-implementation-plan.md` — активный phased plan/handoff документ. Здесь фиксируются `done / in progress / next` по текущей ветке.
- `docs/superpowers/project-map.md` — карта репозитория и граница `core/` vs `statarb/`.
- `docs/superpowers/approval-matrix.md` — что агент может менять автономно и какие шаги требуют founder approval.
- `docs/superpowers/runbooks/alor-test-contour.md` — operational runbook по `AlorAPI` test contour, включая smoke execution log и canonical preview commands.
- `docs/superpowers/runbooks/paper-trading-day.md` — operator workflow для локального historical paper/replay trading day без брокерских соединений.
- `ops/runbooks/local-dev.md` и `ops/runbooks/single-host-prod-like.md` — deployment/recovery runbooks для двух профилей.
- `docs/superpowers/contracts/README.md` — индекс доменных и operational контрактов MVP.
- `docs/superpowers/contracts/read-model-api.md` — frozen backend read-model JSON gateway для future UI.

## Статус по ветке

На текущем этапе уже зафиксированы:

- governance foundation и базовый `Agent Operating System`;
- domain contracts и их executable shell;
- historical `paper/replay` contour;
- `Alor` public data path и safe read-only adapter/runtime slice;
- portfolio-scoped read-only smoke (`D00014`) и canonical mapper preview (`map-preview`);
- file-backed ops/research/risk read-model gateway (`read_model_gateway export`);
- operator workflows для paper/replay и test contour;
- deployment profile runbooks и env examples.

Остаётся explicit approval gate:

1. frontend toolchain + `apps/internal-ui` shell (Phase 5 screens);
2. любой live enablement и `cws` command-path smoke;
3. optional Postgres как обязательная prod-like dependency — только по отдельному approval.

## Долгоживущая архитектура и более широкий контекст

Следующие документы остаются верхнеуровневыми ограничениями и траекторией развития, но не являются текущим execution backlog по ветке:

- `docs/adr/0001-separate-research-and-live-runtime.md`
- `docs/adr/0002-build-core-execution-risk-simulation.md`
- `docs/adr/0003-parquet-clickhouse-postgres-foundation.md`
- `docs/adr/0004-python-first-mvp-runtime.md`
- `docs/target_platform_architecture.md`
- `docs/platform_roadmap.md`
- `docs/frontend_ui_architecture.md`
- `docs/russian_market_integrations.md`
- `docs/project_structure.md`
- `docs/research_analysis_and_improvements.md`

Практическое правило: для текущей реализации сначала смотреть в `design spec`, затем в активный `implementation plan`, а более широкие архитектурные документы использовать как authority layer и ориентир следующей эволюции.
