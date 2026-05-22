# Документация проекта

Этот каталог содержит архитектурные, governance- и операционные артефакты вокруг текущего узкого `agent-operated` MVP для платформы статистического арбитража.

## Актуальный рабочий контур

Сейчас активным контуром проекта считается:

- узкий `agent-operated` MVP;
- `paper-first` runtime;
- controlled `Alor` `test/live` contour, где практический текущий фокус ограничен `test` и read-only шагами;
- `core/` как legacy research/prototype contour;
- `statarb/` как активный platform shell для нового MVP-кода, контрактов-ориентированного runtime и адаптеров.

`Phase 1` и `Phase 2` implementation plan в основном закрыты. Текущий ближайший operational focus: ручной `Alor` `test` contour read-only smoke, затем доработка reference/enrichment wiring, canonical mapping и первых operator/read-model workflows.

## Что читать в первую очередь

- `docs/superpowers/specs/2026-05-21-agent-operated-mvp-design.md` — утвержденный design spec для узкого `agent-operated` MVP: `paper-first platform + controlled Alor test/live contour`.
- `docs/superpowers/plans/2026-05-21-agent-operated-mvp-implementation-plan.md` — активный phased plan/handoff документ. Здесь фиксируются `done / in progress / next` по текущей ветке.
- `docs/superpowers/project-map.md` — карта репозитория и граница `core/` vs `statarb/`.
- `docs/superpowers/approval-matrix.md` — что агент может менять автономно и какие шаги требуют founder approval.
- `docs/superpowers/runbooks/alor-test-contour.md` — operational runbook по `AlorAPI` test contour, включая правило, где допустим public-only partial smoke без `portfolio`, а где `ALOR_TEST_PORTFOLIO` обязателен.
- `docs/superpowers/contracts/README.md` — индекс доменных и operational контрактов MVP.
- `docs/superpowers/contracts/alor-market-data.md` — contract doc по open/public `AlorAPI` market-data surfaces и raw/normalized storage design.

## Статус по ветке

На текущем этапе уже зафиксированы:

- governance foundation и базовый `Agent Operating System`;
- domain contracts и их executable shell;
- historical `paper/replay` contour;
- `Alor` public data path: fetch -> raw -> normalized `Parquet` -> manifest;
- safe read-only `Alor` adapter/runtime slice, включая корректный refresh `token` shape;
- partial public-only smoke path, который может работать без `ALOR_TEST_PORTFOLIO`.

Остаются ближайшими шагами:

1. вручную прогнать `Alor` `test` contour read-only smoke через локальный `.env`;
2. отдельно пройти portfolio-scoped read-only smoke там, где нужны `/md/v2/Clients/...` и `ws_portfolio` surfaces;
3. связать `Securities/availableBoards` enrichment с `fetch/bootstrap`;
4. расширить canonical mapping и первые operator/read-model workflows;
5. только после этого возвращаться к более широким `.env`/profile/live-extension шагам и UI slice.

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
