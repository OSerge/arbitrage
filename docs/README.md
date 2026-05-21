# Документация проекта

Этот каталог содержит архитектурные, исследовательские и governance-артефакты вокруг `agent-operated` MVP для платформы статистического арбитража.

## Что читать в первую очередь

- `docs/superpowers/specs/2026-05-21-agent-operated-mvp-design.md` — согласованный design spec для узкого `agent-operated` MVP: `paper-first platform + controlled Alor test/live contour`.
- `docs/superpowers/plans/2026-05-21-agent-operated-mvp-implementation-plan.md` — детальный phased implementation plan для `agent-operated` MVP: `Agent Operating System`, domain contracts, `AlorAPI` test contour, paper/replay runtime, minimal UI read-model slice, `local-dev` и `single-host-prod-like` profiles.
- `docs/superpowers/contracts/alor-market-data.md` — contract doc по open/public `AlorAPI` market-data surfaces: доступные HTTP endpoints, реальные payload shapes и рекомендуемый raw + normalized storage design для MVP.
- `docs/superpowers/runbooks/alor-test-contour.md` — operational runbook по `AlorAPI` test contour: endpoints, auth lifecycle, reconnect/resubscribe discipline, limits и подготовка к первой реальной интеграционной проверке.
- `docs/target_platform_architecture.md` — целевая архитектура всей платформы, от которой MVP сознательно отрезает лишнюю раннюю сложность.
- `docs/platform_roadmap.md` — roadmap развития от foundation к более зрелому execution/risk контуру.
- `docs/frontend_ui_architecture.md` — решение по бесплатному frontend-стеку и роли `Research UI`, `Ops UI`, `Risk UI`.
- `docs/russian_market_integrations.md` — обзор российских интеграций и практических подключений для `MOEX`-контура.

## Дополнительные материалы

- `docs/adr/0001-separate-research-and-live-runtime.md` — разделение исследовательского и торгового runtime.
- `docs/adr/0002-build-core-execution-risk-simulation.md` — почему execution/risk/simulation должны оставаться core-контуром проекта.
- `docs/adr/0003-parquet-clickhouse-postgres-foundation.md` — стартовый storage/control-plane контур.
- `docs/adr/0004-python-first-mvp-runtime.md` — фиксация `Python-first` стратегии для MVP.
- `docs/project_structure.md` — обзор структуры проекта и направлений эволюции репозитория.
- `docs/research_analysis_and_improvements.md` — более широкий анализ исследовательского контура и направлений улучшения.

## Примечание

Сейчас в ветке собраны и долгосрочные архитектурные документы, и узкие `spec/plan`-артефакты для текущего MVP. Реализацию имеет смысл вести, опираясь в первую очередь на `design spec` и `implementation plan`, а более широкие архитектурные документы использовать как верхнеуровневые ограничения и ориентир развития.
