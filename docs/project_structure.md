# Структура проекта системы статистического арбитража

## Обзор

Этот документ описывает актуальную структуру репозитория для текущего узкого `agent-operated` MVP. Главная граница сейчас проходит между:

- `core/` как legacy research/prototype contour;
- `statarb/` как активным platform shell для нового MVP runtime, адаптеров и contracts-aligned кода;
- `docs/superpowers/` как governance-слоем для agent-operated разработки.

## Верхнеуровневая структура

```text
arbitrage/
├── core/                         # Legacy research/prototype contour
├── statarb/                      # Active MVP platform shell
│   ├── adapters/alor/            # Alor endpoints, auth, HTTP/WS/CWS clients, mapper
│   ├── bridges/                  # Runtime/data bridges между срезами
│   ├── config/                   # Profiles и settings
│   ├── data/                     # Fetch, storage, discovery, reference helpers
│   ├── domain/                   # Executable domain contracts
│   └── runtime/                  # Historical, paper, replay, smoke helpers
├── docs/
│   ├── adr/                      # Durable architectural decisions
│   ├── superpowers/              # Specs, plans, contracts, runbooks, governance
│   └── *.md                      # Broader architecture and roadmap docs
├── ops/
│   └── env/                      # Profile-specific env examples
├── tests/
│   ├── adapters/                 # Adapter-level tests
│   ├── bridges/                  # Bridge/historical smoke tests
│   ├── contracts/                # Contract compatibility tests
│   ├── data/                     # Data/reference/storage tests
│   ├── runtime/                  # Runtime and replay tests
│   └── statarb/                  # Config/settings tests
├── data/                         # Local datasets and artifacts
├── notebooks/                    # Research workspace
├── .cursor/                      # Repo-local rules, skills, settings
├── pyproject.toml                # Project packaging and dependencies
├── uv.lock                       # Locked dependency set
└── .env.example                  # Local env template for MVP profiles
```

## Ключевые зоны репозитория

### `core/`

- Источник legacy research и prototype logic.
- Можно читать для reference, переносить выводы и делать локальные bugfix/refactor без смены границ.
- Не является местом для новой `agent-operated` архитектуры.

### `statarb/`

- Основной кодовый контур текущего MVP.
- Здесь уже живут adapter, data, runtime и contract-aligned slices.
- Новая MVP-логика должна по умолчанию попадать сюда, а не в `core/`.

### `docs/superpowers/`

- Основной governance-слой для agent-operated работы.
- Здесь находятся:
  - `specs/` — активные design constraints;
  - `plans/` — текущий phased implementation plan и handoff;
  - `contracts/` — domain и operational contracts;
  - `runbooks/` — operational procedures;
  - `project-map.md` и `approval-matrix.md` — repo boundaries и decision gates.

### `docs/adr/`

- Долговечные архитектурные решения более высокого authority уровня.
- Эти документы не стоит переписывать ради status-sync, если конфликтов с текущим approved MVP нет.

### `tests/`

- Проверки для legacy и нового shell-кода.
- Основные тестовые срезы уже отражают contracts, adapter, runtime, replay и read-only smoke helpers.

### `ops/`

- Пока содержит только env-примеры для `local-dev` и `single-host-prod-like`.
- Более широкие runbooks и wiring для deployment profiles остаются следующим workstream, а не завершенной частью структуры.

### `data/` и `notebooks/`

- `data/` хранит локальные артефакты и датасеты.
- `notebooks/` остается исследовательским контуром и не считается authority source для runtime semantics.

## Что важно помнить при навигации

1. Для текущего execution backlog сначала читать `docs/README.md`, затем активный `design spec` и `implementation plan`.
2. Если задача меняет контракты, boundaries или runtime semantics, сначала обновляется docs/governance-слой.
3. `core/` и `statarb/` не следует смешивать в одном новом архитектурном срезе без явного migration reason.
4. Internal UI еще не является активным центром работы; до появления этого среза отсутствие `apps/` в репозитории нормально.
