# Agent-Operated MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Довести репозиторий до первого `agent-operated` MVP-среза: `Agent Operating System -> domain contracts -> Alor adapter skeleton -> paper/replay contour`, с учетом реальных ограничений `AlorAPI` test contour и без преждевременного live enablement.

**Architecture:** Реализация остается в одном репозитории и идет поверх текущего research-prototype: существующий `core/` сохраняется как legacy-слой, а новый контур строится через пакет `statarb/`, docs/contracts в `docs/superpowers/` и отдельные deployment profiles `local-dev` и `single-host-prod-like`. Последовательность работ жестко серийная до заморозки контрактов, после чего отдельные потоки можно распараллеливать агентами.

**Tech Stack:** Python 3.12+, `uv`, текущий пакет `core`, новый пакет `statarb`, `.env` для секретов, `Postgres` + `Parquet` для control/data artifacts MVP, `React + TypeScript + Vite + Mantine` только для минимального internal UI после фиксации backend read-model contracts.

---

## 0. Текущий статус реализации

### Что уже зафиксировано

На ветке `feat/agent-operated-mvp` уже собраны и закоммичены такие checkpoint slices:

- `bbe3848` - foundation для `agent-operated` MVP: governance docs, rules/skills, базовый `statarb` shell и config profiles.
- `e877323` - domain contracts и skeleton адаптера `Alor`.
- `512e3e4` - executable historical `paper/replay` slice.
- `2247d6e` - `historical E2E smoke` и open `Alor` market-data contract.
- `03500c1` - foundation для хранения данных `AlorAPI` (`raw json.gz + normalized Parquet + manifest.json`).
- `48a424d` - bootstrap от normalized `Parquet` датасетов `Alor`.
- `5c6d031` - public fetch-runner для открытого `AlorAPI`.
- `0e52ef2` - dataset discovery и pair bootstrap для датасетов `Alor`.
- `b3d5f15` - safe read-only slice для `Alor` test contour и синхронизация handoff в плане.

### Что уже сделано по сути

- governance foundation и `Agent Operating System` базового уровня;
- executable domain contracts в docs и коде;
- `Alor` public history path: fetch -> raw -> normalized `Parquet` -> manifest;
- public reference/enrichment path: `Securities`, `availableBoards`, local symbol/board resolver;
- manifest-driven discovery и pair bootstrap runner;
- deterministic `paper/replay` runtime и historical smoke path;
- canonical bootstrap path от normalized `Alor` dataset с сохранением legacy fallback;
- safe read-only slice для `Alor` test contour: auth refresh/access, private read-only HTTP surfaces, redacted WS plan и ручной `smoke` helper.

### Что остается ключевым на следующую сессию

Ближайший рекомендованный следующий шаг:

1. вручную прогнать `Alor` test contour read-only smoke через `.env`;
2. связать `Securities/availableBoards enrichment` с `fetch/bootstrap` flow, чтобы `instrument_group` и `board` выбирались не вручную, а из локального `Alor`-ориентированного reference slice;
3. затем перейти к более полному adapter mapping из test contour payloads в canonical contracts и к первым read-model/operator workflows.

### Правило handoff

Новая сессия должна начинать работу с этого плана и последних checkpoint-коммитов, а не с повторного обхода всей архитектуры. Если предлагается новый шаг, он должен явно привязываться к одной из фаз ниже или быть оформлен как эволюционное уточнение текущей фазы.

---

## 1. Исходные допущения

- Одобренный первый implementation slice: `Agent Operating System -> domain contracts -> Alor adapter skeleton -> paper/replay contour`.
- Пользователь имеет доступ к тестовому контуру `Alor`; секреты на этом этапе допустимо хранить через `.env`.
- Ветка для работы уже выделена: `feat/agent-operated-mvp`.
- Реализация фич в рамках этого документа не выполняется; это plan/handoff артефакт.
- `core/` не переписывается сразу; MVP строится рядом, с чистой границей между legacy research и новым platform shell.
- Для MVP live-trading остается out-of-scope; допускается только подготовка controlled contour и contract surface.

## 2. Рекомендация по branch/repo strategy

**Рекомендация:** продолжать в текущем репозитории, в отдельной feature-ветке, без выделения отдельного репозитория.

Почему это правильно именно сейчас:

- одобренный MVP опирается на уже существующие `core/`, `tests/`, историческую интеграцию с `AlorAPI` и текущий packaging-контекст;
- `Agent Operating System`, contracts, adapter skeleton и paper/replay должны развиваться рядом с legacy-кодом, чтобы агентам было проще ссылаться на один источник истины;
- отдельный репозиторий сейчас внесет лишний drift: ADR/spec/plan будут жить отдельно от кода, а MVP-срез станет тяжелее для review и агентной навигации;
- отдельная ветка уже дает нужную изоляцию для планомерной миграции без риска задеть старый research-contour.

Когда отдельный репозиторий может стать уместным позднее:

- если `execution/risk/control-plane` превратятся в самостоятельную систему с отдельным release-cycle;
- если появится отдельная команда или отдельные права доступа на runtime-контур;
- если понадобится жесткое организационное разделение research-кода и operational/live-инфраструктуры.

## 3. Ключевые выводы по `AlorAPI` test contour

Ниже перечислены факты, на которых основан план. Источники: официальная документация `alor.dev`.

### 3.1 Контуры и base URLs

- `Боевой` и `Тестовый` контуры изолированы друг от друга.
- Тестовый API-host: `https://apidev.alor.ru`.
- Тестовый auth-host: `https://oauthdev.alor.ru`.
- WebSocket market-data/subscriptions: `wss://apidev.alor.ru/ws`.
- WebSocket order commands: `wss://apidev.alor.ru/cws`.

### 3.2 Аутентификация

- В тестовом контуре поддерживается только JWT-механизм; OAuth service provider для third-party app flow там не используется как в бою.
- `Refresh Token` для test contour живет 1 месяц.
- `Access Token` живет 30 минут.
- Для test account одновременно доступен только один `Refresh Token`.
- Для `/cws` авторизация делается отдельной командой `authorize`; токен нужно обновлять заранее, потому что сервер не присылает proactive notification об истечении.
- Для `/ws` токен передается в каждом subscription request.

### 3.3 Сущности и события, нужные для MVP

- Рыночные данные: `QuotesSubscribe`, `OrderBookGetAndSubscribe`.
- Состояние портфеля: `PositionsGetAndSubscribeV2`, `SummariesGetAndSubscribeV2`.
- Лайфцикл исполнения: `OrdersGetAndSubscribeV2`, `TradesGetAndSubscribe`.
- HTTP API подходит для initial load, справочников и historical data; real-time polling через HTTP делать нельзя.

### 3.4 Лимиты и operational constraints

- HTTP API: не более `100 req/s` на пользователя.
- WebSocket: не более `10` соединений, не более `5000` подписок на сессию, не более `5000` необработанных сообщений в серверном буфере.
- Для order commands документация рекомендует использовать одно `cws`-соединение на весь order-management поток.
- В рамках одного `cws`-соединения каждый `guid` обязан быть уникальным; повторный `guid` может привести к ошибке `400` и разрыву соединения.
- При ошибке авторизации `cws` соединение закрывается сервером.

### 3.5 Важные отличия test vs live

- Test contour использует симуляционные данные, а не реальные биржевые данные.
- Расписание test contour отличается от реального биржевого расписания.
- Торги в test contour могут прерываться техническими работами.
- Состояние тестового счета и операций может обнуляться при перезапусках торгов.

### 3.6 Выводы для дизайна адаптера

- Нужен отдельный `token-refresh manager`, а не разовые токены в коде.
- Нужно раздельно проектировать `read-only ws subscriptions` и `command cws`.
- Нужен локальный `connection supervisor`: reconnect, backoff, idempotent resubscribe, stale-token rotation.
- Нужен строгий mapper из `Alor`-событий в канонические `order/fill/position/account-state` контракты.
- Тестовый контур нельзя использовать как источник истины для latency, market microstructure и PnL-quality; он нужен для integration semantics и operator workflows.

Официальные страницы, которые стоит держать под рукой в реализации:

- `https://alor.dev/docs/api/access/environments`
- `https://alor.dev/docs/en/api/access/authorization/dev-env-auth`
- `https://alor.dev/docs/en/api/usage/orders/overview`
- `https://alor.dev/docs/en/api/websocket/data-subscriptions/overview`
- `https://alor.dev/docs/en/api/usage/limits-and-recommendations`

## 4. Целевой file map для первого MVP-среза

Ниже не "идеальная конечная архитектура", а practical file map для ближайшей реализации.

### 4.1 Docs and governance

- Create: `docs/README.md`
- Create: `docs/superpowers/project-map.md`
- Create: `docs/superpowers/approval-matrix.md`
- Create: `docs/superpowers/contracts/README.md`
- Create: `docs/superpowers/contracts/agent-operating-system.md`
- Create: `docs/superpowers/contracts/domain-data.md`
- Create: `docs/superpowers/contracts/domain-research.md`
- Create: `docs/superpowers/contracts/domain-execution.md`
- Create: `docs/superpowers/contracts/domain-ops.md`
- Create: `docs/superpowers/runbooks/alor-test-contour.md`

### 4.2 Python platform shell

- Modify: `pyproject.toml`
- Create: `statarb/__init__.py`
- Create: `statarb/config/__init__.py`
- Create: `statarb/config/settings.py`
- Create: `statarb/config/profiles.py`
- Create: `statarb/domain/__init__.py`
- Create: `statarb/domain/instruments.py`
- Create: `statarb/domain/research.py`
- Create: `statarb/domain/orders.py`
- Create: `statarb/domain/positions.py`
- Create: `statarb/domain/events.py`

### 4.3 Alor adapter

- Create: `statarb/adapters/__init__.py`
- Create: `statarb/adapters/alor/__init__.py`
- Create: `statarb/adapters/alor/endpoints.py`
- Create: `statarb/adapters/alor/auth.py`
- Create: `statarb/adapters/alor/http_market_data.py`
- Create: `statarb/adapters/alor/ws_market_data.py`
- Create: `statarb/adapters/alor/ws_portfolio.py`
- Create: `statarb/adapters/alor/cws_orders.py`
- Create: `statarb/adapters/alor/mapper.py`

### 4.4 Paper and replay runtime

- Create: `statarb/runtime/__init__.py`
- Create: `statarb/runtime/paper/__init__.py`
- Create: `statarb/runtime/paper/engine.py`
- Create: `statarb/runtime/paper/risk_checks.py`
- Create: `statarb/runtime/paper/ledger.py`
- Create: `statarb/runtime/paper/event_sink.py`
- Create: `statarb/runtime/replay/__init__.py`
- Create: `statarb/runtime/replay/loader.py`
- Create: `statarb/runtime/replay/replayer.py`

### 4.5 Control plane and minimal UI

- Create: `statarb/controlplane/__init__.py`
- Create: `statarb/controlplane/read_models.py`
- Create: `statarb/controlplane/operator_actions.py`
- Create: `apps/internal-ui/package.json`
- Create: `apps/internal-ui/src/main.tsx`
- Create: `apps/internal-ui/src/app/routes.tsx`
- Create: `apps/internal-ui/src/features/ops/runtime-overview.tsx`
- Create: `apps/internal-ui/src/features/risk/guardrails.tsx`
- Create: `apps/internal-ui/src/features/research/replay-runs.tsx`

### 4.6 Tests and fixtures

- Create: `tests/contracts/test_domain_contracts.py`
- Create: `tests/adapters/alor/test_auth.py`
- Create: `tests/adapters/alor/test_mapper.py`
- Create: `tests/runtime/test_paper_engine.py`
- Create: `tests/runtime/test_replayer.py`

## 5. Workstreams

1. `Agent Operating System foundation`
2. `Domain contracts`
3. `Alor adapter and test contour integration`
4. `Paper/replay runtime`
5. `Minimal UI read-model slice`
6. `Environment/config and deployment profiles`

## 6. Sequence of execution

### Phase 1: Agent Operating System foundation

**Objective:** создать обязательный governance-слой, без которого агентная реализация быстро уйдет в architecture drift.

**Serial dependencies:**

- стартовая структура каталогов должна быть подтверждена;
- нужно решение, что `statarb/` становится новой platform-shell точкой входа, а `core/` остается legacy.

**Parallelizable tasks after kickoff:**

- drafting `project-map`;
- drafting `approval-matrix`;
- drafting `alor-test-contour` runbook;
- подготовка пустого `statarb/` package shell.

**Deliverables:**

- `docs/superpowers/project-map.md`
- `docs/superpowers/approval-matrix.md`
- `docs/superpowers/runbooks/alor-test-contour.md`
- пустой, но импортируемый `statarb/`
- обновленный `pyproject.toml` для packaging нового shell

**Acceptance criteria:**

- есть явный ответ, где заканчивается `core/` и начинается `statarb/`;
- зафиксировано, что агент может менять автономно, а что требует approval;
- для `Alor` test contour есть отдельный runbook по токенам, reconnect и operational caveats;
- любой новый агент получает понятную карту репозитория без чтения всего кода.

**User approvals required:**

- подтверждение package boundary: `core` как legacy, `statarb` как новый shell;
- подтверждение, что `.env` остается допустимым secret-mechanism для MVP.

- [x] Создать `docs/README.md` и базовый индекс `docs/superpowers/`.
- [x] Зафиксировать `project-map` и `approval-matrix`.
- [x] Добавить runbook по `Alor` test contour.
- [x] Подготовить `statarb/` package shell и packaging changes.
- [x] Проверить, что docs и package shell не меняют existing research behavior.

### Phase 2: Domain contracts

**Objective:** зафиксировать канонические contracts до адаптера и runtime, чтобы paper/replay и broker integration строились на одном языке данных.

**Serial dependencies:**

- Phase 1 завершена;
- `project-map` и `approval-matrix` приняты;
- подтвержден состав обязательных сущностей: `instrument`, `research run`, `order intent`, `order event`, `fill event`, `position snapshot`, `account summary`, `replay envelope`.

**Parallelizable tasks after contract skeleton is accepted:**

- data/research contracts;
- execution/ops contracts;
- типы идентификаторов и event envelopes;
- тестовые фикстуры контрактов.

**Deliverables:**

- docs в `docs/superpowers/contracts/`
- Python modules в `statarb/domain/`
- базовые contract tests в `tests/contracts/`

**Acceptance criteria:**

- paper runtime, replay и Alor mapper используют одни и те же доменные объекты;
- user-facing approvals и operator actions имеют отдельный contract, а не зашиты в произвольные dict-структуры;
- есть минимальный contract-test набор на сериализацию и backward-compatible shape.

**User approvals required:**

- freeze списка обязательных domain entities;
- отдельное подтверждение любых breaking changes в уже зафиксированных contracts.

- [x] Зафиксировать docs contracts по data/research/execution/ops.
- [x] Реализовать executable contract shell в `statarb/domain/`.
- [x] Добавить fixture-driven tests для сериализации и базовой совместимости.
- [x] Согласовать contract freeze перед началом adapter work.

### Phase 3: `AlorAPI` adapter/test contour research and integration

**Objective:** построить безопасный adapter skeleton, ориентированный сначала на test contour semantics, read-only subscriptions и canonical event mapping.

**Serial dependencies:**

- Phase 2 contract freeze завершен;
- известны env var names для токенов, portfolio IDs и contour selection;
- есть runbook по обновлению токенов и reconnect policy.

**Parallelizable tasks after adapter boundary is frozen:**

- `auth.py` и token rotation;
- `http_market_data.py` для справочников/истории;
- `ws_market_data.py` для quotes/order book;
- `ws_portfolio.py` для positions/orders/trades/summaries;
- `mapper.py` из Alor payloads в domain events.

**Deliverables:**

- endpoint constants для `api`, `apidev`, `ws`, `cws`;
- refresh/access token manager;
- read-only websocket clients;
- command websocket skeleton без live enablement;
- canonical mapper и adapter-level tests;
- docs section с test-vs-live caveats.

**Acceptance criteria:**

- test contour выбирается конфигурацией, а не хардкодом;
- токен rotation отделен от business logic;
- adapter умеет читать quotes/orders/trades/positions/account state и маппить их в внутренние контракты;
- order-command path существует как skeleton, но не создает неявный live-ready behavior;
- reconnect и duplicate-guid policy задокументированы и покрыты тестами на уровне логики.

**Dependencies/blockers:**

- валидный `Refresh Token`;
- известный `portfolio` для тестового счета;
- фактическая доступность test contour в момент smoke-check;
- возможные отличия в payloads по сравнению с live.

**User approvals required:**

- approval env naming convention (`ALOR_TEST_REFRESH_TOKEN`, `ALOR_TEST_PORTFOLIO`, и т.п.);
- отдельный approval перед любым command smoke-check, который выходит за пределы read-only semantics.

- [x] Зафиксировать endpoint map и contour selection.
- [x] Реализовать auth/token rotation shell.
- [x] Добавить read-only HTTP/WS clients.
- [x] Добавить canonical mapper для orders/fills/positions/account state.
- [x] Подготовить `cws` command skeleton без включения real trading path.
- [ ] Выполнить только safe smoke-checks против test contour.

### Phase 4: Paper/replay runtime

**Objective:** довести новый runtime до состояния, где target/intents проходят через risk checks, paper ledger и replay, не требуя реального брокерского исполнения.

**Serial dependencies:**

- Phase 3 adapter mapping стабилен;
- order/fill/position/account state contracts уже не меняются каждый день;
- понятна стратегия event persistence для MVP.

**Parallelizable tasks after event envelope is frozen:**

- `paper/ledger.py`;
- `paper/risk_checks.py`;
- `paper/event_sink.py`;
- `replay/loader.py`;
- `replay/replayer.py`;
- test fixtures из canonical events.

**Deliverables:**

- paper execution engine;
- deterministic event sink;
- replay loader/replayer;
- contract-compatible audit trail;
- tests на lifecycle: intent -> accepted/rejected -> fill -> position update -> replay parity.

**Acceptance criteria:**

- paper runtime не зависит от UI;
- replay воспроизводит тот же event shape, что и paper/adapter path;
- риск-проверки выражены явно и тестируются отдельно;
- можно разобрать торговый день post-factum без обращения к исходным брокерским payloads.

**Dependencies/blockers:**

- решение по минимальному event journal format;
- решение, где хранятся replay artifacts в `local-dev` и `single-host-prod-like`;
- наличие репрезентативных test contour payload samples.

**User approvals required:**

- approval минимального набора risk guards;
- approval event retention strategy, если она тянет за собой новые storage dependencies.

- [x] Собрать paper engine поверх domain contracts.
- [x] Добавить explicit risk checks и paper ledger.
- [x] Реализовать event sink и replay loader.
- [x] Зафиксировать replay parity tests.
- [ ] Подготовить короткий operator workflow для paper trading day.

### Phase 5: Minimal UI read-model slice

**Objective:** поднять минимальный internal UI, который читает готовые read-models и помогает наблюдать paper/replay контур, но не держит execution logic.

**Serial dependencies:**

- Phase 4 завершена;
- read-model shape стабилен;
- UI stack formally approved.

**Parallelizable tasks after read-model API shape is frozen:**

- `apps/internal-ui` shell;
- `ops` route;
- `risk` route;
- `research/replay` route.

**Deliverables:**

- минимальный SPA-контур;
- три route-группы: `research`, `ops`, `risk`;
- один обзорный экран runtime;
- один экран guardrails/approvals;
- один экран replay runs и event inspection.

**Acceptance criteria:**

- UI показывает read-only состояние runtime и replay;
- UI не открывает брокерские соединения;
- деградация UI не влияет на paper runtime;
- screen scope остается минимальным: status, ledger summary, approvals, replay inspection.

**Dependencies/blockers:**

- появление `Node/Vite` toolchain в репозитории;
- решение, где живет control-plane API или file-backed read-model gateway;
- ограничение объема UI, чтобы не сорвать MVP в frontend workstream.

**User approvals required:**

- отдельный approval на добавление frontend toolchain;
- approval конкретного минимального набора экранов до начала реализации.

- [ ] Подготовить `apps/internal-ui` shell.
- [ ] Зафиксировать read-model contract между backend и UI.
- [ ] Реализовать по одному минимальному экрану для `research`, `ops`, `risk`.
- [ ] Проверить, что UI остается вне critical path.

### Phase 6: Environment/config strategy and deployment profiles

**Objective:** сделать так, чтобы один и тот же MVP-код корректно жил в двух режимах: `local-dev` и `single-host-prod-like`.

**Deployment profile: `local-dev`**

- локальная `.env` конфигурация;
- Python-процессы под `uv`;
- file-backed replay artifacts;
- локальный `Postgres` либо containerized dependency only when needed;
- ручной запуск и ручные smoke-checks.

**Deployment profile: `single-host-prod-like`**

- один хост, несколько процессов или compose-services;
- те же contracts и env names, что и в `local-dev`;
- `Postgres` обязателен для control-plane state;
- process supervision, restart policy, structured logs;
- никакого live enablement по умолчанию.

**Files:**

- Create: `.env.example`
- Create: `ops/env/local-dev.env.example`
- Create: `ops/env/single-host-prod-like.env.example`
- Create: `ops/compose/single-host-prod-like.yml`
- Create: `ops/runbooks/local-dev.md`
- Create: `ops/runbooks/single-host-prod-like.md`

**Acceptance criteria:**

- все config knobs описаны явно;
- test/live contour selection делается profile config, а не редактированием кода;
- секции с секретами и portfolio IDs отделены от non-secret config;
- `single-host-prod-like` не требует новой архитектуры, только другой wiring.

**User approvals required:**

- approval списка env vars;
- approval добавления `Postgres` как обязательной control-plane зависимости для prod-like profile;
- отдельный approval на любые live-related toggles, даже если они остаются выключенными.

- [x] Зафиксировать `settings.py` и `profiles.py`.
- [x] Подготовить `.env.example` и profile-specific env examples.
- [ ] Описать запуск и recovery runbooks для двух профилей.
- [x] Проверить одинаковость contract surface между профилями.

## 7. Что можно делать параллельно агентами

После завершения серийной цепочки `Phase 1 -> Phase 2` допускается безопасное распараллеливание:

- один агент пишет docs contracts, другой одновременно собирает executable `statarb/domain/` types;
- один агент делает `Alor auth/endpoints`, другой `mapper`, третий `ws subscriptions`;
- один агент делает `paper ledger`, второй `risk checks`, третий `replay loader`;
- один агент поднимает `read_models`, другой — минимальный UI shell, но только после freeze API shape.

## 8. Что требует строгого serial order

- freeze package layout;
- freeze domain contracts;
- подтверждение env/config naming;
- переход от adapter skeleton к runtime wiring;
- freeze read-model contracts перед UI;
- любые шаги, которые затрагивают live semantics, order commands или risk limits.

## 9. Основные blockers и зависимости

- наличие валидного test account и актуального `Refresh Token`;
- знание конкретных `portfolio` identifiers для test contour;
- подтверждение package boundary между `core/` и `statarb/`;
- согласование минимального набора risk guards;
- решение по event journal format и storage path для replay artifacts;
- отдельный user approval на добавление frontend toolchain.

## 10. Definition of done для первого implementation slice

Срез считается завершенным, если одновременно выполняются все пункты:

- есть `Agent Operating System` foundation docs и `project-map`;
- frozen `domain contracts` существуют и в docs, и в executable shell;
- `Alor` test contour подключен через adapter skeleton с auth rotation и read-only subscriptions;
- `paper runtime` и `replay` используют одни и те же canonical events;
- минимальный `ops/risk/research` read-model slice доступен без подключения UI к critical path;
- существуют два deployment profiles: `local-dev` и `single-host-prod-like`;
- live trading по-прежнему не включается без отдельного approval.

## 11. Порядок handoff в следующую сессию

- `Phase 1`, `Phase 2`, существенная часть `Phase 3`, `Phase 4` и config-basis `Phase 6` уже выполнены.
- Начинать не с переосмысления foundation, а с ближайшего открытого шага: ручной `Alor test contour read-only smoke`, затем `enrichment-aware fetch/bootstrap helper`.
- Не перескакивать сразу к `UI` или `live`.
- После каждого phase gate делать review against this plan и contracts.
- Любой новый domain object сначала добавлять в contracts/docs, потом в код.
- Любой намек на live enablement выносить на отдельное пользовательское решение.
