# Целевая архитектура платформы статистического арбитража

## 1. Исходная точка репозитория

Сейчас репозиторий представляет собой качественный research-prototype для MOEX:

- Python-only стек вокруг `core/`;
- исторические данные через `AlorAPI`;
- локальное хранение в CSV;
- базовый анализ коинтеграции через Engle-Granger;
- walk-forward backtest пары;
- сохранение результатов в `runs/`;
- unit/integration-style тесты для текущего контура.

Это хорошая отправная точка для исследований, но не платформа хедж-фондового уровня. Сейчас отсутствуют:

- потоковый ingestion и нормализация market data;
- единый канонический event model;
- data lakehouse и serving-слой;
- feature store и версионирование датасетов;
- paper/live execution runtime;
- pre-trade и intraday risk;
- event-driven simulation/replay;
- internal UI для research, ops и risk;
- experiment tracking и model registry;
- observability, audit, reconciliation и compliance trail.

## 2. Целевой принцип платформы

Нужна не одна большая программа, а многослойная платформа, где research, simulation, control plane и live trading разделены, но говорят на одном языке данных и событий.

Базовые принципы:

1. `Immutable data first` — сырые рыночные события и reference data никогда не переписываются.
2. `Research/live parity` — те же event schema, что идут в live, должны воспроизводиться в replay/simulation.
3. `Version everything` — схемы событий, датасеты, признаки, модели, параметры портфеля и execution-policy должны быть версионированы.
4. `Core framework in-house` — execution, risk guards, replay, broker adapters и audit trail не должны зависеть от retail-библиотек.
5. `Python-first for MVP` — для foundation, research platform и execution/risk MVP дефолтным языком остается Python; системные языки появляются только после профилирования и подтвержденной необходимости.
6. `UI outside critical path` — human-facing UI обязателен как слой платформы, но он не должен держать брокерские соединения, исполнять ордера или быть точкой отказа для live-контуров.

## 3. Платформенные слои

## 3.1 Connectivity и ingestion

Задачи слоя:

- подключение к MOEX-совместимым каналам market data;
- reference data ingestion: инструменты, лоты, шаг цены, комиссии, расписания, клиринг;
- ingestion брокерских execution-событий: заявки, сделки, позиции, деньги, лимиты;
- нормализация всего в канонический внутренний event schema.

Рекомендуемый подход:

- `MOEX ISS` использовать для bootstrap/reference/historical metadata, но не как основной live-feed;
- для MVP/live-start использовать брокерские API и стримы;
- для low-latency phase 3/4 готовить direct market access контур;
- все адаптеры должны писать в общий event journal, а не напрямую в research-код.

## 3.2 Market data lakehouse

Слои данных:

- `raw` — неизмененные события биржи/брокера и reference snapshots;
- `normalized` — очищенные и нормализованные трейды, стаканы, бары, статусы торгов, справочники;
- `research` — датасеты под сигналы, альфы, риск, симуляцию;
- `serving` — представления для быстрых аналитических запросов, мониторинга и отчётности.

Стартовый storage-контур:

- `MinIO` как S3-совместимое объектное хранилище;
- `Parquet` как основной файловый формат сырого и research-слоя;
- `ClickHouse` как hot analytical serving-слой;
- `Postgres` как control plane для метаданных, конфигурации, экспериментов, оркестрации и аудита.

## 3.3 Feature store и dataset registry

Нужны два типа признаков:

- `offline features` для исследований, кросс-валидации и переобучения;
- `online features` для signal runtime, risk overlay и execution logic.

Обязательные категории признаков:

- mean reversion / spread state;
- VECM error-correction terms;
- Kalman state estimates и time-varying hedge ratio;
- cross-sectional residuals и factor-neutral z-scores;
- microstructure features: imbalance, queue pressure, spread regime, short-term toxicity proxies;
- liquidity/capacity features;
- regime features: vol regime, auction/clearing regime, session regime;
- broker/account state features.

Каждый feature set должен иметь:

- версию схемы;
- timestamp/valid-from;
- код генератора;
- lineage до raw dataset;
- ссылку на experiment/model/portfolio run.

## 3.4 Research environment

Research-контур должен поддерживать:

- notebooks для быстрых гипотез;
- package-based research code для productionizable logic;
- reproducible experiments;
- большие batch-run’ы и sweep’ы параметров;
- reproducible dataset snapshots.

Рекомендуемый stack:

- `Python 3.12+`, `uv`, `JupyterLab`;
- `Polars`, `DuckDB`, `Pandas`, `NumPy`, `statsmodels`, `SciPy`;
- `scikit-learn`, `PyTorch` для ML-блока;
- `MLflow` для experiment tracking и model registry;
- `Ray` позже, когда появится реальная потребность в распределенных sweep/backtest job.

## 3.5 Signal generation

Сигнальный слой должен поддерживать не один метод, а семейство альф:

- pair cointegration;
- basket/VECM statarb;
- state-space/Kalman residual trading;
- factor-neutral cross-sectional mean reversion;
- graph/network residual propagation;
- regime-aware alpha switching;
- event-aware short-horizon dislocation models.

Минимальный research roadmap по методам:

1. Engle-Granger + Johansen/VECM.
2. Dynamic hedge ratio через Kalman/state-space.
3. Cross-sectional residuals после нейтрализации по факторам.
4. Regime detection через HMM/Markov-switching или volatility-state models.
5. Graph/network overlays на correlated universe.

## 3.6 Portfolio construction

Портфельный слой нужен отдельно от сигнального:

- агрегировать альфы в общую книгу;
- нейтрализовать beta/sector/factor/issuer exposure;
- контролировать gross/net/leverage;
- учитывать liquidity/capacity;
- вводить turnover и borrow/financing constraints;
- оптимизировать не только alpha, но и alpha-after-cost.

Для MOEX-контекста особенно важны:

- роллирование фьючерсов;
- разные торговые сессии и клиринги;
- маржинальные требования и шаг цены;
- риск концентрации на отдельных эмитентах и секторах.

## 3.7 Execution, OMS и EMS

Execution-runtime должен быть event-driven, headless и отделен как от research, так и от UI.

Нужные сервисы:

- `signal intake` — принимает таргет-позиции и execution intents;
- `OMS` — жизненный цикл заявки и child-order orchestration;
- `EMS` — маршрутизация по брокеру/каналу/счету;
- `position service` — каноническое состояние позиций и денег;
- `execution analytics` — fill quality, queue loss, slippage, cancel ratio.

Минимальные execution-policy:

- passive maker-first;
- urgency-based taker fallback;
- schedule/TWAP-VWAP style slicing;
- liquidity caps и participation limits;
- session-aware execution around auctions/clearing.

Для MVP допустима Python-реализация этих сервисов, если они остаются отдельными процессами и имеют явные контракты событий. Переход на Rust/C++ допускается только для конкретных hotspot-компонентов после измерений.

## 3.8 Risk и controls

Risk должен жить как независимый слой, а не как несколько проверок внутри стратегии.

Нужны:

- pre-trade лимиты;
- intraday drawdown/volatility kill-switch;
- exposure by issuer/sector/factor/account;
- model drift и signal health monitoring;
- liquidity and capacity controls;
- stress scenarios;
- reconciliation broker vs internal ledger.

Риск-декомпозиция должна покрывать:

- factor risk;
- idiosyncratic spread risk;
- liquidity risk;
- execution risk;
- operational risk;
- model risk.

## 3.9 Simulation, backtest и replay

Нужен единый симуляционный контур:

- bar-based research backtest для быстрых циклов;
- event-driven replay для realistic execution simulation;
- market calendar/session engine;
- order lifecycle simulation;
- cost/slippage/capacity model;
- replay broker/exchange events в том же формате, что live.

Ключевой принцип: research backtest и execution replay должны использовать общие контракты событий, иначе перенос в live будет ломаться на интерфейсах.

## 3.10 Internal UI и control plane

Internal UI должен быть полноценной частью платформы, а не побочной мыслью. При этом UI не должен входить в execution critical path.

На текущем этапе достаточно одного внутреннего web-приложения с тремя доменными зонами:

- `Research UI` — каталог датасетов, run registry, backtest/replay inspection, experiment tracking, сравнение сигналов и портфельных конфигураций;
- `Ops/Trading UI` — состояние брокерских подключений, заявки, fills, позиции, cash, latency/reject dashboards, ручные operator actions и kill-switch;
- `Risk UI` — лимиты, exposure, breach history, scenario dashboards, model health и журнал approvals/overrides.

Архитектурные правила для UI:

- UI читает из read-optimized projections в `ClickHouse`/`Postgres`, а не напрямую из горячих внутренних state-машин;
- любые write-actions идут через отдельный `control-plane API`/command layer с аутентификацией, аудитом и явными ролями;
- отказ UI не должен останавливать `risk-service`, `execution-gateway` или брокерские адаптеры;
- UI не должен держать прямые брокерские соединения и не должен инкапсулировать торговую логику;
- research, ops и risk могут жить в одном frontend-репозитории, но должны быть разделены маршрутами, API-контрактами и permission-моделью.

## 3.11 Observability, audit и compliance

Обязательные элементы:

- trace-id на каждый signal, order, fill, risk decision;
- structured logs;
- metrics и SLO по latency/data gaps/order rejects;
- immutable audit trail;
- хранение конфигураций и parameter snapshots;
- reproducible reconstruction любого торгового дня.

## 4. Что писать самим, а что брать готовым

### Писать самим как core framework

- канонические event schema;
- market data normalizer;
- broker adapters и direct gateway adapters;
- OMS/EMS;
- position and ledger service;
- risk guards и kill-switches;
- event-driven replay/simulator;
- portfolio construction engine;
- execution cost/capacity models;
- audit/reconciliation layer;
- control-plane command semantics для торговых и risk-действий.

### Брать готовым и интегрировать

- econometrics/ML библиотеки;
- notebook tooling;
- object storage;
- analytical database;
- orchestration;
- observability stack;
- experiment tracking/model registry;
- frontend component libraries и charting.

Не стоит делать core платформу вокруг `backtrader`, `vectorbt`, `zipline` или похожих retail-framework’ов. Их можно использовать только как вспомогательный sandbox, но не как основу production/runtime.

## 5. Рекомендуемый технологический стек

## 5.1 MVP runtime и service stack

- `Python 3.12+` как дефолт для foundation, research platform и execution/risk MVP;
- `FastAPI` и `Pydantic` для control APIs, сервисных контрактов и операторских команд;
- `asyncio`-ориентированные сервисы там, где нужны стримы и сетевой I/O;
- `NATS JetStream` как стартовый bus для команд, событий и durable fan-out;
- `Protobuf` или совместимые версионируемые схемы сообщений для межсервисных контрактов;
- `uv`, `pytest`, `ruff` как базовая инженерная дисциплина;
- `Rust` или `C++` только позже, для выборочных hotspot-сервисов.

Почему так:

- Python дает кратчайший путь к research/live parity на текущем этапе проекта;
- существующий код, экспертиза и инструменты уже сосредоточены вокруг Python;
- один основной toolchain уменьшает time-to-MVP и упрощает сопровождение;
- системные языки имеют смысл только после измерений latency, jitter, throughput и failure modes.

## 5.2 Research stack

- `Python`, `JupyterLab`, `uv`;
- `Polars` и `DuckDB` как базовый исследовательский слой;
- `statsmodels`, `SciPy`, `scikit-learn`, `PyTorch`;
- `MLflow` для experiments/models;
- `pytest` и `ruff` как базовая инженерная гигиена.

## 5.3 Storage и serving

- `MinIO` для raw archive и dataset snapshots;
- `Parquet` для файлового слоя;
- `ClickHouse` для online analytics, screening, UI read-models и мониторинга;
- `Postgres` для метаданных, конфигурации, run registry, RBAC и audit control plane.

## 5.4 Frontend/UI stack

- `Vite` + `React` + `TypeScript` как дефолт для internal SPA;
- `Mantine` как основной UI-kit и layout/forms layer;
- `React Router` для route-based разделения `Research UI`, `Ops/Trading UI` и `Risk UI`;
- `TanStack Query` для server state и API-кэшей;
- `Zod` для boundary validation в UI/API;
- `lightweight-charts` для market/time-series и trading-style charting;
- `Apache ECharts` для risk, heatmap, scatter, distribution и dashboard-визуализаций;
- `AG Grid Community` применять точечно на самых плотных таблицах, если базовых Mantine-экранов уже недостаточно.

Почему так:

- стек полностью бесплатный и permissive: `Mantine` — MIT, `lightweight-charts` — Apache 2.0, `Apache ECharts` — Apache 2.0, `AG Grid Community` — MIT;
- `Mantine` дает связный API и готовые primitives вроде `AppShell`, форм, модалок, notifications и dates без платных расширений;
- `MUI` остается сильной альтернативой, но его `MUI X` является open-core: Community MIT, а продвинутые grid/picker/chart features уходят в Pro/Premium коммерческие лицензии;
- `@mantine/charts` полезен для простых KPI и базовых line/bar charts, но не должен быть основным charting engine для trading UI.

Подробное сравнение и practical recommendation зафиксированы в `docs/frontend_ui_architecture.md`.

## 5.5 Orchestration и workflow

- `Dagster` для batch/asset orchestration;
- отдельный live supervisor/runtime, не завязанный на DAG-инструмент;
- CI/CD через обычный git-based pipeline по мере появления сервисов.

## 5.6 Observability и operations

- `OpenTelemetry`;
- `Prometheus` + `Grafana`;
- `Loki` для логов;
- `Tempo` или совместимый trace backend;
- алерты по data quality, order rejects, position mismatch, risk limit breaches.

## 6. Минимальная целевая топология сервисов

На фазе foundation/research MVP достаточно следующего логического набора:

- `ingestion-adapters`;
- `reference-data-service`;
- `dataset-builder`;
- `research-jobs`;
- `feature-jobs`;
- `signal-batch-service`;
- `simulation-service`;
- `portfolio-service`;
- `risk-service`;
- `execution-supervisor`;
- `execution-gateway`;
- `audit-ledger`;
- `ui-read-models`;
- `control-plane-api`;
- `internal-ui`;
- `ops/monitoring`.

`internal-ui` и `control-plane-api` обязательны для operator workflow, но live-контур должен продолжать работать при их деградации или перезапуске.

## 7. Как этот repo должен эволюционировать

Практически разумный путь такой:

1. Сохранить `core/` как текущий working research-prototype.
2. Параллельно развивать новый платформенный каркас в `statarb/`.
3. Постепенно переносить reusable logic из `core/` в новые модули, не ломая текущие сценарии.
4. Выделять headless execution/risk сервисы и control-plane слой как отдельные процессы, даже если они остаются на Python.
5. Когда появится paper/live runtime, `core/` станет legacy-research compatibility layer, а не центром системы.
6. Рассматривать Rust/C++ только для подтвержденных hotspot-участков, а не как преждевременный baseline для всей платформы.
