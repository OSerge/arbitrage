# Roadmap развития платформы

> Текущий активный execution contour уже, чем весь этот roadmap: рабочим backlog для ветки служит узкий `agent-operated` MVP из `docs/superpowers/specs/2026-05-21-agent-operated-mvp-design.md` и его implementation plan. Этот roadmap остается долгосрочной траекторией, а не списком ближайших задач на следующую сессию.

## Принципы roadmap

Roadmap строится не вокруг "добавить еще пару моделей", а вокруг последовательного появления платформенных слоев:

1. foundation;
2. research platform;
3. execution/risk MVP;
4. institutionalization.

Каждая следующая фаза должна использовать артефакты предыдущей, а не переписывать их с нуля.

## Фаза 1. Foundation

Цель: превратить текущий репозиторий из point-solution в платформенную заготовку.

### Результат фазы

- единый архитектурный контур и ADR;
- новый пакетный каркас `statarb/`;
- канонические домены: data, research, simulation, execution, risk, ops;
- `Python-first` стратегия для foundation и MVP;
- versioned config/schema discipline;
- базовый control-plane storage: Postgres/MinIO/ClickHouse как целевой контур;
- решение по internal UI для `Research UI`, `Ops/Trading UI`, `Risk UI`;
- каталог источников данных и брокерских адаптеров;
- требования к audit и observability.

### Конкретные задачи

- описать symbol master и naming conventions;
- ввести единый trade/session/calendar model для MOEX;
- определить формат raw event journal;
- выделить reference data как отдельный asset;
- определить backlog direct vs broker connectivity;
- зафиксировать UI/control-plane границу относительно execution critical path;
- выбрать бесплатный permissive frontend stack для internal UI;
- зафиксировать build-vs-buy границу.

### Критерий завершения

Есть понятный технический контур, в который можно безопасно добавлять новые сервисы и исследования без дальнейшего архитектурного дрейфа.

## Фаза 2. Research Platform

Цель: сделать сильный исследовательский контур, который масштабируется лучше текущих pair-scripts.

### Результат фазы

- reproducible dataset snapshots;
- offline feature store;
- experiment tracking и model registry;
- batch-пайплайны по universe construction и feature generation;
- первый рабочий `Research UI` для dataset/run/model inspection;
- набор продвинутых alpha-моделей.

### Приоритетные методы

- Engle-Granger + Johansen/VECM;
- dynamic hedge ratio через Kalman/state-space;
- factor-neutral cross-sectional mean reversion;
- regime detection;
- graph/network residual propagation.

### Инженерные задачи

- перевести research code из notebook-first в package-first;
- сделать event-replay datasets;
- научиться запускать parameter sweeps и walk-forward experiments воспроизводимо;
- подключить UI к registry, backtest и replay read-models;
- внедрить capacity/slippage estimation в исследовательский контур, а не считать его "последним шагом".

### Критерий завершения

Новая модель, признак или universe rule попадает в reproducible pipeline с понятным lineage, а не живет только в одном notebook.

## Фаза 3. Execution/Risk MVP

Цель: получить paper/live execution для MOEX через брокерские адаптеры и независимый risk layer, сохраняя Python-first реализацию и отделяя UI от critical path.

### Результат фазы

- broker adapters минимум для одного основного и одного резервного подключения;
- signal-to-order pipeline;
- OMS/EMS light;
- position service;
- pre-trade limits;
- intraday risk limits;
- `Ops/Trading UI` и `Risk UI` поверх read-model/control-plane слоя;
- paper trading и controlled live pilot.

### Практический scope

- `Alor` как стартовый адаптер;
- `Finam` или `T-Invest` как второй адаптер для redundancy и сравнения operational profile;
- order/exec event schema;
- reconciliation broker statements vs internal ledger;
- headless execution/risk services, не завязанные на web UI;
- dashboards по latency, reject rate, slippage, position mismatch.

### Критерий завершения

Платформа способна принять signal intent, прогнать risk checks, отправить заявки, получить fills, пересчитать позицию и воспроизвести весь день по audit trail.

## Фаза 4. Institutionalization

Цель: довести систему до уровня устойчивой командной/институциональной эксплуатации.

### Результат фазы

- direct/hosted market connectivity для low-latency use cases;
- вынос отдельных hotspot-компонентов в системные языки только при доказанной необходимости;
- строгий change-management;
- replayable incident response;
- disaster recovery и резервирование;
- формализованный model governance;
- compliance-grade audit retention;
- multi-strategy и multi-venue extensibility.

### Что появляется именно здесь

- direct MOEX connectivity, если капитал и организационный контур это оправдывают;
- Rust/C++ только для подтвержденных latency/jitter bottlenecks;
- расширение beyond MOEX;
- formal release train и environment promotion;
- сервисные SLO и on-call operational playbooks;
- полномасштабный capacity management.

### Критерий завершения

Система устойчиво работает как платформа, а не как набор исследовательских скриптов, привязанных к одному человеку и одному брокеру.

## Последовательность внедрения по времени

Ориентир без обещания сроков:

1. `Foundation` — 4-6 недель.
2. `Research Platform` — 6-10 недель.
3. `Execution/Risk MVP` — 8-12 недель.
4. `Institutionalization` — по мере подтверждения alpha, капитала и operational maturity.

## Что не стоит делать слишком рано

- не строить colocation/direct-exchange контур до подтверждения research edge;
- не тянуть тяжелый distributed stack до реальной потребности;
- не переписывать execution/risk контур на системные языки до профилирования;
- не завязывать runtime на notebook-код;
- не завязывать paper/live execution на доступность internal UI;
- не смешивать signal logic, portfolio logic и execution logic в одном модуле;
- не опираться на один брокер и один канал market data как на "вечное" решение.
