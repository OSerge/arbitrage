# Дизайн agent-operated MVP

- Статус: accepted working baseline
- Дата: 2026-05-21

## Контекст и проблема

Текущий набор архитектурных документов уже хорошо описывает целевую платформу, но в основном задает образ большой statarb-системы, а не узкого исполнимого MVP. Для следующего шага нужен design doc, который фиксирует не "идеальную конечную архитектуру", а реалистичный первый рабочий контур.

Ключевое ограничение проекта: это не обычный solo-founder workflow, а `single-founder + AI agents` operating model. Пользователь утверждает только ключевые архитектурные, торговые и risk-решения. Подготовка кода, docs, skills, research-review, рефакторинг и большая часть операционной инженерной работы должны максимально выполняться агентами в пределах заранее заданных контрактов и decision gates.

Из этого следует, что MVP должен включать не только data/research/execution контур, но и `Agent Operating System` как first-class слой управления агентной разработкой и эксплуатацией.

## Цель дизайна

Зафиксировать исполнимую целевую форму `MVP = paper-first platform + controlled Alor test/live contour`, чтобы:

- сузить проект до реально достижимого первого торгового контура;
- задать пять верхнеуровневых слоев системы;
- определить обязательный MVP scope и явный out-of-scope;
- закрепить contract-first и replay-first подход;
- зафиксировать authority hierarchy и decision gates для агентной работы;
- не допустить появления новых сущностей, слоев и abstraction boundaries без обновления contract/doc/rule слоя.

## Принципы

1. `Executable MVP first` — сначала исполнимый paper/live pilot, а не имитация полной платформы фонда.
2. `Paper-first, live-by-gate` — paper execution обязателен раньше live; live допустим только как контролируемый пилот.
3. `Replay-first` — все критичные execution и risk-события должны быть пригодны для последующего replay и разбора.
4. `Contract-first` — данные, research run, signal intent, target position, order lifecycle и risk decisions оформляются через явные контракты.
5. `Agent-operated by default` — операционная модель проектируется так, чтобы агенты могли безопасно выполнять большую часть работы самостоятельно.
6. `Founder approval on key decisions` — стратегии, risk appetite, live enablement и архитектурные развилки не делегируются агентам по умолчанию.
7. `UI outside critical path` — internal UI обязателен для наблюдаемости и управления, но не должен быть точкой отказа для runtime.
8. `Python-first for MVP` — системные языки и тяжелая распределенная инфраструктура не являются gate для раннего runtime.
9. `No silent architecture drift` — агент не должен вводить новую сущность, слой или abstraction boundary без синхронного обновления contracts/docs/rules.

## Scope MVP

В MVP входят только следующие обязательные части:

1. `Data foundation`:
   - MOEX historical/reference data;
   - symbol master;
   - trade/session/calendar model;
   - Parquet как основной формат dataset/data snapshots;
   - Postgres как control-plane и metadata store.
2. `Research kernel`:
   - стандартизованный `research run`;
   - dataset snapshots;
   - walk-forward/validation pipeline;
   - одна основная alpha family: `pairs/baskets mean reversion + cointegration/VECM + simple regime filters`.
3. `Portfolio and risk overlay`:
   - target position model;
   - простые exposure/size limits;
   - cost/slippage assumptions;
   - portfolio constraints.
4. `Paper execution and replay`.
5. `Internal UI`:
   - `Research UI`;
   - `Ops UI`;
   - `Risk UI`.
6. `Agent Operating System`:
   - rules;
   - skills;
   - project map;
   - domain contracts;
   - approval matrix;
   - review/validation playbooks.
7. `AlorAPI adapter` в двух режимах:
   - тестовый/безопасный контур;
   - ограниченный боевой контур с ручным approval.

## Out of Scope

В этот MVP намеренно не входят:

- ранний multi-broker abstraction как обязательный платформенный слой;
- direct MOEX connectivity;
- heavy distributed stack;
- большой feature store;
- широкий ML stack;
- многоальфовая платформа;
- системные языки как обязательная часть early runtime;
- fully autonomous live trading без ручного approval на критичных шагах.

## Целевая форма MVP

MVP фиксируется как система из пяти верхнеуровневых слоев.

### 1. Research Kernel

Слой отвечает за reproducible research и выпуск стандартизованных research outputs для downstream portfolio/risk слоя, а не за live-исполнение. Его минимальная форма:

- стандартизованный `research run` с manifest и метриками;
- dataset snapshots и walk-forward validation;
- одна alpha family для парного и basket mean reversion;
- коинтеграция/VECM и простые regime filters;
- единый выходной артефакт для downstream portfolio/risk/runtime.

### 2. Trading Runtime

Слой отвечает за перевод target positions в paper/live действия под risk gates. Его минимальная форма:

- portfolio overlay и target position model;
- простые pre-trade exposure/size checks;
- paper execution;
- event replay;
- `AlorAPI adapter` в test/safe режиме и в ограниченном live-контуре;
- kill switch, audit trail и ручной approval для критичных live-действий.

### 3. Data Platform

Слой отвечает за канонический data foundation для research и runtime:

- MOEX historical/reference data;
- symbol master;
- trade/session/calendar model;
- versioned dataset snapshots;
- Parquet для dataset/data artifacts;
- Postgres для metadata, control-plane state и audit registry.

Полная storage-эволюция из более широких архитектурных документов остается допустимой траекторией, но не должна расширять MVP gate сверх этого минимального набора.

### 4. Internal Console

Слой дает пользователю и агентам наблюдаемость и операционное управление:

- `Research UI` для run/dataset/result inspection;
- `Ops UI` для статуса paper/live runtime, ордеров и событий;
- `Risk UI` для лимитов, блокировок, предупреждений и replay-разбора.

Это internal control plane, а не публичный продукт и не execution core.

### 5. Agent Operating System

Слой определяет, как репозиторий и проект управляются агентами без потери архитектурной дисциплины:

- repo/project map;
- rules и invariants;
- domain contracts;
- approval matrix;
- review/validation playbooks;
- skill layer для повторяемых агентных workflows.

Этот слой входит в MVP наравне с research и runtime, потому что без него проект снова превращается в founder-driven ручной контур.

## Agent Operating System

`Agent Operating System` должен сделать агентную работу безопасной, быстрой и проверяемой. Минимальный набор артефактов:

- `rules/invariants` — постоянные ограничения проекта и неоспоримые архитектурные правила;
- `skills/playbooks` — повторяемые workflows для docs, refactoring, review, validation, research-review и runtime-safe задач;
- `project map` — карта модулей, каталогов, ответственности и допустимых границ изменений;
- `domain contracts` — канонические интерфейсы между data, research, risk, execution, UI и agent workflows;
- `approval matrix` — что агент может менять автономно, а что требует founder approval;
- `review/validation playbooks` — как агент проверяет docs, contracts, tests, replay и risk-sensitive изменения перед передачей на ревью.

Требование слоя: если агент добавляет новую сущность, слой, domain object, boundary или runtime behavior, он обязан одновременно обновить соответствующий contract/doc/rule слой. Код не считается достаточным источником истины.

## Domain Contracts And Authority Hierarchy

### Ключевые domain contracts

Для MVP должны быть явно оформлены как минимум следующие контракты:

- `data contracts` — symbol master, instrument metadata, trade/session/calendar model, dataset snapshot manifest;
- `research contracts` — структура `research run`, параметры, входные dataset IDs, validation outputs, result artifacts;
- `portfolio/risk contracts` — target position model, exposure/size limits, constraint decisions;
- `execution contracts` — order intent, order event, fill event, position/ledger state, replay event schema;
- `ops contracts` — audit trail, operator actions, approval state, kill-switch state;
- `agent contracts` — repo map, allowed boundaries, required validation steps, definition of done для агентных задач.

### Иерархия authority sources

При конфликте источников действует следующий порядок, сверху вниз:

1. `ADR / project invariants`
2. `domain contracts`
3. `MVP scope / roadmap constraints`
4. `repo map / module boundaries`
5. `task-specific specs / plans`
6. `code`

Практическое следствие: код не может молча переопределять контракт или scope. Если нижний уровень требует отклонения от верхнего, агент должен сначала обновить более высокий слой и вынести решение на нужный gate approval.

## Decision Gates / Approval Model

### Агенты могут делать самостоятельно

- писать и уточнять docs/specs в пределах уже утвержденного дизайна;
- улучшать project map, skills, rules и playbooks без изменения торговой или архитектурной политики;
- выполнять рефакторинг внутри существующих contracts и module boundaries;
- добавлять tests, validation checks и replay tooling;
- развивать paper-only workflows и internal read models;
- готовить research-review, code-review и risk-review материалы.

### Требуется approval пользователя

- новая alpha family или существенное изменение торговой гипотезы;
- изменение target position logic, risk limits, slippage/cost assumptions или execution policy;
- любое live enablement, live scope expansion или ослабление guardrails;
- новый broker adapter, direct connectivity или обязательный multi-broker abstraction;
- новый платформенный слой, крупная abstraction boundary или breaking change в contracts;
- расширение инфраструктуры за пределы узкого MVP, если это влияет на стоимость, сложность или операционный риск.

### Обязательные ручные gate для controlled live contour

- включение live-режима;
- изменение перечня торговых инструментов или счетов для live;
- повышение risk limits и отключение kill switch;
- выполнение действий, после которых ордер может уйти в боевой контур без обратимого dry-run;
- принятие решения о переходе от test/safe режима к ограниченному live pilot.

## Порядок внедрения

1. `Зафиксировать spec и operating model`.
   На этом шаге фиксируются MVP scope, authority hierarchy, approval matrix и обязательные contracts для data/research/runtime/agents.
2. `Собрать data foundation и research kernel`.
   Исторические и reference данные MOEX, symbol master, calendar/session model, dataset snapshots, research run и walk-forward pipeline.
3. `Добавить portfolio/risk overlay, paper execution и replay`.
   Target positions, простые лимиты, cost/slippage assumptions, paper execution, audit trail и replay parity.
4. `Поднять internal console`.
   Минимальные `Research UI`, `Ops UI`, `Risk UI` поверх control-plane/read-model слоя.
5. `Подключить controlled Alor contour`.
   Сначала test/safe режим, затем ограниченный live pilot с жесткими pre-trade лимитами, kill switch, audit trail, replay и ручным approval.

## Acceptance Criteria

MVP считается оформленным правильно, если одновременно выполняются следующие условия:

- есть явная граница между `Research Kernel`, `Trading Runtime`, `Data Platform`, `Internal Console`, `Agent Operating System`;
- research запускается как стандартизованный `research run`, а не как набор разрозненных скриптов;
- dataset snapshots и validation pipeline воспроизводимы;
- одна alpha family покрывает pairs/baskets mean reversion с cointegration/VECM и simple regime filters;
- portfolio/risk overlay выпускает target positions под простыми exposure/size limits и cost/slippage assumptions;
- paper execution и replay используют совместимые contracts и пригодны для разбора торгового дня;
- internal UI покрывает research, ops и risk-наблюдаемость без встраивания UI в critical path;
- `Agent Operating System` оформлен как обязательный слой с rules, skills, project map, contracts, approval matrix и validation playbooks;
- `AlorAPI adapter` имеет два режима: test/safe и ограниченный live;
- live-пилот невозможен без kill switch, жестких pre-trade лимитов, audit trail, replay и ручного approval критичных действий;
- MVP не зависит от раннего multi-broker abstraction, direct MOEX connectivity, heavy distributed stack, большого feature store или широкого ML слоя.

## Риски и открытые вопросы

- `Alor test/live semantics` нужно отдельно уточнить на уровне конкретных режимов, ограничений и безопасного процесса включения live.
- `Risk thresholds` для controlled live pilot должны утверждаться пользователем отдельно; в этом spec фиксируется только необходимость gate и guardrails.
- `Storage rollout` шире Parquet/Postgres может понадобиться позже, но не должен незаметно превратиться в блокирующую часть MVP.
- `UI scope creep` остается риском: UI нужен как internal console, а не как отдельный продуктовый фронтенд.
- `Agent drift` остается риском без строгого соблюдения authority hierarchy; поэтому contract/doc/rule слой обязателен до любых крупных кодовых изменений.
