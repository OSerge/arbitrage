# Frontend/UI архитектура для внутренней платформы

## 1. Контекст и цель

Платформе нужен не "маркетинговый frontend", а рабочий internal UI для трех задач:

- `Research UI` — анализ датасетов, backtest/replay runs, экспериментов и сигналов;
- `Ops/Trading UI` — мониторинг торгового дня, позиций, заявок, fills, коннективности и operator actions;
- `Risk UI` — лимиты, exposure, breach history, сценарии и approvals.

Требования текущего этапа:

- полностью бесплатный стек без обязательных платных расширений;
- permissive лицензии;
- пригодность для AI-agent coding;
- минимальный риск "хрупкого frontend-ада";
- нормальная поддерживаемость человеком, который не является сильным фронтендером;
- пригодность для data-dense trading/research экранов.

## 2. Архитектурное решение

На текущем этапе рекомендуется один internal SPA-контур на `React + TypeScript`, но с явным разделением доменов по маршрутам и API:

- `/research/*`
- `/ops/*`
- `/risk/*`

Это одно приложение, но не один смешанный модуль. Research, ops и risk должны иметь:

- свои read-models;
- свои permission boundaries;
- свои наборы экранов и рабочих сценариев.

`Internal UI` должен быть отделен от execution critical path:

- UI читает из `ClickHouse`/`Postgres` read-models и control-plane API;
- UI не держит брокерские соединения;
- UI не должен быть местом, где исполняется торговая логика;
- деградация UI не должна останавливать `risk-service`, `execution-supervisor` или `execution-gateway`.

## 3. Рекомендуемый базовый стек

### Базовый стек

- `Vite`
- `React`
- `TypeScript`
- `Mantine`
- `React Router`
- `TanStack Query`
- `Zod`

### Визуализация и плотные экраны

- `lightweight-charts` как основной движок для market/time-series и trading-style charts;
- `Apache ECharts` для heatmap, scatter, histogram, treemap, exposure, risk dashboards;
- `AG Grid Community` применять точечно на самых плотных таблицах, где обычных Mantine-экранов уже недостаточно.

### Почему это базовый выбор

- весь базовый стек бесплатный и permissive;
- `Mantine` дает связную систему компонентов, layout, форм, модалок, notifications и дат без перехода на платные пакеты;
- стек хорошо подходит для кода, который генерируют AI-агенты: меньше ручной стилизации, меньше "склейки" из разнородных примитивов, меньше мест для CSS/runtime drift;
- `Vite` проще для internal SPA, чем более тяжелые SSR-фреймворки, если на старте не нужен публичный SEO/SSR-контур;
- `TanStack Query` и `Zod` уменьшают класс типичных agent mistakes на API-границах и в кэшировании server state.

## 4. Почему `Mantine` — основной выбор

По официальной документации Mantine:

- все `@mantine/*` пакеты распространяются под `MIT`;
- в стартовом наборе уже есть `AppShell`, формы, notifications, modals, dates и hooks;
- `@mantine/charts` существует как встроенный пакет для простых графиков и основан на `Recharts`.

Практически для этого проекта `Mantine` хорош тем, что:

- дает цельный API для большинства internal screen patterns;
- снижает объем ручного CSS и тему можно держать централизованно;
- хорошо подходит для route-based application shell;
- не требует premium-лицензий для базовой enterprise/internal ergonomics;
- выглядит более pragmatic и менее "конструкторским", чем стек из headless primitives.

Слабое место `Mantine`: это не готовый heavy trading grid platform. Поэтому dense tables лучше решать отдельно, а не пытаться силой протащить всё через базовый `Table`.

## 5. Сравнение ключевых альтернатив

### `MUI`

Что важно по лицензированию:

- `Material UI` / `MUI Core` — open-source, `MIT`;
- `MUI X Community` — `MIT`, free forever;
- `MUI X Pro` и `MUI X Premium` — коммерческие лицензии.

По официальной лицензии и pricing-документации MUI:

- платными являются продвинутые возможности `Data Grid`, `Date and Time Range Pickers`, часть advanced chart features и другие X-расширения;
- для коммерческих пакетов нужен license key;
- лицензирование считается по числу concurrent front-end developers, меняющих код проекта.

Вывод:

- `MUI` остается сильной опцией, если нужен batteries-included React UI с отличной документацией;
- но для этого проекта он не лучший default, потому что бесплатная граница заканчивается как раз там, где internal trading UI часто начинает хотеть больше: rich data grid, advanced interactions, scheduler-like patterns и enterprise conveniences;
- если есть жесткое требование "полностью бесплатный стек", `MUI` создает слишком высокий риск позднего упора в paid features.

### `Ant Design`

По лицензии `Ant Design` — `MIT`.

Сильные стороны:

- очень зрелый enterprise/internal UI toolkit;
- хорошо подходит для data-dense форм, таблиц и admin-style экранов;
- в экосистеме много готовых паттернов.

Почему не default:

- дизайн и API более тяжеловесны и opinionated;
- при множественных agent edits выше риск стилистического и архитектурного "разъезда";
- для этого проекта `Mantine` обычно проще в сопровождении и мягче в кастомизации.

### `AG Grid Community`

По официальной документации:

- `AG Grid Community` — free for everyone, включая production use;
- лицензия — `MIT`;
- row/column virtualization включены по умолчанию;
- `Enterprise` добавляет grouping, pivoting, Excel export, integrated charts и другие advanced features.

Вывод:

- это очень сильная бесплатная опция именно для тяжелых trading/ops/risk таблиц;
- но как фундамент всего frontend-стека `AG Grid` слишком специализирован;
- лучший практический вариант — не строить на нем весь UI, а использовать его точечно внутри общего `Mantine`-shell.

### `Chakra UI`

Open-source часть `Chakra UI` распространяется под `MIT`.

Это хороший, понятный и дружелюбный toolkit, но для данного проекта он слабее как default, потому что обычно менее убедителен именно на data-dense internal/trading screens, чем связка `Mantine` плюс отдельные специализированные grid/chart libraries.

## 6. Charting options

### `lightweight-charts`

По официальному сайту TradingView:

- библиотека open-source;
- лицензия `Apache 2.0`;
- ориентирована на financial charts;
- поддерживает realtime updates и большие массивы баров.

Это лучший default для:

- candles, bars, line, volume;
- intraday price views;
- spread/time-series charts;
- компактных trading panels.

Это не универсальная dashboard-библиотека. Ее не стоит делать единственным charting слоем для всей платформы.

### `Apache ECharts`

По официальному сайту:

- лицензия `Apache 2.0`;
- 20+ типов графиков;
- поддержка Canvas/SVG;
- есть progressive rendering и stream loading для очень больших наборов данных.

Это лучший default для:

- risk dashboards;
- exposure/correlation heatmaps;
- scatter/regime/distribution views;
- operational and research analytics panels.

### `Plotly.js`

По официальной документации Plotly:

- `Plotly.js` бесплатен;
- лицензия `MIT`;
- не требует регистрации, токенов или внешнего сервиса;
- может работать offline.

Практический вывод:

- хороший инструмент для ad hoc analytical visualizations и научно-исследовательских экранов;
- но как основной runtime chart stack для ops/trading UI обычно тяжелее и менее pragmatic, чем связка `lightweight-charts` + `ECharts`.

### `@mantine/charts`

По документации Mantine пакет основан на `Recharts`.

Это полезный слой для:

- KPI cards;
- простых line/bar/area charts;
- быстрых административных экранов.

Но его не стоит делать основным charting engine для trading-style market views.

## 7. Практическая рекомендация на текущий этап

Зафиксировать следующий базовый выбор:

1. `Vite + React + TypeScript`
2. `Mantine` как основной UI-kit и application shell
3. `React Router` для доменного разделения `Research UI`, `Ops/Trading UI`, `Risk UI`
4. `TanStack Query` + `Zod` для API boundary и server state
5. `lightweight-charts` для рыночных и execution-aware time-series графиков
6. `Apache ECharts` для risk/research/ops analytics
7. `AG Grid Community` только там, где действительно нужен heavy grid

Это лучший баланс между:

- бесплатностью;
- permissive лицензиями;
- AI-agent friendliness;
- сопровождением человеком без сильной frontend-специализации;
- пригодностью для data-dense trading platform UI.

`MUI` не запрещен, но не рекомендуется как базовый выбор для этого проекта именно из-за open-core границы в `MUI X`, которая плохо сочетается с требованием на полностью бесплатный стек.
