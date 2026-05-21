# Российский контур: MOEX, брокеры и дополнительные площадки

## 1. Главный вывод

Для пользователя в российском контуре разумная последовательность такая:

1. `MOEX` как основной рынок для statarb MVP.
2. `Брокерские API` как стартовый execution/data layer.
3. `Прямые MOEX каналы` только после подтверждения стратегии, капитала и требований к latency.
4. `Дополнительные площадки` подключать как phase 2, а не распылять усилия в foundation-фазе.

## 2. Что использовать на MOEX

## 2.1 Данные и reference

### MOEX ISS

Подходит для:

- справочников инструментов;
- исторических баров и reference data;
- bootstrapping symbol master;
- sanity-check и сверки.

Не подходит как основной live-market data слой для serious execution.

### MOEX FAST / прямые каналы market data

Это путь для low-latency и более институционального контура:

- real-time market data;
- высокая скорость доставки;
- пригодно для advanced execution и microstructure-aware моделей.

Использовать имеет смысл, когда:

- уже есть подтвержденный alpha;
- есть требования к минимальному latency jitter;
- готова инфраструктура под direct connectivity и эксплуатацию.

### PLAZA II / Spectra и прямой trading connectivity

Для производного рынка MOEX direct connectivity нужен, если цель — более жесткий execution-контур, чем брокерский REST/WebSocket.

Практический вывод:

- для research и MVP это слишком рано;
- для mature derivatives desk это правильная долгосрочная цель.

## 2.2 Брокерские адаптеры

### 1. Alor

Почему ставить первым:

- в репозитории уже есть интеграция с `AlorAPI`;
- есть понятный старт без смены основного поставщика;
- есть HTTP/WebSocket контур;
- подходит для быстрого перехода к paper/live MVP.

Ограничение:

- это все еще брокерский контур, а не институциональный direct gateway.

### 2. Finam Trade API

Почему полезен как второй адаптер:

- современный API-контур;
- годится как резервный execution/data path;
- полезен для сравнения operational profile, rate limits и reliability.

Роль в платформе:

- резервный брокер;
- дополнительный источник broker events;
- страховка от vendor lock-in.

### 3. T-Invest API

Почему стоит учитывать:

- хорошая документация и SDK;
- годится для быстрого paper/live experimentation;
- полезен как массовый retail-grade канал.

Почему не делать его главным low-latency контуром:

- удобство интеграции не равно институциональной пригодности;
- для statarb execution platform он скорее хороший adapter tier, чем финальная runtime-цель.

### 4. QUIK/QLua как fallback

Подходит только как совместимость с конкретным брокером, если другого API нет.

Почему не стоит делать на нем core:

- operational fragility;
- зависимость от terminal-driven среды;
- хуже воспроизводимость и поддерживаемость;
- неудобно строить institutional-grade replay/risk/audit вокруг GUI-bound ecosystem.

## 3. Приоритеты по инструментам на MOEX

Для первой версии платформы разумнее всего:

- ликвидные фьючерсы на индексы, валюты и commodities;
- затем ликвидные single-stock futures;
- затем cash equities для cross-sectional моделей и factor-neutral overlays;
- затем опционы только после появления полноценного volatility/risk framework.

Почему фьючерсы важны в начале:

- меньше проблем с borrow/short locate;
- проще execution и capital efficiency;
- более прямой путь к intraday mean reversion и spread trading.

## 4. Что учитывать именно для MOEX

- разные торговые сессии и вечерка;
- клиринги и расписание торгов;
- спецификации контрактов, лоты и шаги цены;
- гарантийное обеспечение и изменения риск-параметров;
- roll logic по фьючерсам;
- аукционы и нерыночные режимы;
- особенности ликвидности вокруг новостей, экспираций и клирингов.

Без отдельного session/calendar/reference-data слоя MOEX-контур быстро начнет порождать ложные сигналы и ошибочные execution decisions.

## 5. Phase 2: что реально добавлять после MOEX

## 5.1 СПБ Биржа

Подключать только при понятной доступности инструментов, ликвидности и реальной торговой ценности.

Это не foundation-приоритет, а опциональный second venue.

## 5.2 Crypto venues

Если пользователь допускает отдельный risk/legal perimeter, то как phase 2 имеют смысл:

- liquid spot/perp venues;
- 24/7 market structure для тестирования коротких horizon-моделей;
- cross-venue statarb и market microstructure research.

Но это должен быть отдельный контур:

- отдельные лимиты;
- отдельные compliance assumptions;
- отдельный execution/risk profile;
- без смешивания PnL и риск-моделей с MOEX в первой итерации.

## 5.3 Другие брокерские/биржевые подключения

Подключать только если есть одновременно:

- доступ из российской юрисдикции;
- стабильный API;
- достаточная ликвидность;
- понятный operational/compliance режим.

Если хотя бы один пункт под вопросом, лучше отложить до institutionalization-фазы.

## 6. Практичная стратегия подключения

### Фаза foundation

- `MOEX ISS` для reference/historical bootstrap.
- `Alor` как первый адаптер.

### Фаза execution MVP

- добавить `Finam` или `T-Invest` как второй адаптер;
- унифицировать broker events;
- сделать broker-agnostic OMS и reconciliation.

### Фаза institutionalization

- рассмотреть direct market data и direct trading connectivity;
- вводить colocated/hosted runtime только при доказанной потребности.

## 7. Рекомендация build vs buy по интеграциям

Нужно писать самим:

- adapter abstraction;
- canonical order/fill/position event model;
- reconciliation;
- market session/calendar layer;
- failover policy между брокерами;
- throttling/retry/idempotency logic.

Не нужно писать самим:

- сетевые SDK там, где у брокера есть стабильный официальный интерфейс;
- базовые HTTP/WebSocket клиенты;
- стандартные protobuf/grpc runtime-библиотеки.
