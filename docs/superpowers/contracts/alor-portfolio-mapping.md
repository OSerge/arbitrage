# `AlorAPI` Portfolio Mapping Contract

- Статус: additive clarification
- Дата: 2026-05-23

## Назначение

Этот документ фиксирует MVP-правила перевода `Alor` portfolio-scoped payloads (`Simple` format) в канонические execution contracts (`OrderEvent`, `FillEvent`, `PositionSnapshot`, `AccountSummary`) без изменения domain shape.

## Change type

- additive

## Affected artifacts

- `docs/superpowers/contracts/alor-portfolio-mapping.md`
- `docs/superpowers/contracts/README.md`
- `statarb/adapters/alor/mapper.py`
- `tests/adapters/alor/test_mapper.py`

## Compatibility

- domain contracts из `domain-execution.md` не меняются;
- `PlaceholderAlorMapper` остаётся для opcode classification без domain mapping;
- `CanonicalAlorMapper` добавляет deterministic mapping для portfolio HTTP/WS surfaces;
- пустые `positions`/`orders`/`trades` arrays считаются валидным mapped результатом с `0` events.

## Required approval

- none

## Supported surfaces

### HTTP (`/md/v2/Clients/{exchange}/{portfolio}/...`)

| surface | payload shape | canonical output |
| --- | --- | --- |
| `positions` | array of position objects | `0..N` `PositionSnapshot` |
| `summary` | single summary object | `1` `AccountSummary` |
| `orders` | array of order objects | `0..N` `OrderEvent` |
| `trades` | array of trade objects | `0..N` `FillEvent` |

### WebSocket (`ws_portfolio` opcodes)

| opcode | payload envelope | canonical output |
| --- | --- | --- |
| `PositionsGetAndSubscribeV2` | `{ "data": position }` | `1` `PositionSnapshot` |
| `SummariesGetAndSubscribeV2` | `{ "data": summary }` | `1` `AccountSummary` |
| `OrdersGetAndSubscribeV2` | `{ "data": order }` | `1` `OrderEvent` |
| `TradesGetAndSubscribe` | `{ "data": trade }` | `1` `FillEvent` |

## Field mapping rules

### Common

- `contour` для test contour mapping: `broker-test`;
- quantity fields: предпочитать `qtyUnits` (instrument units), fallback `qty`;
- timestamps: `updateTime`/`transTime`/`date` в UTC ISO-8601;
- instrument identity: `InstrumentRef(symbol, exchange, instrument_type)`; default `instrument_type=equity`, futures heuristic по шаблону `-` + maturity suffix.

### Order -> `OrderEvent`

- `order_id` и `broker_order_id`: `id`;
- `order_type`: `limit`/`market`;
- `side`: `buy`/`sell`;
- `status`: `working`/`filled`/`canceled`/`rejected`; `working` + `filledQtyUnits > 0` -> `partially-filled`;
- `quantity`: `qtyUnits`;
- `limit_price`: `price` для `limit`, иначе `null`;
- `occurred_at`: `updateTime`, fallback `transTime`.

### Trade -> `FillEvent`

- `fill_id` и `broker_trade_id`: `id`;
- `order_id`: `orderno`;
- `fill_quantity`: `qtyUnits`;
- `fill_price`: `price`;
- `commission`: `commission`, default `0`;
- `occurred_at`: `date`.

### Position -> `PositionSnapshot`

- skip rows with `isCurrency=true` (cash balances не маппятся в equity `PositionSnapshot`);
- `net_quantity`: `qtyUnits`;
- `average_price`: `avgPrice` when present;
- `unrealized_pnl`: `unrealisedPl`, default `0`.

### Summary -> `AccountSummary`

- `account_id`: configured portfolio id from mapping context;
- `equity`: `portfolioEvaluation`;
- `available_cash`: `buyingPower`;
- `buying_power`: `buyingPower`;
- `margin_used`: `initialMargin` when present;
- `captured_at`: explicit mapping context timestamp when summary payload has no event time.

## Non-goals (MVP)

- stop-order surfaces;
- repo-specific trade fields;
- auto-selection portfolio id;
- live contour enablement;
- market-data opcode mapping (`QuotesSubscribe`, etc.).

## Verification sources

- portfolio-scoped read-only smoke на test contour (`D00014`, 2026-05-23): empty `positions`/`orders`/`trades`, non-empty `summary`;
- официальные `Alor` docs/examples для `Simple` order/trade/position payloads.
