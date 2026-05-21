# `AlorAPI` Open Market Data Contract

- Статус: additive clarification
- Дата: 2026-05-21

## Назначение

Этот документ фиксирует практический MVP-контракт для открытых `AlorAPI` market-data surfaces, доступных без токена, и рекомендуемую схему хранения данных. Его задача - дать `docs-first` опору для следующего ingestion slice без дрейфа в сторону старых `CSV`-представлений и без преждевременной инфраструктурной тяжести.

## Change type

- additive

## Affected artifacts

- `docs/superpowers/contracts/alor-market-data.md`
- `docs/superpowers/contracts/README.md`
- `docs/README.md`
- `statarb/adapters/alor/http_market_data.py`
- `tests/adapters/alor/test_http_market_data.py`

## Compatibility

- текущая архитектурная ставка на `Parquet` как основной dataset format остается валидной;
- `Postgres` по-прежнему не требуется как хранилище самих market-data artifacts;
- новый документ не меняет runtime boundaries и не расширяет live scope;
- public/open `Alor` data трактуется как отдельный ingestion surface, совместимый с будущим authorized contour.

## Required approval

- none

## Официальные факты, на которых основан контракт

### 1. Public path для MVP - это `HTTP`, а не `WebSocket`

- `HTTP API` допускает часть публичных запросов без авторизации.
- Для анонимных запросов `Alor` возвращает данные с задержкой `15 минут`.
- `WebSocket API` без авторизации использовать нельзя; все запросы к `/ws` и `/cws` требуют валидный `Access Token`.
- `GraphQL API` также допускает public resources без обязательной авторизации, но для MVP этот slice фиксирует именно `HTTP` как стартовый ingestion path: он проще, воспроизводимее и лучше подходит для batch research bootstrap.

Практический вывод: если задача ограничена public/historical bootstrap без токенов, канонический MVP-вход - `HTTP GET` к public endpoints с явной фиксацией признака `authorized=false` и `data_delay_minutes=15`.

### 2. Public/open endpoints, полезные для MVP

#### Канонические public endpoints

- `GET /md/v2/history`
  - public;
  - historical candles;
  - подходит для batch bootstrap исторических баров;
  - anonymous mode возвращает данные с задержкой `15 минут`.
- `GET /md/v2/Securities`
  - public;
  - поиск и выгрузка instrument metadata;
  - полезен как публичный symbol-master snapshot.
- `GET /md/v2/Securities/{symbols}/quotes`
  - public;
  - snapshot quotes для списка `exchange:symbol` пар;
  - подходит для lightweight snapshot enrichment, но не для real-time feed.
- `GET /md/v2/Securities/{exchange}/{symbol}/availableBoards`
  - public;
  - список доступных бордов для инструмента;
  - нужен для нормализации `board`/`primary board` semantics.
- `GET /md/v2/Securities/currencyPairs`
  - public;
  - вспомогательный reference surface.

#### Публичные, но вторичные для текущего slice

- `GET /md/v2/time`
- `GET /md/v2/risk/rates`

Они могут быть полезны для diagnostic/reference use cases, но не являются primary storage surface для текущего MVP-slice.

#### Не public и не должны считаться open canonical source

- `GET /md/v2/Securities/{exchange}`
- `GET /md/v2/Securities/{exchange}/{symbol}`
- `GET /md/v2/orderbooks/{exchange}/{symbol}`
- `GET /md/v2/Securities/{exchange}/{symbol}/alltrades`
- `GET /md/v2/Securities/{exchange}/{symbol}/alltrades/history`
- любые portfolio/order/trade endpoints под `/md/v2/Clients/...`
- все `WebSocket` subscriptions и `cws` commands

Практический вывод: public MVP-ingestion нельзя проектировать как будто `order book`, all-trades и detailed single-security endpoint доступны анонимно. Для open bootstrap доступны прежде всего `history`, `Securities`, `quotes`, `availableBoards`.

## Payload formats

### Общая договоренность по `format`

`Alor` возвращает JSON-объекты в одном из трех форматов:

- `Simple` - legacy/original shape;
- `Slim` - transport-optimized short-key shape;
- `Heavy` - полный, расширяемый shape.

### Рекомендация для MVP

- для хранения и дальнейшей нормализации использовать `format=Heavy`, когда endpoint это поддерживает;
- `Simple` и `Slim` считать совместимыми input-shapes, которые ingestion shell обязан уметь распознать, но не считать preferred storage shape;
- не использовать `Slim` как canonical persisted representation: он оптимизирован под передачу, а не под читаемость и schema governance.

## Contract: Historical bars

### Official response shape

`GET /md/v2/history` возвращает object envelope:

- `history` или `h` - массив баров;
- `next` - timestamp начала следующего бара;
- `prev` - timestamp начала предыдущего бара.

Элементы бара содержат:

- `time` / `t`
- `open` / `o`
- `high` / `h`
- `low` / `l`
- `close` / `c`
- `volume` / `v`

### Primary MVP fields

Нормализованный bar record должен явно хранить:

- `source = "alor"`
- `endpoint = "md/v2/history"`
- `authorized`
- `data_delay_minutes`
- `response_format`
- `fetched_at_utc`
- `exchange`
- `symbol`
- `board`
- `timeframe`
- `bar_start_utc`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `split_adjust`
- `untraded`

### Caveats

- `board` не приходит внутри bar payload; для `history` это query-level metadata из `instrumentGroup`, поэтому его нужно сохранять отдельно от raw bar rows.
- `open interest` в `history` candles не возвращается.
- `from` и `to` задаются в `UTC Unix Time Seconds`.
- `countBack` может сдвигать фактическую левую границу выборки, поэтому реальный диапазон данных нельзя восстанавливать только из исходного `from`.

## Contract: Instrument snapshot

### Official response shape

`GET /md/v2/Securities` возвращает array of instrument objects. В `Heavy`/`Simple` shape для MVP важны:

- `symbol`
- `exchange`
- `market`
- `shortName` / `shortname`
- `description`
- `board`
- `primaryBoard` / `primary_board`
- `type`
- `lotSize` / `lotsize`
- `minStep` / `minstep`
- `faceValue` / `facevalue`
- `currency`
- `ISIN`
- `tradingStatus`
- `tradingStatusInfo`

### Primary MVP fields

Нормализованный instrument snapshot должен хранить:

- `source = "alor"`
- `endpoint = "md/v2/Securities"`
- `authorized`
- `data_delay_minutes`
- `response_format`
- `fetched_at_utc`
- `exchange`
- `symbol`
- `board`
- `primary_board`
- `market`
- `instrument_type`
- `short_name`
- `description`
- `lot_size`
- `min_step`
- `face_value`
- `currency`
- `isin`
- `trading_status`
- `trading_status_info`

### Caveats

- публичный `/md/v2/Securities` - более полезный стартовый symbol-master source, чем auth-only `/{exchange}/{symbol}`;
- `includeNonBaseBoards=true` полезен для полного snapshot-а board variants, но без него можно потерять часть торговых режимов.

## Contract: Quote snapshot

### Official response shape

`GET /md/v2/Securities/{symbols}/quotes` возвращает array of quote objects. Для MVP важны:

- `symbol`
- `exchange`
- `lastPriceTimestamp` / `last_price_timestamp` / `tst`
- `openPrice` / `open_price` / `o`
- `highPrice` / `high_price` / `h`
- `lowPrice` / `low_price` / `l`
- `lastPrice` / `last_price` / `c`
- `volume` / `v`
- `openInterest` / `open_interest` / `oi`
- `bid`
- `ask`
- `bidVol` / `bid_vol` / `bv`
- `askVol` / `ask_vol` / `av`
- `lotSize` / `lotsize` / `lot`
- `type`

### Primary MVP fields

Нормализованный quote snapshot должен хранить:

- `source = "alor"`
- `endpoint = "md/v2/Securities/{symbols}/quotes"`
- `authorized`
- `data_delay_minutes`
- `response_format`
- `fetched_at_utc`
- `exchange`
- `symbol`
- `quote_time_utc`
- `open`
- `high`
- `low`
- `last`
- `volume`
- `open_interest`
- `bid`
- `ask`
- `bid_volume`
- `ask_volume`
- `lot_size`
- `instrument_type`

### Caveats

- `board` не приходит в quote payload; его нужно получать либо из symbol-master snapshot, либо из query context другого source.
- anonymous quotes - это delayed snapshot, а не source of truth для live-grade intraday decisions.

## Recommended storage design for MVP

### Short answer

Оптимальный стартовый формат хранения данных от `AlorAPI` для MVP:

1. exact raw response в `json.gz`;
2. normalized analytical datasets в `Parquet`;
3. file-backed `manifest.json` на каждый dataset snapshot/load batch.

### Почему именно так

- `Parquet` уже зафиксирован в approved MVP docs и ADR как основной формат dataset/data artifacts.
- Raw `json.gz` сохраняет vendor payload без потери shape, что важно для replay, schema drift debugging и повторной нормализации.
- `manifest.json` закрывает lineage и воспроизводимость без необходимости сразу поднимать тяжелый catalog layer.
- Такой набор естественно совместим с будущим `MinIO/Parquet` lakehouse path и не тащит premature infrastructure в текущий slice.

### Почему не `CSV`

- `CSV` не выражает надежно vendor schema variants `Simple/Slim/Heavy`;
- `CSV` плохо фиксирует request-level metadata (`authorized`, `format`, `instrumentGroup`, `splitAdjust`, `countBack`);
- `CSV` неудобен для schema evolution и для хранения mixed reference/snapshot payloads.

### Почему не `JSONL` как primary analytical format

- `JSONL` полезен как transport/debug representation, но слабее `Parquet` для batch research scans;
- у `Parquet` лучше типизация чисел и timestamps;
- `Parquet` лучше подходит под column pruning, predicate pushdown и будущий ClickHouse/lakehouse путь.

### Layout recommendation

#### Raw zone

Raw layer должна хранить exact response body и request envelope:

```text
data/raw/vendor=alor/dataset=bars/load_date=YYYY-MM-DD/request_id=<id>/response.json.gz
data/raw/vendor=alor/dataset=instruments/load_date=YYYY-MM-DD/request_id=<id>/response.json.gz
data/raw/vendor=alor/dataset=quotes/load_date=YYYY-MM-DD/request_id=<id>/response.json.gz
```

Рядом допустим `manifest.json` или `request.json` с:

- endpoint;
- base_url;
- query params;
- fetched_at_utc;
- authorized;
- data_delay_minutes;
- response_format;
- sha256;
- row_count;
- source_doc_version.

#### Normalized zone

`Parquet` должен быть primary read path:

```text
data/normalized/vendor=alor/dataset=bars/exchange=MOEX/timeframe=60/date=YYYY-MM-DD/part-*.parquet
data/normalized/vendor=alor/dataset=instruments/as_of_date=YYYY-MM-DD/exchange=MOEX/part-*.parquet
data/normalized/vendor=alor/dataset=quotes/as_of_date=YYYY-MM-DD/exchange=MOEX/part-*.parquet
```

### Partitioning rule

- не партиционировать по `symbol` на старте;
- хранить `symbol`, `board`, `primary_board` как колонки;
- использовать coarse partitions по `dataset`, `exchange`, `timeframe`, `date/as_of_date`.

Причина: symbol-level partitioning на раннем этапе быстро создает small-files problem и мешает batch research.

### Manifest rule

Каждый normalized dataset batch обязан иметь manifest с:

- `dataset_id`
- `dataset_kind`
- `storage_uri`
- `raw_storage_uri`
- `source = "alor"`
- `authorized`
- `data_delay_minutes`
- `response_format`
- `request_params`
- `row_count`
- `min_event_time_utc`
- `max_event_time_utc`
- `created_at_utc`

## MVP ingestion conventions

- Для public bootstrap по умолчанию использовать `authorized=false`.
- Для canonical snapshots запрашивать `format=Heavy`.
- Для bars всегда сохранять `instrumentGroup`/`board` в manifest, даже если payload его не содержит.
- Quote datasets считать snapshot-слоем, а не replacement для bar history.
- Instrument snapshot и `availableBoards` нужно хранить отдельно, чтобы последующий ingestion мог надежно enrich-ить `board` semantics.

## Verification

- docs: this contract plus linked indices
- tests: adapter-level request/shape tests without network access
- intentionally not run: real `Alor` HTTP calls

## Источники

- `https://alor.dev/docs/en/api/http/md-v-2-history-get`
- `https://alor.dev/docs/en/api/usage/faq`
- `https://alor.dev/rawdocs2/WarpOpenAPIv2.yml`
