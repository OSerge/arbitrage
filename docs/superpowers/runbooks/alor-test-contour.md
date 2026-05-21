# `AlorAPI` test contour runbook

- Статус: рабочая опорная инструкция для MVP
- Дата: 2026-05-21

## Назначение

Этот runbook фиксирует практические правила эксплуатации `AlorAPI` test contour в рамках `Phase 1: Agent Operating System foundation` и следующей волны интеграции адаптера. Документ не включает live enablement и не переопределяет implementation plan; его задача - перевести уже согласованные выводы в повторяемые operational шаги.

## Scope и границы решений

- Для MVP безопасным рабочим контуром считается только `test`.
- Секреты на текущем этапе допустимо хранить в `.env`.
- Переключение `test`/`live` должно делаться конфигурацией, а не редактированием кода.
- Любой шаг, который выходит за пределы read-only semantics или приближает систему к реальной отправке команд, требует отдельного пользовательского approval.

## Base URLs и contour map

### Test contour

- Auth host: `https://oauthdev.alor.ru`
- API host: `https://apidev.alor.ru`
- Market-data and portfolio subscriptions: `wss://apidev.alor.ru/ws`
- Order command channel: `wss://apidev.alor.ru/cws`

### Live contour

- Контур изолирован от `test`; токены, портфели и данные между контурами невзаимозаменяемы.
- Для live используются live-hosts `Alor`, но этот runbook осознанно описывает только safe test contour.
- Любой переход к `live` остается отдельным gate и не должен происходить "по умолчанию" после успешного тестового smoke-check.

## Соглашения по переменным окружения

Для первой интеграционной проверки runbook исходит из уже встречающихся в репозитории имен переменных:

```dotenv
STATARB_PROFILE=local-dev
ALOR_CONTOUR=test
ALOR_TEST_REFRESH_TOKEN=replace-me
ALOR_TEST_PORTFOLIO=replace-me
```

Дополнительно в репозитории уже существуют live-oriented имена `ALOR_LIVE_REFRESH_TOKEN` и `ALOR_LIVE_PORTFOLIO`, но они не должны использоваться для test contour smoke-check. Доступ к live по-прежнему отделен от test и требует отдельного enablement.

Операционное правило:

- хранить в `.env` только `Refresh Token`;
- `Access Token` получать и ротировать runtime-менеджером;
- не прошивать access token в код, docs или постоянные конфиги.

## Auth lifecycle

### 1. Refresh token

- В test contour используется JWT flow, а не third-party OAuth provider flow из live-кейсов.
- `Refresh Token` живет `1 месяц`.
- Для test account одновременно доступен только один активный `Refresh Token`.

Практическое следствие: если пользователь перевыпускает refresh token, предыдущий токен нужно считать недействительным и обновить `.env` до следующего запуска агента или runtime.

### 2. Access token

- `Access Token` живет `30 минут`.
- Он получается через auth host test contour и используется для HTTP и WebSocket операций.
- Жизненный цикл access token должен управляться отдельным `token refresh manager`.

Рекомендуемая политика для MVP:

- обновлять access token заранее, а не по факту ошибки;
- использовать refresh threshold не позже чем за `5 минут` до ожидаемого истечения;
- логировать время получения, расчетный `expires_at` и факт ротации.

### 3. `/ws` lifecycle

- `wss://apidev.alor.ru/ws` используется для read-only subscriptions.
- Токен передается в каждом subscription request.
- После reconnect клиент должен пересобрать соединение, получить свежий access token при необходимости и выполнить idempotent resubscribe из локального registry подписок.

### 4. `/cws` lifecycle

- `wss://apidev.alor.ru/cws` используется для order commands.
- После открытия соединения требуется отдельная команда `authorize` с валидным access token.
- Сервер не присылает proactive warning об истечении токена.
- При ошибке авторизации `cws` соединение закрывается сервером.

Рекомендуемая политика для MVP:

- держать `cws` в отдельном клиенте от `ws`;
- не смешивать command path и read-only subscriptions;
- при плановой ротации токена предпочитать controlled reconnect + repeat `authorize`, а не надеяться на "прозрачное" продление внутри уже живущего command session;
- не поднимать `cws` вообще, пока текущая задача ограничена read-only smoke-check.

## Ограничения и лимиты

- HTTP API: не более `100 req/s` на пользователя.
- WebSocket: не более `10` соединений одновременно.
- Не более `5000` подписок на сессию.
- Не более `5000` необработанных сообщений в серверном буфере.
- Для order-management документация рекомендует одно `cws`-соединение на поток command execution.
- В пределах одного `cws`-соединения каждый `guid` обязан быть уникальным.
- Повторный `guid` может привести к ошибке `400` и разрыву соединения.
- HTTP годится для initial load, справочников и historical data, но не для real-time polling.

Следствие для MVP: не нужно проектировать много соединений. Практический безопасный старт - `1` менеджер токенов, `1` market-data `ws`, `1` portfolio/lifecycle `ws`, и `0..1` `cws` только при отдельном approval.

## Test vs live: что реально отличается

- `Test` и `live` архитектурно похожи, но изолированы.
- В `test` используются симуляционные, а не реальные биржевые данные.
- Расписание `test` не обязано совпадать с реальным биржевым расписанием.
- Торги в `test` могут прерываться техническими работами.
- Состояние тестового счета и операций может обнуляться при перезапусках торгов.
- Успешность `test` contour подтверждает integration semantics и operator workflow, но не подтверждает production-grade latency, market microstructure quality или PnL realism.

Операционное правило: результаты smoke-check в test contour нельзя интерпретировать как доказательство готовности к live trading.

## Рекомендуемая operational topology для MVP

### `token refresh manager`

Отдельный компонент или сервис, который:

- читает `Refresh Token` из конфигурации;
- получает и кеширует `Access Token`;
- отдает текущий access token HTTP/WS/CWS клиентам;
- ротирует токен проактивно;
- уведомляет connection supervisor о stale token.

### `ws` clients

Рекомендуется разнести минимум на два логических read-only клиента:

- `ws_market_data` для `QuotesSubscribe` и `OrderBookGetAndSubscribe`;
- `ws_portfolio` для `PositionsGetAndSubscribeV2`, `SummariesGetAndSubscribeV2`, `OrdersGetAndSubscribeV2`, `TradesGetAndSubscribe`.

Причина разбиения:

- рыночные данные и portfolio lifecycle имеют разную плотность сообщений и разный recovery priority;
- портфельные события легче переподнимать отдельно от market-data;
- следующей волне адаптера будет проще изолировать mapper и replay envelopes.

### `cws` client

- Использовать отдельный `cws_orders` клиент только для command path.
- Для MVP держать одно `cws`-соединение на весь поток команд.
- Не создавать command client в тех проверках, где достаточно read-only semantics.

## Reconnect и resubscribe discipline

### Обязательные правила

- Любое соединение поднимается через локальный `connection supervisor`.
- Все подписки регистрируются в локальном in-memory registry с достаточными данными для повторной отправки.
- Reconnect выполняется с exponential backoff и jitter.
- Перед reconnect нужно проверить срок жизни текущего access token; при малом остатке токен обновляется до пересоздания сокета.
- После reconnect подписки восстанавливаются только из registry, а не из случайного состояния в runtime.
- Obsolete subscriptions нужно явно удалять из registry, чтобы не копить мусор и не приближаться к лимиту `5000`.

### Правила для `guid` и command idempotency

- `guid` для `cws` должен генерироваться клиентом и не переиспользоваться.
- Безопаснее считать `guid` глобально уникальным, а не только уникальным "в пределах текущей идеи".
- После reconnect нужно создавать новые command identifiers, а не повторять старые значения вслепую.
- Любой order-related retry должен быть привязан к явной локальной политике idempotency, а не к надежде, что `Alor` сам распознает дубль.

## Safe scope для первой интеграционной проверки

Разрешенный scope без дополнительного approval:

- проверка загрузки `.env` и contour selection;
- refresh -> access token exchange;
- read-only HTTP запросы для справочников, initial load или historical data;
- read-only `ws` subscriptions на market-data и portfolio/order lifecycle events;
- проверка reconnect/resubscribe логики на test contour.

Вне safe scope и требует отдельного explicit approval:

- любые `cws` smoke-check, которые затрагивают command path;
- попытки приблизить адаптер к live-ready execution behavior;
- любые действия в `live` contour.

## Read-only runner для первой проверки

В репозитории есть manual helper:

```bash
uv run python -m statarb.runtime.alor_test_read_only plan --env-file .env
```

Что делает `plan`:

- читает `ALOR_CONTOUR`, `ALOR_TEST_REFRESH_TOKEN` и `ALOR_TEST_PORTFOLIO` из `.env`;
- валидирует, что выбран именно `test` contour;
- печатает redacted auth/http/ws plan без сетевых вызовов и без `cws`.

Реальный read-only smoke после заполнения `.env`:

```bash
uv run python -m statarb.runtime.alor_test_read_only smoke --env-file .env
```

Что делает `smoke`:

- выполняет `refresh -> access token` exchange;
- делает только read-only HTTP запросы `positions`, `summary`, `orders`, `trades`;
- печатает HTTP payloads и redacted `ws` subscription envelopes для следующего ручного шага;
- не поднимает `cws` и не содержит order placement / cancel surfaces.

Если на первой проверке достаточно только базового account context, можно сузить scope:

```bash
uv run python -m statarb.runtime.alor_test_read_only smoke --env-file .env --skip-orders --skip-trades
```

## Что пользователь должен подготовить перед первой реальной проверкой

Минимальный набор:

1. Валидный test contour `Refresh Token`.
2. Идентификатор test portfolio для этого токена.
3. Подтверждение, что `.env` допустим как временное хранилище секрета на этапе MVP.
4. Окно времени, когда test contour доступен и не находится на технических работах.
5. Небольшой список инструментов для smoke-check read-only подписок.

Практически это означает:

- заполнить `.env` значениями `ALOR_CONTOUR=test`, `ALOR_TEST_REFRESH_TOKEN=...`, `ALOR_TEST_PORTFOLIO=...`;
- помнить, что выпуск нового refresh token инвалидирует предыдущий;
- заранее определить 1-3 инструмента для проверки market-data;
- не ожидать совпадения test behavior с реальным биржевым днем;
- отдельно дать approval, если следующая сессия должна трогать `cws` command skeleton дальше чисто структурной реализации.

## Checklist для следующей волны адаптера

- Зафиксировать endpoint constants для `oauthdev`, `apidev`, `/ws`, `/cws`.
- Реализовать `token refresh manager` как отдельный слой, а не utility function.
- Развести `ws_market_data`, `ws_portfolio` и `cws_orders` по разным client responsibilities.
- Завести subscription registry и reconnect supervisor до первых smoke-check.
- Считать test contour источником integration semantics, но не источником истины для live-quality оценок.

## Источники

- `docs/superpowers/plans/2026-05-21-agent-operated-mvp-implementation-plan.md`
- `https://alor.dev/docs/api/access/environments`
- `https://alor.dev/docs/en/api/access/authorization/dev-env-auth`
- `https://alor.dev/docs/en/api/usage/limits-and-recommendations`
- `https://alor.dev/docs/en/api/websocket/data-subscriptions/overview`
- `https://alor.dev/docs/en/api/usage/orders/overview`
