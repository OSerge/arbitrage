from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from statarb.domain.events import ExecutionContour, _coerce_decimal, _require_non_empty
from statarb.domain.instruments import InstrumentRef, InstrumentType
from statarb.domain.orders import FillEvent, OrderEvent, OrderSide, OrderStatus, OrderType
from statarb.domain.positions import AccountSummary, PositionSnapshot


class CanonicalEventKind(StrEnum):
    ORDER_EVENT = "order-event"
    FILL_EVENT = "fill-event"
    POSITION_SNAPSHOT = "position-snapshot"
    ACCOUNT_SUMMARY = "account-summary"
    RAW_BROKER_EVENT = "raw-broker-event"


class MappingStatus(StrEnum):
    PLACEHOLDER = "placeholder"
    DEFERRED = "deferred"
    MAPPED = "mapped"


@dataclass(frozen=True, slots=True)
class CanonicalEventPlaceholder:
    kind: CanonicalEventKind
    source_stream: str
    opcode: str
    payload: Mapping[str, object]


AlorMappedEvent = OrderEvent | FillEvent | PositionSnapshot | AccountSummary | CanonicalEventPlaceholder


@dataclass(frozen=True, slots=True)
class AlorMappingContext:
    portfolio: str
    exchange: str = "MOEX"
    contour: ExecutionContour = ExecutionContour.BROKER_TEST
    captured_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class MappingResult:
    status: MappingStatus
    payload: Mapping[str, object]
    events: tuple[AlorMappedEvent, ...] = ()
    unmapped_reason: str | None = None


@runtime_checkable
class AlorPayloadMapper(Protocol):
    def map_payload(
        self,
        *,
        stream: str,
        opcode: str,
        payload: Mapping[str, object],
        context: AlorMappingContext | None = None,
    ) -> MappingResult: ...


class PlaceholderAlorMapper:
    _CANONICAL_EVENT_KINDS: dict[str, CanonicalEventKind] = {
        "OrdersGetAndSubscribeV2": CanonicalEventKind.ORDER_EVENT,
        "TradesGetAndSubscribe": CanonicalEventKind.FILL_EVENT,
        "PositionsGetAndSubscribeV2": CanonicalEventKind.POSITION_SNAPSHOT,
        "SummariesGetAndSubscribeV2": CanonicalEventKind.ACCOUNT_SUMMARY,
    }

    def map_payload(
        self,
        *,
        stream: str,
        opcode: str,
        payload: Mapping[str, object],
        context: AlorMappingContext | None = None,
    ) -> MappingResult:
        del context
        event_kind = self._CANONICAL_EVENT_KINDS.get(opcode)
        if event_kind is None:
            return MappingResult(
                status=MappingStatus.DEFERRED,
                payload=payload,
                unmapped_reason="No canonical event contract is frozen for this payload yet.",
            )
        placeholder = CanonicalEventPlaceholder(
            kind=event_kind,
            source_stream=stream,
            opcode=opcode,
            payload=payload,
        )
        return MappingResult(
            status=MappingStatus.PLACEHOLDER,
            payload=payload,
            events=(placeholder,),
        )


class CanonicalAlorMapper:
    _WS_OPCODE_HANDLERS: dict[str, str] = {
        "OrdersGetAndSubscribeV2": "order",
        "TradesGetAndSubscribe": "trade",
        "PositionsGetAndSubscribeV2": "position",
        "SummariesGetAndSubscribeV2": "summary",
    }

    def map_payload(
        self,
        *,
        stream: str,
        opcode: str,
        payload: Mapping[str, object],
        context: AlorMappingContext | None = None,
    ) -> MappingResult:
        del stream
        mapping_context = _require_context(context)
        handler = self._WS_OPCODE_HANDLERS.get(opcode)
        if handler is None:
            return MappingResult(
                status=MappingStatus.DEFERRED,
                payload=payload,
                unmapped_reason="No canonical event contract is frozen for this payload yet.",
            )
        item = _unwrap_ws_payload(payload)
        events = self._map_item(handler, item, context=mapping_context)
        return MappingResult(
            status=MappingStatus.MAPPED,
            payload=payload,
            events=events,
        )

    def map_http_surface(
        self,
        *,
        surface: str,
        payload: Any,
        context: AlorMappingContext,
    ) -> MappingResult:
        mapping_context = _require_context(context)
        normalized_surface = surface.strip().lower()
        if normalized_surface == "summary":
            item = _as_mapping(payload, field_name="summary payload")
            events = self._map_item("summary", item, context=mapping_context)
        elif normalized_surface in {"positions", "orders", "trades"}:
            rows = _as_sequence(payload, field_name=f"{normalized_surface} payload")
            handler = normalized_surface.rstrip("s")
            events = tuple(
                event
                for row in rows
                for event in self._map_item(handler, _as_mapping(row, field_name=f"{handler} row"), context=mapping_context)
            )
        else:
            return MappingResult(
                status=MappingStatus.DEFERRED,
                payload=_payload_as_mapping(payload),
                unmapped_reason=f"Unsupported Alor HTTP surface for canonical mapping: {surface!r}.",
            )
        return MappingResult(
            status=MappingStatus.MAPPED,
            payload=_payload_as_mapping(payload),
            events=events,
        )

    def _map_item(
        self,
        kind: str,
        item: Mapping[str, Any],
        *,
        context: AlorMappingContext,
    ) -> tuple[AlorMappedEvent, ...]:
        if kind == "order":
            return (map_alor_order(item, context=context),)
        if kind == "trade":
            return (map_alor_trade(item, context=context),)
        if kind == "position":
            position = map_alor_position(item, context=context)
            return () if position is None else (position,)
        if kind == "summary":
            return (map_alor_summary(item, context=context),)
        raise ValueError(f"Unsupported Alor mapping kind: {kind!r}.")


def map_alor_order(item: Mapping[str, Any], *, context: AlorMappingContext) -> OrderEvent:
    order_id = _required_text(item, "id")
    symbol = _required_text(item, "symbol", "sym")
    exchange = _optional_text(item, "exchange", "ex") or context.exchange
    order_type = _map_order_type(_required_text(item, "type"))
    side = _map_order_side(_required_text(item, "side"))
    quantity = _quantity_units(item)
    status = _map_order_status(item)
    occurred_at = _parse_alor_datetime(
        _optional_text(item, "updateTime", "update_time")
        or _required_text(item, "transTime", "trans_time")
    )
    limit_price = None
    if order_type is OrderType.LIMIT:
        limit_price = _coerce_decimal(_required_value(item, "price"), "limit_price")
    return OrderEvent(
        event_id=f"alor-order:{order_id}:{occurred_at.isoformat()}",
        order_id=order_id,
        instrument=_instrument_ref(symbol, exchange),
        order_type=order_type,
        side=side,
        quantity=quantity,
        status=status,
        occurred_at=occurred_at,
        contour=context.contour,
        limit_price=limit_price,
        broker_order_id=order_id,
    )


def map_alor_trade(item: Mapping[str, Any], *, context: AlorMappingContext) -> FillEvent:
    fill_id = _required_text(item, "id")
    order_id = _required_text(item, "orderno", "orderNo", "order_no")
    symbol = _required_text(item, "symbol", "sym")
    exchange = _optional_text(item, "exchange", "ex") or context.exchange
    side = _map_order_side(_required_text(item, "side"))
    fill_quantity = _quantity_units(item)
    fill_price = _coerce_decimal(_required_value(item, "price"), "fill_price")
    occurred_at = _parse_alor_datetime(_required_text(item, "date"))
    commission = _optional_decimal(item, "commission") or Decimal("0")
    return FillEvent(
        event_id=f"alor-fill:{fill_id}:{occurred_at.isoformat()}",
        fill_id=fill_id,
        order_id=order_id,
        instrument=_instrument_ref(symbol, exchange),
        side=side,
        fill_quantity=fill_quantity,
        fill_price=fill_price,
        occurred_at=occurred_at,
        contour=context.contour,
        commission=commission,
        broker_trade_id=fill_id,
    )


def map_alor_position(item: Mapping[str, Any], *, context: AlorMappingContext) -> PositionSnapshot | None:
    if _optional_bool(item, "isCurrency", "is_currency") is True:
        return None
    symbol = _required_text(item, "symbol", "sym")
    exchange = _optional_text(item, "exchange", "ex") or context.exchange
    captured_at = context.captured_at or datetime.now(UTC)
    net_quantity = _coerce_decimal(_required_value(item, "qtyUnits", "qty_units", "qty"), "net_quantity")
    average_price = _optional_decimal(item, "avgPrice", "avg_price")
    unrealized_pnl = _optional_decimal(item, "unrealisedPl", "unrealised_pl") or Decimal("0")
    return PositionSnapshot(
        snapshot_id=f"alor-position:{context.portfolio}:{exchange}:{symbol}:{captured_at.isoformat()}",
        instrument=_instrument_ref(symbol, exchange),
        net_quantity=net_quantity,
        captured_at=_require_aware(captured_at),
        contour=context.contour,
        average_price=average_price,
        unrealized_pnl=unrealized_pnl,
    )


def map_alor_summary(item: Mapping[str, Any], *, context: AlorMappingContext) -> AccountSummary:
    captured_at = context.captured_at or datetime.now(UTC)
    equity = _coerce_decimal(_required_value(item, "portfolioEvaluation", "portfolio_evaluation"), "equity")
    buying_power = _coerce_decimal(_required_value(item, "buyingPower", "buying_power"), "buying_power")
    margin_used = _optional_decimal(item, "initialMargin", "initial_margin")
    return AccountSummary(
        summary_id=f"alor-summary:{context.portfolio}:{captured_at.isoformat()}",
        account_id=context.portfolio,
        equity=equity,
        available_cash=buying_power,
        captured_at=_require_aware(captured_at),
        contour=context.contour,
        buying_power=buying_power,
        margin_used=margin_used,
    )


def _require_context(context: AlorMappingContext | None) -> AlorMappingContext:
    if context is None:
        raise ValueError("Canonical Alor mapping requires an AlorMappingContext with portfolio.")
    portfolio = context.portfolio.strip()
    if not portfolio:
        raise ValueError("Canonical Alor mapping requires a non-empty portfolio identifier.")
    return AlorMappingContext(
        portfolio=portfolio,
        exchange=context.exchange,
        contour=context.contour,
        captured_at=context.captured_at,
    )


def _unwrap_ws_payload(payload: Mapping[str, object]) -> Mapping[str, Any]:
    data = payload.get("data")
    if isinstance(data, Mapping):
        return dict(data)
    return dict(payload)


def _payload_as_mapping(payload: Any) -> Mapping[str, object]:
    if isinstance(payload, Mapping):
        return payload
    return {"value": payload}


def _as_sequence(payload: Any, *, field_name: str) -> Sequence[Any]:
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return payload
    raise ValueError(f"Alor {field_name} must be a JSON array.")


def _as_mapping(payload: Any, *, field_name: str) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    raise ValueError(f"Alor {field_name} must be a JSON object.")


def _required_value(item: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    joined = ", ".join(keys)
    raise ValueError(f"Alor payload is missing required field: one of {joined}.")


def _required_text(item: Mapping[str, Any], *keys: str) -> str:
    return _require_non_empty(str(_required_value(item, *keys)), keys[0])


def _optional_text(item: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return None


def _optional_decimal(item: Mapping[str, Any], *keys: str) -> Decimal | None:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        return _coerce_decimal(value, key)
    return None


def _optional_bool(item: Mapping[str, Any], *keys: str) -> bool | None:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
        raise ValueError(f"Alor payload field {key!r} must be a boolean-compatible value.")
    return None


def _quantity_units(item: Mapping[str, Any]) -> Decimal:
    if "qtyUnits" in item or "qty_units" in item:
        return _coerce_decimal(_required_value(item, "qtyUnits", "qty_units"), "quantity")
    return _coerce_decimal(_required_value(item, "qty"), "quantity")


def _map_order_type(value: str) -> OrderType:
    normalized = value.strip().lower()
    if normalized == "market":
        return OrderType.MARKET
    if normalized == "limit":
        return OrderType.LIMIT
    raise ValueError(f"Unsupported Alor order type: {value!r}.")


def _map_order_side(value: str) -> OrderSide:
    normalized = value.strip().lower()
    if normalized == "buy":
        return OrderSide.BUY
    if normalized == "sell":
        return OrderSide.SELL
    raise ValueError(f"Unsupported Alor order side: {value!r}.")


def _map_order_status(item: Mapping[str, Any]) -> OrderStatus:
    status = _required_text(item, "status").lower()
    filled_units = _optional_decimal(item, "filledQtyUnits", "filled_qty_units", "filled")
    if status == "working" and filled_units is not None and filled_units > Decimal("0"):
        return OrderStatus.PARTIALLY_FILLED
    mapping = {
        "working": OrderStatus.WORKING,
        "filled": OrderStatus.FILLED,
        "canceled": OrderStatus.CANCELED,
        "cancelled": OrderStatus.CANCELED,
        "rejected": OrderStatus.REJECTED,
    }
    mapped = mapping.get(status)
    if mapped is None:
        raise ValueError(f"Unsupported Alor order status: {status!r}.")
    return mapped


def _instrument_ref(symbol: str, exchange: str) -> InstrumentRef:
    return InstrumentRef(
        symbol=symbol,
        exchange=exchange,
        instrument_type=_infer_instrument_type(symbol),
    )


def _infer_instrument_type(symbol: str) -> InstrumentType:
    normalized = symbol.strip()
    if "-" in normalized and normalized.split("-")[-1][:1].isdigit():
        return InstrumentType.FUTURE
    return InstrumentType.EQUITY


def _parse_alor_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    return _require_aware(parsed)


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "AlorMappedEvent",
    "AlorMappingContext",
    "AlorPayloadMapper",
    "CanonicalAlorMapper",
    "CanonicalEventKind",
    "CanonicalEventPlaceholder",
    "MappingResult",
    "MappingStatus",
    "PlaceholderAlorMapper",
    "map_alor_order",
    "map_alor_position",
    "map_alor_summary",
    "map_alor_trade",
]
