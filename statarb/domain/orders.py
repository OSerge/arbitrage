from __future__ import annotations

"""Canonical order and fill contracts."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from statarb.domain.events import (
    ExecutionContour,
    SerializableContract,
    _coerce_decimal,
    _require_aware_datetime,
    _require_non_empty,
)
from statarb.domain.instruments import InstrumentRef


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"

    @classmethod
    def coerce(cls, value: "OrderSide | str") -> "OrderSide":
        if isinstance(value, cls):
            return value

        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member

        allowed = ", ".join(member.value for member in cls)
        raise ValueError(f"Unsupported order side: {value!r}. Expected one of: {allowed}.")


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"

    @classmethod
    def coerce(cls, value: "OrderType | str") -> "OrderType":
        if isinstance(value, cls):
            return value

        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member

        allowed = ", ".join(member.value for member in cls)
        raise ValueError(f"Unsupported order type: {value!r}. Expected one of: {allowed}.")


class OrderStatus(StrEnum):
    CREATED = "created"
    ACCEPTED = "accepted"
    WORKING = "working"
    PARTIALLY_FILLED = "partially-filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"

    @classmethod
    def coerce(cls, value: "OrderStatus | str") -> "OrderStatus":
        if isinstance(value, cls):
            return value

        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member

        allowed = ", ".join(member.value for member in cls)
        raise ValueError(f"Unsupported order status: {value!r}. Expected one of: {allowed}.")


@dataclass(frozen=True, slots=True)
class OrderEvent(SerializableContract):
    event_id: str
    order_id: str
    instrument: InstrumentRef
    order_type: OrderType
    side: OrderSide
    quantity: Decimal
    status: OrderStatus
    occurred_at: datetime
    contour: ExecutionContour
    limit_price: Decimal | None = None
    broker_order_id: str | None = None
    reason: str | None = None
    target_position_id: str | None = None
    signal_intent_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _require_non_empty(self.event_id, "event_id"))
        object.__setattr__(self, "order_id", _require_non_empty(self.order_id, "order_id"))
        if not isinstance(self.instrument, InstrumentRef):
            raise ValueError("instrument must be an InstrumentRef instance.")

        object.__setattr__(self, "order_type", OrderType.coerce(self.order_type))
        object.__setattr__(self, "side", OrderSide.coerce(self.side))
        quantity = _coerce_decimal(self.quantity, "quantity")
        if quantity <= Decimal("0"):
            raise ValueError("quantity must be greater than 0.")
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "status", OrderStatus.coerce(self.status))
        object.__setattr__(
            self,
            "occurred_at",
            _require_aware_datetime(self.occurred_at, "occurred_at"),
        )
        object.__setattr__(self, "contour", ExecutionContour.coerce(self.contour))

        if self.limit_price is not None:
            limit_price = _coerce_decimal(self.limit_price, "limit_price")
            if limit_price <= Decimal("0"):
                raise ValueError("limit_price must be greater than 0.")
            object.__setattr__(self, "limit_price", limit_price)

        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit_price is required for limit orders.")

        for field_name in ("broker_order_id", "reason", "target_position_id", "signal_intent_id"):
            value = getattr(self, field_name)
            if value is not None:
                normalized = value.strip()
                object.__setattr__(self, field_name, normalized or None)


@dataclass(frozen=True, slots=True)
class FillEvent(SerializableContract):
    event_id: str
    fill_id: str
    order_id: str
    instrument: InstrumentRef
    side: OrderSide
    fill_quantity: Decimal
    fill_price: Decimal
    occurred_at: datetime
    contour: ExecutionContour
    commission: Decimal = Decimal("0")
    broker_trade_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _require_non_empty(self.event_id, "event_id"))
        object.__setattr__(self, "fill_id", _require_non_empty(self.fill_id, "fill_id"))
        object.__setattr__(self, "order_id", _require_non_empty(self.order_id, "order_id"))
        if not isinstance(self.instrument, InstrumentRef):
            raise ValueError("instrument must be an InstrumentRef instance.")

        object.__setattr__(self, "side", OrderSide.coerce(self.side))
        fill_quantity = _coerce_decimal(self.fill_quantity, "fill_quantity")
        if fill_quantity <= Decimal("0"):
            raise ValueError("fill_quantity must be greater than 0.")
        object.__setattr__(self, "fill_quantity", fill_quantity)

        fill_price = _coerce_decimal(self.fill_price, "fill_price")
        if fill_price <= Decimal("0"):
            raise ValueError("fill_price must be greater than 0.")
        object.__setattr__(self, "fill_price", fill_price)

        commission = _coerce_decimal(self.commission, "commission")
        if commission < Decimal("0"):
            raise ValueError("commission must be greater than or equal to 0.")
        object.__setattr__(self, "commission", commission)

        object.__setattr__(
            self,
            "occurred_at",
            _require_aware_datetime(self.occurred_at, "occurred_at"),
        )
        object.__setattr__(self, "contour", ExecutionContour.coerce(self.contour))

        if self.broker_trade_id is not None:
            broker_trade_id = self.broker_trade_id.strip()
            object.__setattr__(self, "broker_trade_id", broker_trade_id or None)


__all__ = [
    "FillEvent",
    "OrderEvent",
    "OrderSide",
    "OrderStatus",
    "OrderType",
]
