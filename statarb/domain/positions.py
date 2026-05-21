from __future__ import annotations

"""Canonical target, position, and account contracts."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from statarb.domain.events import (
    ExecutionContour,
    SerializableContract,
    _coerce_decimal,
    _require_aware_datetime,
    _require_non_empty,
)
from statarb.domain.instruments import InstrumentRef


@dataclass(frozen=True, slots=True)
class TargetPosition(SerializableContract):
    target_position_id: str
    instrument: InstrumentRef
    target_quantity: Decimal
    as_of: datetime
    signal_intent_id: str | None = None
    research_run_id: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_position_id",
            _require_non_empty(self.target_position_id, "target_position_id"),
        )
        if not isinstance(self.instrument, InstrumentRef):
            raise ValueError("instrument must be an InstrumentRef instance.")

        object.__setattr__(
            self,
            "target_quantity",
            _coerce_decimal(self.target_quantity, "target_quantity"),
        )
        object.__setattr__(self, "as_of", _require_aware_datetime(self.as_of, "as_of"))

        for field_name in ("signal_intent_id", "research_run_id", "reason"):
            value = getattr(self, field_name)
            if value is not None:
                normalized = value.strip()
                object.__setattr__(self, field_name, normalized or None)


@dataclass(frozen=True, slots=True)
class PositionSnapshot(SerializableContract):
    snapshot_id: str
    instrument: InstrumentRef
    net_quantity: Decimal
    captured_at: datetime
    contour: ExecutionContour
    average_price: Decimal | None = None
    mark_price: Decimal | None = None
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_id", _require_non_empty(self.snapshot_id, "snapshot_id"))
        if not isinstance(self.instrument, InstrumentRef):
            raise ValueError("instrument must be an InstrumentRef instance.")

        object.__setattr__(self, "net_quantity", _coerce_decimal(self.net_quantity, "net_quantity"))
        object.__setattr__(
            self,
            "captured_at",
            _require_aware_datetime(self.captured_at, "captured_at"),
        )
        object.__setattr__(self, "contour", ExecutionContour.coerce(self.contour))

        for field_name in ("average_price", "mark_price"):
            value = getattr(self, field_name)
            if value is not None:
                decimal_value = _coerce_decimal(value, field_name)
                if decimal_value <= Decimal("0"):
                    raise ValueError(f"{field_name} must be greater than 0.")
                object.__setattr__(self, field_name, decimal_value)

        object.__setattr__(self, "realized_pnl", _coerce_decimal(self.realized_pnl, "realized_pnl"))
        object.__setattr__(
            self,
            "unrealized_pnl",
            _coerce_decimal(self.unrealized_pnl, "unrealized_pnl"),
        )


@dataclass(frozen=True, slots=True)
class AccountSummary(SerializableContract):
    summary_id: str
    account_id: str
    equity: Decimal
    available_cash: Decimal
    captured_at: datetime
    contour: ExecutionContour
    buying_power: Decimal | None = None
    margin_used: Decimal | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary_id", _require_non_empty(self.summary_id, "summary_id"))
        object.__setattr__(self, "account_id", _require_non_empty(self.account_id, "account_id"))
        object.__setattr__(self, "equity", _coerce_decimal(self.equity, "equity"))
        object.__setattr__(
            self,
            "available_cash",
            _coerce_decimal(self.available_cash, "available_cash"),
        )
        object.__setattr__(
            self,
            "captured_at",
            _require_aware_datetime(self.captured_at, "captured_at"),
        )
        object.__setattr__(self, "contour", ExecutionContour.coerce(self.contour))

        for field_name in ("buying_power", "margin_used"):
            value = getattr(self, field_name)
            if value is not None:
                decimal_value = _coerce_decimal(value, field_name)
                if decimal_value < Decimal("0"):
                    raise ValueError(f"{field_name} must be greater than or equal to 0.")
                object.__setattr__(self, field_name, decimal_value)


__all__ = ["AccountSummary", "PositionSnapshot", "TargetPosition"]
