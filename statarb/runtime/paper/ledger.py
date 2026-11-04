"""Shared cash/equity ledger for deterministic paper, backtest and replay."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal

from statarb.domain.events import (
    ExecutionContour,
    _coerce_decimal,
    _require_aware_datetime,
    _require_non_empty,
)
from statarb.domain.instruments import InstrumentRef, InstrumentType
from statarb.domain.orders import FillEvent, OrderSide
from statarb.domain.positions import AccountSummary, PositionSnapshot, PriceMark
from statarb.domain.specifications import InstrumentCatalog


@dataclass(slots=True)
class _LedgerPosition:
    instrument: InstrumentRef
    net_quantity: Decimal = Decimal("0")
    average_price: Decimal | None = None
    realized_pnl: Decimal = Decimal("0")
    last_price: Decimal | None = None
    point_value: Decimal = Decimal("1")


class PaperLedger:
    def __init__(
        self,
        *,
        account_id: str,
        starting_cash,
        contour: ExecutionContour = ExecutionContour.PAPER,
        instrument_catalog: InstrumentCatalog | None = None,
    ):
        self.account_id = _require_non_empty(account_id, "account_id")
        self.starting_cash = _coerce_decimal(starting_cash, "starting_cash")
        if self.starting_cash <= 0:
            raise ValueError("starting_cash must be positive.")
        self.cash_balance = self.starting_cash
        self.contour = ExecutionContour.coerce(contour)
        self.instrument_catalog = instrument_catalog or InstrumentCatalog()
        self._positions: dict[InstrumentRef, _LedgerPosition] = {}
        self._fills: dict[tuple[str, str], dict] = {}
        self._marks: dict[str, dict] = {}
        self._last_event_at: datetime | None = None

    def fork(self) -> PaperLedger:
        candidate = copy(self)
        candidate._positions = {
            instrument: replace(position)
            for instrument, position in self._positions.items()
        }
        # Historical fingerprints are immutable after insertion; copy their indexes,
        # not every nested instrument payload on each quote.
        candidate._fills = self._fills.copy()
        candidate._marks = self._marks.copy()
        return candidate

    def commit_from(self, candidate: PaperLedger):
        if (
            candidate.account_id != self.account_id
            or candidate.starting_cash != self.starting_cash
        ):
            raise ValueError("Cannot commit a different account's ledger.")
        self.cash_balance = candidate.cash_balance
        self._positions = candidate._positions
        self._fills = candidate._fills
        self._marks = candidate._marks
        self._last_event_at = candidate._last_event_at

    @property
    def instruments(self) -> tuple[InstrumentRef, ...]:
        return tuple(self._positions)

    def current_quantity(self, instrument: InstrumentRef) -> Decimal:
        position = self._positions.get(instrument)
        return position.net_quantity if position else Decimal("0")

    def _validate_time(self, timestamp: datetime):
        timestamp = _require_aware_datetime(timestamp, "event timestamp")
        if self._last_event_at is not None and timestamp < self._last_event_at:
            raise ValueError("Ledger events must be chronological.")

    def apply_fill(
        self,
        *,
        fill_event: FillEvent,
        snapshot_id: str,
        captured_at: datetime,
        contour: ExecutionContour | None = None,
    ) -> PositionSnapshot:
        captured_at = _require_aware_datetime(captured_at, "captured_at")
        if captured_at < fill_event.occurred_at:
            raise ValueError("A snapshot cannot precede its fill.")
        key = (fill_event.contour.value, fill_event.fill_id)
        fingerprint = fill_event.to_dict()
        fingerprint.pop("event_id")
        if key in self._fills:
            if self._fills[key] != fingerprint:
                raise ValueError(f"Conflicting duplicate fill: {fill_event.fill_id}")
            return self.build_position_snapshot(
                instrument=fill_event.instrument,
                snapshot_id=snapshot_id,
                captured_at=captured_at,
                contour=contour,
            )
        self._validate_time(fill_event.occurred_at)
        point_value = Decimal("1")
        if fill_event.instrument.instrument_type is InstrumentType.FUTURE:
            spec = self.instrument_catalog.resolve(
                fill_event.instrument, fill_event.occurred_at, trading=True
            )
            point_value = spec.point_value
            if fill_event.fill_quantity != fill_event.fill_quantity.to_integral_value():
                raise ValueError("Futures fills require integer contracts.")
            if fill_event.fill_price % spec.tick_size:
                raise ValueError("Futures fill price must align to tick_size.")
        previous = self._positions.get(fill_event.instrument)
        position = (
            replace(previous)
            if previous
            else _LedgerPosition(fill_event.instrument, point_value=point_value)
        )
        if position.net_quantity and position.point_value != point_value:
            raise ValueError(
                "Changing point value requires a separate settlement model."
            )
        position.point_value = point_value
        delta = fill_event.fill_quantity * (
            1 if fill_event.side is OrderSide.BUY else -1
        )
        realized_before = position.realized_pnl
        self._apply_position_effect(
            position, delta, fill_event.fill_price, fill_event.commission
        )
        if fill_event.instrument.instrument_type is InstrumentType.FUTURE:
            cash_delta = position.realized_pnl - realized_before
        else:
            cash_delta = -delta * fill_event.fill_price - fill_event.commission
        self._positions[fill_event.instrument] = position
        self.cash_balance += cash_delta
        self._fills[key] = fingerprint
        self._last_event_at = fill_event.occurred_at
        return self.build_position_snapshot(
            instrument=fill_event.instrument,
            snapshot_id=snapshot_id,
            captured_at=captured_at,
            contour=contour,
        )

    def apply_mark(self, mark: PriceMark):
        fingerprint = mark.to_dict()
        if mark.event_id in self._marks:
            if self._marks[mark.event_id] != fingerprint:
                raise ValueError("Conflicting duplicate price mark.")
            return
        self._validate_time(mark.occurred_at)
        if mark.instrument.instrument_type is InstrumentType.FUTURE:
            spec = self.instrument_catalog.resolve(mark.instrument, mark.occurred_at)
            position = self._positions.get(mark.instrument)
            if (
                position is not None
                and position.net_quantity
                and position.point_value != spec.point_value
            ):
                raise ValueError(
                    "Changing point value on an open position requires a settlement model."
                )
        position = self._positions.get(mark.instrument)
        if position is not None:
            position.last_price = mark.price
        self._marks[mark.event_id] = fingerprint
        self._last_event_at = mark.occurred_at

    def build_position_snapshot(
        self,
        *,
        instrument: InstrumentRef,
        snapshot_id: str,
        captured_at: datetime,
        contour: ExecutionContour | None = None,
    ) -> PositionSnapshot:
        captured_at = _require_aware_datetime(captured_at, "captured_at")
        if self._last_event_at is not None and captured_at < self._last_event_at:
            raise ValueError("Cannot build a historical snapshot from a later ledger.")
        position = self._positions.get(instrument)
        unrealized = Decimal("0")
        if (
            position
            and position.net_quantity
            and position.average_price is not None
            and position.last_price is not None
        ):
            unrealized = (
                (position.last_price - position.average_price)
                * position.net_quantity
                * position.point_value
            )
        return PositionSnapshot(
            snapshot_id=snapshot_id,
            instrument=instrument,
            net_quantity=position.net_quantity if position else Decimal("0"),
            average_price=position.average_price if position else None,
            mark_price=position.last_price if position else None,
            realized_pnl=position.realized_pnl if position else Decimal("0"),
            unrealized_pnl=unrealized,
            captured_at=captured_at,
            contour=contour or self.contour,
        )

    def build_account_summary(
        self,
        *,
        summary_id: str,
        captured_at: datetime,
        contour: ExecutionContour | None = None,
    ) -> AccountSummary:
        captured_at = _require_aware_datetime(captured_at, "captured_at")
        if self._last_event_at is not None and captured_at < self._last_event_at:
            raise ValueError("Cannot build a historical account from a later ledger.")
        equity = self.cash_balance
        margin = Decimal("0")
        for position in self._positions.values():
            if not position.net_quantity or position.last_price is None:
                continue
            if position.instrument.instrument_type is InstrumentType.FUTURE:
                spec = self.instrument_catalog.resolve(position.instrument, captured_at)
                if spec.point_value != position.point_value:
                    raise ValueError(
                        "Changing point value on an open position requires a settlement model."
                    )
                equity += (
                    (position.last_price - position.average_price)
                    * position.net_quantity
                    * position.point_value
                )
                margin += spec.margin(position.net_quantity)
            else:
                equity += position.net_quantity * position.last_price
                margin += abs(position.net_quantity) * position.last_price
        free = equity - margin
        return AccountSummary(
            summary_id=summary_id,
            account_id=self.account_id,
            equity=equity,
            available_cash=free,
            buying_power=max(Decimal("0"), free),
            margin_used=margin,
            captured_at=captured_at,
            contour=contour or self.contour,
        )

    @staticmethod
    def _apply_position_effect(position, delta, price, commission):
        current, average = position.net_quantity, position.average_price
        if not current or average is None:
            position.average_price = price
        elif current * delta > 0:
            position.average_price = (
                abs(current) * average + abs(delta) * price
            ) / abs(current + delta)
        else:
            closed = min(abs(current), abs(delta))
            position.realized_pnl += (
                closed
                * (price - average)
                * (1 if current > 0 else -1)
                * position.point_value
            )
            if current + delta == 0:
                position.average_price = None
            elif current * (current + delta) < 0:
                position.average_price = price
        position.net_quantity = current + delta
        position.realized_pnl -= commission
        position.last_price = price


__all__ = ["PaperLedger"]
