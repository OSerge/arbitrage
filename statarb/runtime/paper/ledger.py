from __future__ import annotations

"""Deterministic paper ledger state."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from statarb.domain.events import ExecutionContour, _coerce_decimal, _require_non_empty
from statarb.domain.instruments import InstrumentRef
from statarb.domain.orders import FillEvent, OrderSide
from statarb.domain.positions import AccountSummary, PositionSnapshot


@dataclass(slots=True)
class _LedgerPosition:
    instrument: InstrumentRef
    net_quantity: Decimal = Decimal("0")
    average_price: Decimal | None = None
    realized_pnl: Decimal = Decimal("0")
    last_price: Decimal | None = None


class PaperLedger:
    def __init__(
        self,
        *,
        account_id: str,
        starting_cash: Decimal | int | float | str,
        contour: ExecutionContour = ExecutionContour.PAPER,
    ) -> None:
        self.account_id = _require_non_empty(account_id, "account_id")
        self.starting_cash = _coerce_decimal(starting_cash, "starting_cash")
        self.cash_balance = self.starting_cash
        self.contour = ExecutionContour.coerce(contour)
        self._positions: dict[tuple[str, str, str], _LedgerPosition] = {}

    def current_quantity(self, instrument: InstrumentRef) -> Decimal:
        position = self._positions.get(self._position_key(instrument))
        if position is None:
            return Decimal("0")
        return position.net_quantity

    def build_position_snapshot(
        self,
        *,
        instrument: InstrumentRef,
        snapshot_id: str,
        captured_at: datetime,
        contour: ExecutionContour | None = None,
    ) -> PositionSnapshot:
        position = self._positions.get(self._position_key(instrument))
        if position is None:
            return PositionSnapshot(
                snapshot_id=snapshot_id,
                instrument=instrument,
                net_quantity=Decimal("0"),
                average_price=None,
                mark_price=None,
                realized_pnl=Decimal("0"),
                unrealized_pnl=Decimal("0"),
                captured_at=captured_at,
                contour=contour or self.contour,
            )

        mark_price = position.last_price
        unrealized_pnl = Decimal("0")
        if (
            mark_price is not None
            and position.average_price is not None
            and position.net_quantity != Decimal("0")
        ):
            if position.net_quantity > Decimal("0"):
                unrealized_pnl = (mark_price - position.average_price) * position.net_quantity
            else:
                unrealized_pnl = (position.average_price - mark_price) * abs(position.net_quantity)

        return PositionSnapshot(
            snapshot_id=snapshot_id,
            instrument=position.instrument,
            net_quantity=position.net_quantity,
            average_price=position.average_price,
            mark_price=mark_price,
            realized_pnl=position.realized_pnl,
            unrealized_pnl=unrealized_pnl,
            captured_at=captured_at,
            contour=contour or self.contour,
        )

    def apply_fill(
        self,
        *,
        fill_event: FillEvent,
        snapshot_id: str,
        captured_at: datetime,
        contour: ExecutionContour | None = None,
    ) -> PositionSnapshot:
        key = self._position_key(fill_event.instrument)
        position = self._positions.get(key)
        if position is None:
            position = _LedgerPosition(instrument=fill_event.instrument)
            self._positions[key] = position

        delta_quantity = fill_event.fill_quantity
        if fill_event.side is OrderSide.SELL:
            delta_quantity = -delta_quantity

        self._apply_cash_effect(fill_event=fill_event)
        self._apply_position_effect(
            position=position,
            delta_quantity=delta_quantity,
            fill_price=fill_event.fill_price,
            commission=fill_event.commission,
        )

        return self.build_position_snapshot(
            instrument=fill_event.instrument,
            snapshot_id=snapshot_id,
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
        market_value = Decimal("0")
        margin_used = Decimal("0")
        for position in self._positions.values():
            if position.last_price is None:
                continue
            market_value += position.net_quantity * position.last_price
            margin_used += abs(position.net_quantity) * position.last_price

        equity = self.cash_balance + market_value
        return AccountSummary(
            summary_id=summary_id,
            account_id=self.account_id,
            equity=equity,
            available_cash=self.cash_balance,
            buying_power=self.cash_balance,
            margin_used=margin_used,
            captured_at=captured_at,
            contour=contour or self.contour,
        )

    @staticmethod
    def _position_key(instrument: InstrumentRef) -> tuple[str, str, str]:
        return (
            instrument.symbol,
            instrument.exchange,
            instrument.instrument_type.value,
        )

    def _apply_cash_effect(self, *, fill_event: FillEvent) -> None:
        notional = fill_event.fill_quantity * fill_event.fill_price
        if fill_event.side is OrderSide.BUY:
            self.cash_balance -= notional
        else:
            self.cash_balance += notional
        self.cash_balance -= fill_event.commission

    @staticmethod
    def _apply_position_effect(
        *,
        position: _LedgerPosition,
        delta_quantity: Decimal,
        fill_price: Decimal,
        commission: Decimal,
    ) -> None:
        current_quantity = position.net_quantity
        current_average = position.average_price

        if current_quantity == Decimal("0") or current_average is None:
            position.net_quantity = delta_quantity
            position.average_price = None if delta_quantity == Decimal("0") else fill_price
            position.realized_pnl -= commission
            position.last_price = fill_price
            return

        if current_quantity * delta_quantity > Decimal("0"):
            new_quantity = current_quantity + delta_quantity
            position.average_price = (
                (abs(current_quantity) * current_average) + (abs(delta_quantity) * fill_price)
            ) / abs(new_quantity)
            position.net_quantity = new_quantity
            position.realized_pnl -= commission
            position.last_price = fill_price
            return

        closed_quantity = min(abs(current_quantity), abs(delta_quantity))
        if current_quantity > Decimal("0"):
            position.realized_pnl += closed_quantity * (fill_price - current_average)
        else:
            position.realized_pnl += closed_quantity * (current_average - fill_price)

        new_quantity = current_quantity + delta_quantity
        if new_quantity == Decimal("0"):
            position.average_price = None
        elif current_quantity * new_quantity > Decimal("0"):
            position.average_price = current_average
        else:
            position.average_price = fill_price

        position.net_quantity = new_quantity
        position.realized_pnl -= commission
        position.last_price = fill_price


__all__ = ["PaperLedger"]
