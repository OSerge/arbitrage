from __future__ import annotations

"""Minimal deterministic paper execution engine."""

from dataclasses import dataclass
from decimal import Decimal

from statarb.domain.events import ExecutionContour, _coerce_decimal
from statarb.domain.orders import FillEvent, OrderEvent, OrderSide, OrderStatus, OrderType
from statarb.domain.positions import AccountSummary, PositionSnapshot, TargetPosition
from statarb.domain.research import SignalDirection, SignalIntent
from statarb.runtime.paper.event_sink import JsonlEventSink
from statarb.runtime.paper.ledger import PaperLedger
from statarb.runtime.paper.risk_checks import (
    PaperRiskPolicy,
    RiskCheckResult,
    evaluate_paper_risk_checks,
)


@dataclass(frozen=True, slots=True)
class PaperExecutionResult:
    target_position: TargetPosition
    risk_results: tuple[RiskCheckResult, ...]
    order_events: tuple[OrderEvent, ...]
    fill_events: tuple[FillEvent, ...]
    position_snapshot: PositionSnapshot | None = None
    account_summary: AccountSummary | None = None


class PaperExecutionEngine:
    def __init__(
        self,
        *,
        ledger: PaperLedger,
        risk_policy: PaperRiskPolicy,
        event_sink: JsonlEventSink | None = None,
        default_signal_quantity: Decimal | int | float | str = Decimal("1"),
    ) -> None:
        self.ledger = ledger
        self.risk_policy = risk_policy
        self.event_sink = event_sink
        self.default_signal_quantity = _coerce_decimal(
            default_signal_quantity,
            "default_signal_quantity",
        )
        if self.default_signal_quantity <= Decimal("0"):
            raise ValueError("default_signal_quantity must be greater than 0.")

        self._id_counters: dict[str, int] = {}

    def execute_signal_intent(
        self,
        *,
        signal_intent: SignalIntent,
        fill_price: Decimal | int | float | str,
    ) -> PaperExecutionResult:
        target_position = self.target_for_signal(signal_intent)
        return self.execute_target_position(
            target_position=target_position,
            fill_price=fill_price,
        )

    def target_for_signal(self, signal_intent: SignalIntent) -> TargetPosition:
        target_quantity = Decimal("0")
        if signal_intent.direction is SignalDirection.LONG:
            target_quantity = self.default_signal_quantity
        elif signal_intent.direction is SignalDirection.SHORT:
            target_quantity = -self.default_signal_quantity

        return TargetPosition(
            target_position_id=self._next_id("paper-target"),
            instrument=signal_intent.instrument,
            target_quantity=target_quantity,
            as_of=signal_intent.generated_at,
            signal_intent_id=signal_intent.intent_id,
            research_run_id=signal_intent.research_run_id,
            reason="derived-from-signal-intent",
        )

    def execute_target_position(
        self,
        *,
        target_position: TargetPosition,
        fill_price: Decimal | int | float | str,
    ) -> PaperExecutionResult:
        normalized_fill_price = _coerce_decimal(fill_price, "fill_price")
        current_quantity = self.ledger.current_quantity(target_position.instrument)
        delta_quantity = target_position.target_quantity - current_quantity
        order_quantity = abs(delta_quantity)

        if order_quantity == Decimal("0"):
            return PaperExecutionResult(
                target_position=target_position,
                risk_results=(),
                order_events=(),
                fill_events=(),
                position_snapshot=self.ledger.build_position_snapshot(
                    instrument=target_position.instrument,
                    snapshot_id=self._next_id("paper-position-snapshot"),
                    captured_at=target_position.as_of,
                    contour=ExecutionContour.PAPER,
                ),
                account_summary=self.ledger.build_account_summary(
                    summary_id=self._next_id("paper-account-summary"),
                    captured_at=target_position.as_of,
                    contour=ExecutionContour.PAPER,
                ),
            )

        risk_results = evaluate_paper_risk_checks(
            current_quantity=current_quantity,
            target_position=target_position,
            order_quantity=order_quantity,
            policy=self.risk_policy,
        )

        side = OrderSide.BUY if delta_quantity > Decimal("0") else OrderSide.SELL
        if any(not result.passed for result in risk_results):
            rejection = OrderEvent(
                event_id=self._next_id("paper-order-event"),
                order_id=self._next_id("paper-order"),
                instrument=target_position.instrument,
                order_type=OrderType.MARKET,
                side=side,
                quantity=order_quantity,
                status=OrderStatus.REJECTED,
                occurred_at=target_position.as_of,
                contour=ExecutionContour.PAPER,
                reason="; ".join(
                    result.reason for result in risk_results if result.reason is not None
                ),
                target_position_id=target_position.target_position_id,
                signal_intent_id=target_position.signal_intent_id,
            )
            self._append_event(
                rejection,
                lineage={
                    "target_position_id": target_position.target_position_id,
                    "signal_intent_id": target_position.signal_intent_id,
                },
            )
            return PaperExecutionResult(
                target_position=target_position,
                risk_results=risk_results,
                order_events=(rejection,),
                fill_events=(),
            )

        accepted = OrderEvent(
            event_id=self._next_id("paper-order-event"),
            order_id=self._next_id("paper-order"),
            instrument=target_position.instrument,
            order_type=OrderType.MARKET,
            side=side,
            quantity=order_quantity,
            status=OrderStatus.ACCEPTED,
            occurred_at=target_position.as_of,
            contour=ExecutionContour.PAPER,
            target_position_id=target_position.target_position_id,
            signal_intent_id=target_position.signal_intent_id,
        )
        fill = FillEvent(
            event_id=self._next_id("paper-fill-event"),
            fill_id=self._next_id("paper-fill"),
            order_id=accepted.order_id,
            instrument=target_position.instrument,
            side=side,
            fill_quantity=order_quantity,
            fill_price=normalized_fill_price,
            occurred_at=target_position.as_of,
            contour=ExecutionContour.PAPER,
        )
        position_snapshot = self.ledger.apply_fill(
            fill_event=fill,
            snapshot_id=self._next_id("paper-position-snapshot"),
            captured_at=target_position.as_of,
            contour=ExecutionContour.PAPER,
        )
        account_summary = self.ledger.build_account_summary(
            summary_id=self._next_id("paper-account-summary"),
            captured_at=target_position.as_of,
            contour=ExecutionContour.PAPER,
        )

        shared_lineage = {
            "target_position_id": target_position.target_position_id,
            "signal_intent_id": target_position.signal_intent_id,
            "order_id": accepted.order_id,
        }
        self._append_event(accepted, lineage=shared_lineage)
        self._append_event(
            fill,
            lineage={
                **shared_lineage,
                "fill_id": fill.fill_id,
            },
        )
        self._append_event(position_snapshot, lineage=shared_lineage)
        self._append_event(account_summary, lineage=shared_lineage)

        return PaperExecutionResult(
            target_position=target_position,
            risk_results=risk_results,
            order_events=(accepted,),
            fill_events=(fill,),
            position_snapshot=position_snapshot,
            account_summary=account_summary,
        )

    def _append_event(
        self,
        event: OrderEvent | FillEvent | PositionSnapshot | AccountSummary,
        *,
        lineage: dict[str, str | None],
    ) -> None:
        if self.event_sink is None:
            return
        self.event_sink.append(event, lineage=lineage)

    def _next_id(self, prefix: str) -> str:
        current = self._id_counters.get(prefix, 0) + 1
        self._id_counters[prefix] = current
        return f"{prefix}-{current:06d}"


__all__ = ["PaperExecutionEngine", "PaperExecutionResult"]
