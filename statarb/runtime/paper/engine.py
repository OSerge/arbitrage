"""Atomic pair transactions using the same ledger as backtest and replay."""

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal, ROUND_FLOOR
from typing import Mapping, Sequence
from uuid import uuid4

from statarb.domain.events import ExecutionContour, _coerce_decimal, _require_non_empty
from statarb.domain.instruments import InstrumentRef, InstrumentType
from statarb.domain.orders import (
    FillEvent,
    OrderEvent,
    OrderSide,
    OrderStatus,
    OrderType,
)
from statarb.domain.positions import (
    AccountSummary,
    PositionSnapshot,
    PriceMark,
    TargetPosition,
)
from statarb.domain.research import SignalDirection, SignalIntent
from statarb.runtime.paper.event_sink import (
    EventJournalEntry,
    JsonlEventSink,
    build_event_journal_entry,
)
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


@dataclass(frozen=True, slots=True)
class PaperExecutionBatch:
    accepted: bool
    results: tuple[PaperExecutionResult, ...]


class PaperExecutionEngine:
    def __init__(
        self,
        *,
        ledger: PaperLedger,
        risk_policy: PaperRiskPolicy,
        event_sink: JsonlEventSink | None = None,
        default_signal_quantity=Decimal("1"),
        session_id: str | None = None,
        _recovered: bool = False,
    ):
        self.ledger, self.risk_policy, self.event_sink = ledger, risk_policy, event_sink
        if ledger.contour is not ExecutionContour.PAPER:
            raise ValueError("PaperExecutionEngine requires a paper-contour ledger.")
        self.default_signal_quantity = _coerce_decimal(
            default_signal_quantity, "default_signal_quantity"
        )
        if self.default_signal_quantity <= 0:
            raise ValueError("default_signal_quantity must be positive.")
        if (
            isinstance(event_sink, JsonlEventSink)
            and event_sink.path.exists()
            and event_sink.path.stat().st_size
            and not _recovered
        ):
            raise ValueError(
                "Existing journal: use PaperExecutionEngine.recover or a new journal path."
            )
        self.session_id = _require_non_empty(session_id or uuid4().hex, "session_id")
        self._counter = 0
        self.journal: list[EventJournalEntry] = []
        self._transactions: dict[str, tuple[tuple, PaperExecutionBatch]] = {}
        self._journal_failed = False
        self._committed_transaction_ids: set[str] = set()

    @classmethod
    def recover(
        cls,
        *,
        journal_path,
        account_id,
        starting_cash,
        risk_policy,
        instrument_catalog=None,
        new_journal_path=None,
    ):
        from statarb.runtime.replay.loader import load_event_journal
        from statarb.runtime.replay.replayer import RuntimeReplayer

        records = load_event_journal(journal_path)
        replay = RuntimeReplayer(
            account_id=account_id,
            starting_cash=starting_cash,
            instrument_catalog=instrument_catalog,
        ).replay(records)
        if records and replay.session.source_contour is not ExecutionContour.PAPER:
            raise ValueError("Paper recovery accepts only a paper-contour journal.")
        if new_journal_path is not None:
            from pathlib import Path

            if (
                Path(new_journal_path).exists()
                and Path(new_journal_path).stat().st_size
            ):
                raise ValueError("Recovery destination must be a new journal.")
            sink = JsonlEventSink(new_journal_path)
            if records:
                entries = []
                for record in records:
                    lineage = dict(record.lineage)
                    if record.transaction_id:
                        lineage.setdefault(
                            "original_transaction_id", record.transaction_id
                        )
                    entries.append(
                        EventJournalEntry(record.event_type, record.payload, lineage)
                    )
                sink.append_batch(
                    entries,
                    transaction_id=f"recovery-{uuid4().hex}",
                )
        else:
            sink = JsonlEventSink(journal_path)
        replay.ledger.contour = ExecutionContour.PAPER
        engine = cls(
            ledger=replay.ledger,
            risk_policy=risk_policy,
            event_sink=sink,
            _recovered=True,
        )
        engine._committed_transaction_ids = {
            record.transaction_id for record in records if record.transaction_id
        }
        engine._committed_transaction_ids.update(
            record.lineage["original_transaction_id"]
            for record in records
            if record.lineage.get("original_transaction_id")
        )
        return engine

    def _next_id(self, prefix):
        self._counter += 1
        return f"{self.session_id}-{prefix}-{self._counter:08d}"

    def target_for_signal(self, signal_intent: SignalIntent):
        quantity = (
            self.default_signal_quantity
            * {
                SignalDirection.LONG: 1,
                SignalDirection.SHORT: -1,
                SignalDirection.FLAT: 0,
            }[signal_intent.direction]
        )
        return TargetPosition(
            self._next_id("target"),
            signal_intent.instrument,
            quantity,
            signal_intent.generated_at,
            signal_intent.intent_id,
            signal_intent.research_run_id,
            "derived-from-signal-intent",
        )

    def execute_signal_intent(self, *, signal_intent, fill_price):
        return self.execute_target_position(
            target_position=self.target_for_signal(signal_intent), fill_price=fill_price
        )

    def execute_target_position(
        self, *, target_position, fill_price, commission=Decimal("0")
    ):
        return self.execute_target_positions(
            target_positions=(target_position,),
            fill_prices={target_position.instrument: fill_price},
            commissions={target_position.instrument: commission},
        ).results[0]

    def execute_target_positions(
        self,
        *,
        target_positions: Sequence[TargetPosition],
        fill_prices: Mapping[InstrumentRef, Decimal],
        commissions: Mapping[InstrumentRef, Decimal] | None = None,
        mark_prices: Mapping[InstrumentRef, Decimal] | None = None,
        transaction_id: str | None = None,
    ) -> PaperExecutionBatch:
        targets = tuple(target_positions)
        if not targets or len({target.instrument for target in targets}) != len(
            targets
        ):
            raise ValueError("A target group requires unique instruments.")
        timestamp = targets[0].as_of
        if any(target.as_of != timestamp for target in targets):
            raise ValueError("A target group must share one execution timestamp.")
        if self._journal_failed:
            raise OSError("Journal failure: recover before further execution.")
        prices = {
            target.instrument: _coerce_decimal(
                fill_prices[target.instrument], "fill_price"
            )
            for target in targets
        }
        marks = {
            target.instrument: _coerce_decimal(
                (mark_prices if mark_prices is not None else prices)[target.instrument],
                "mark_price",
            )
            for target in targets
        }
        fees = {
            target.instrument: _coerce_decimal(
                (commissions or {}).get(target.instrument, 0), "commission"
            )
            for target in targets
        }
        if any(value <= 0 for value in (*prices.values(), *marks.values())) or any(
            value < 0 for value in fees.values()
        ):
            raise ValueError("Prices must be positive and commissions non-negative.")
        transaction_id = transaction_id or self._next_id("transaction")
        fingerprint = (
            targets,
            tuple(prices.items()),
            tuple(marks.items()),
            tuple(fees.items()),
        )
        if transaction_id in self._transactions:
            old_fingerprint, result = self._transactions[transaction_id]
            if old_fingerprint != fingerprint:
                raise ValueError("Conflicting duplicate execution transaction.")
            return result
        if transaction_id in self._committed_transaction_ids:
            raise ValueError(
                "Execution transaction was already committed before recovery."
            )
        candidate = self.ledger.fork()
        results = []
        events = []
        all_reducing = True
        for target in targets:
            current = candidate.current_quantity(target.instrument)
            delta = target.target_quantity - current
            reducing = (
                abs(target.target_quantity) <= abs(current)
                and current * target.target_quantity >= 0
            )
            all_reducing &= reducing
            order_limit = self.risk_policy.max_order_quantity
            if target.instrument.instrument_type is InstrumentType.FUTURE and delta:
                order_limit = order_limit.to_integral_value(rounding=ROUND_FLOOR)
                if order_limit < 1:
                    raise ValueError(
                        "Futures max_order_quantity must allow at least one integer contract."
                    )
            order_check_quantity = (
                min(abs(delta), order_limit) if reducing else abs(delta)
            )
            checks = evaluate_paper_risk_checks(
                current_quantity=current,
                target_position=target,
                order_quantity=order_check_quantity,
                policy=self.risk_policy,
            )
            orders, fills = [], []
            remaining, remaining_fee = abs(delta), fees[target.instrument]
            while remaining:
                quantity = min(remaining, order_limit) if reducing else remaining
                fee = (
                    remaining_fee
                    if quantity == remaining
                    else fees[target.instrument] * quantity / abs(delta)
                )
                side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                accepted = OrderEvent(
                    event_id=self._next_id("order-event"),
                    order_id=self._next_id("order"),
                    instrument=target.instrument,
                    order_type=OrderType.MARKET,
                    side=side,
                    quantity=quantity,
                    status=OrderStatus.ACCEPTED,
                    occurred_at=timestamp,
                    contour=ExecutionContour.PAPER,
                    target_position_id=target.target_position_id,
                    signal_intent_id=target.signal_intent_id,
                )
                fill = FillEvent(
                    event_id=self._next_id("fill-event"),
                    fill_id=self._next_id("fill"),
                    order_id=accepted.order_id,
                    instrument=target.instrument,
                    side=side,
                    fill_quantity=quantity,
                    fill_price=prices[target.instrument],
                    occurred_at=timestamp,
                    contour=ExecutionContour.PAPER,
                    commission=fee,
                )
                candidate.apply_fill(
                    fill_event=fill,
                    snapshot_id=self._next_id("internal-snapshot"),
                    captured_at=timestamp,
                )
                terminal = replace(
                    accepted,
                    event_id=self._next_id("order-event"),
                    status=OrderStatus.FILLED,
                )
                events.extend((accepted, fill, terminal))
                orders.extend((accepted, terminal))
                fills.append(fill)
                remaining -= quantity
                remaining_fee -= fee
            results.append(
                PaperExecutionResult(target, checks, tuple(orders), tuple(fills))
            )
        for target in targets:
            mark = PriceMark(
                self._next_id("mark"),
                target.instrument,
                marks[target.instrument],
                timestamp,
                ExecutionContour.PAPER,
            )
            candidate.apply_mark(mark)
            events.append(mark)
        summary = candidate.build_account_summary(
            summary_id=self._next_id("account"), captured_at=timestamp
        )
        collateral_ok = summary.available_cash >= 0 or all_reducing
        failed = not collateral_ok or any(
            not check.passed for result in results for check in result.risk_results
        )
        if failed:
            rejected = []
            rejection_events = []
            for result in results:
                checks = result.risk_results + (
                    RiskCheckResult(
                        "group_collateral",
                        collateral_ok,
                        None if collateral_ok else "insufficient group collateral",
                    ),
                    RiskCheckResult("atomic_group", False, "target group rejected"),
                )
                orders = tuple(
                    replace(
                        order,
                        status=OrderStatus.REJECTED,
                        reason="; ".join(
                            check.reason
                            for check in checks
                            if not check.passed and check.reason
                        ),
                    )
                    for order in result.order_events
                    if order.status is OrderStatus.ACCEPTED
                )
                rejection_events.extend(orders)
                rejected.append(
                    PaperExecutionResult(result.target_position, checks, orders, ())
                )
            # Rejected no-op groups have no order events and no financial state change.
            self._publish(self.ledger.fork(), rejection_events, transaction_id, targets)
            batch = PaperExecutionBatch(False, tuple(rejected))
        else:
            completed = []
            for result in results:
                snapshot = candidate.build_position_snapshot(
                    instrument=result.target_position.instrument,
                    snapshot_id=self._next_id("position"),
                    captured_at=timestamp,
                )
                events.append(snapshot)
                completed.append(
                    replace(result, position_snapshot=snapshot, account_summary=summary)
                )
            events.append(summary)
            self._publish(candidate, events, transaction_id, targets)
            batch = PaperExecutionBatch(True, tuple(completed))
        self._transactions[transaction_id] = (fingerprint, batch)
        return batch

    def mark_to_market(
        self, *, prices: Mapping[InstrumentRef, Decimal], occurred_at: datetime
    ):
        if self._journal_failed:
            raise OSError("Journal failure: recover before further valuation.")
        candidate = self.ledger.fork()
        events = []
        for instrument, price in prices.items():
            mark = PriceMark(
                self._next_id("mark"),
                instrument,
                price,
                occurred_at,
                ExecutionContour.PAPER,
            )
            candidate.apply_mark(mark)
            events.append(mark)
        for instrument in prices:
            events.append(
                candidate.build_position_snapshot(
                    instrument=instrument,
                    snapshot_id=self._next_id("position"),
                    captured_at=occurred_at,
                )
            )
        summary = candidate.build_account_summary(
            summary_id=self._next_id("account"), captured_at=occurred_at
        )
        events.append(summary)
        self._publish(candidate, events, self._next_id("mark-transaction"), ())
        return summary

    def _publish(self, candidate, events, transaction_id, targets):
        by_instrument = {target.instrument: target for target in targets}
        entries = []
        for event in events:
            target = by_instrument.get(getattr(event, "instrument", None))
            lineage = {"session_id": self.session_id}
            if target is not None:
                lineage.update(
                    target_position_id=target.target_position_id,
                    signal_intent_id=target.signal_intent_id,
                    research_run_id=target.research_run_id,
                )
            entries.append(build_event_journal_entry(event, lineage=lineage))
        if self.event_sink is not None and entries:
            try:
                self.event_sink.append_batch(entries, transaction_id=transaction_id)
            except OSError:
                self._journal_failed = True
                raise
        self.ledger.commit_from(candidate)
        self.journal.extend(entries)


__all__ = ["PaperExecutionEngine", "PaperExecutionResult", "PaperExecutionBatch"]
