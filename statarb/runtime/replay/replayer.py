"""Reconstruct every financial checkpoint from fills and independent price marks."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Mapping

from statarb.domain.events import ExecutionContour, ReplaySession, ReplaySessionStatus
from statarb.domain.orders import FillEvent, OrderEvent, OrderStatus
from statarb.domain.positions import AccountSummary, PositionSnapshot, PriceMark
from statarb.domain.specifications import InstrumentCatalog
from statarb.runtime.paper.ledger import PaperLedger
from statarb.runtime.replay.loader import EventJournalRecord, decode_event_record


@dataclass(frozen=True, slots=True)
class ReplayResult:
    session: ReplaySession
    order_events: tuple[OrderEvent, ...]
    fill_events: tuple[FillEvent, ...]
    recorded_position_snapshots: tuple[PositionSnapshot, ...]
    recorded_account_summaries: tuple[AccountSummary, ...]
    position_snapshot: PositionSnapshot | None = None
    account_summary: AccountSummary | None = None
    position_snapshots: Mapping = field(default_factory=dict)
    ledger: PaperLedger | None = field(default=None, repr=False, compare=False)


class RuntimeReplayer:
    def __init__(
        self,
        *,
        account_id: str,
        starting_cash: Decimal | int | float | str,
        instrument_catalog: InstrumentCatalog | None = None,
    ):
        self.account_id, self.starting_cash = account_id, starting_cash
        self.instrument_catalog = instrument_catalog
        self._session_counter = 0

    def replay(
        self,
        records: tuple[EventJournalRecord, ...],
        *,
        replayed_at: datetime | None = None,
    ) -> ReplayResult:
        timestamp = replayed_at or datetime.now(UTC)
        ledger = PaperLedger(
            account_id=self.account_id,
            starting_cash=self.starting_cash,
            contour=ExecutionContour.REPLAY,
            instrument_catalog=self.instrument_catalog,
        )
        orders, fills, recorded_positions, recorded_accounts = [], [], [], []
        snapshots = {}
        last_snapshot = last_account = None
        source_contour = None
        last_time = None
        event_ids = {}
        order_states = {}
        filled_quantities = {}
        for record in records:
            event = decode_event_record(record)
            event_id = getattr(event, "event_id", None)
            if event_id is not None and event_id in event_ids:
                if event_ids[event_id] != event.to_dict():
                    raise ValueError("Conflicting duplicate event_id in replay.")
                continue
            if event_id is not None:
                event_ids[event_id] = event.to_dict()
            if source_contour is None:
                source_contour = event.contour
            elif event.contour is not source_contour:
                raise ValueError("A replay journal cannot mix source contours.")
            event_time = getattr(
                event, "occurred_at", getattr(event, "captured_at", None)
            )
            if last_time is not None and event_time < last_time:
                raise ValueError("Replay events must be chronological.")
            last_time = event_time
            if isinstance(event, OrderEvent):
                previous = order_states.get(event.order_id)
                if previous is not None:
                    for name in ("instrument", "side", "quantity", "order_type"):
                        if getattr(previous, name) != getattr(event, name):
                            raise ValueError(f"Order lifecycle mismatch on {name}.")
                    if previous.status in (
                        OrderStatus.FILLED,
                        OrderStatus.CANCELED,
                        OrderStatus.REJECTED,
                    ):
                        raise ValueError("Order transition after terminal state.")
                if (
                    event.status is OrderStatus.FILLED
                    and filled_quantities.get(event.order_id, 0) != event.quantity
                ):
                    raise ValueError("FILLED status does not match cumulative fills.")
                order_states[event.order_id] = event
                orders.append(event)
            elif isinstance(event, FillEvent):
                known = order_states.get(event.order_id)
                if known is not None and (
                    known.instrument != event.instrument
                    or known.side != event.side
                    or known.status in (OrderStatus.REJECTED, OrderStatus.CANCELED)
                ):
                    raise ValueError("Fill contradicts its order.")
                duplicate_key = (event.contour.value, event.fill_id)
                already_applied = duplicate_key in ledger._fills
                last_snapshot = ledger.apply_fill(
                    fill_event=event,
                    snapshot_id=self._next_session_id("position"),
                    captured_at=event.occurred_at,
                )
                if not already_applied:
                    filled_quantities[event.order_id] = (
                        filled_quantities.get(event.order_id, Decimal("0"))
                        + event.fill_quantity
                    )
                    if (
                        known is not None
                        and filled_quantities[event.order_id] > known.quantity
                    ):
                        raise ValueError("Cumulative fills exceed order quantity.")
                    fills.append(event)
                snapshots[event.instrument] = last_snapshot
            elif isinstance(event, PriceMark):
                ledger.apply_mark(event)
            elif isinstance(event, PositionSnapshot):
                actual = ledger.build_position_snapshot(
                    instrument=event.instrument,
                    snapshot_id=event.snapshot_id,
                    captured_at=event.captured_at,
                    contour=event.contour,
                )
                _assert_fields(
                    event,
                    actual,
                    (
                        "net_quantity",
                        "average_price",
                        "mark_price",
                        "realized_pnl",
                        "unrealized_pnl",
                    ),
                    "position",
                )
                recorded_positions.append(event)
                snapshots[event.instrument] = actual
                last_snapshot = actual
            elif isinstance(event, AccountSummary):
                actual = ledger.build_account_summary(
                    summary_id=event.summary_id,
                    captured_at=event.captured_at,
                    contour=event.contour,
                )
                _assert_fields(
                    event,
                    actual,
                    (
                        "account_id",
                        "equity",
                        "available_cash",
                        "buying_power",
                        "margin_used",
                    ),
                    "account",
                )
                recorded_accounts.append(event)
                last_account = actual
        if last_time is not None:
            # A final fill/mark without a recorded checkpoint must still be returned
            # as reconstructed state, rather than returning a stale prior checkpoint.
            summary_id = (
                last_account.summary_id
                if last_account and last_account.captured_at == last_time
                else self._next_session_id("account")
            )
            last_account = ledger.build_account_summary(
                summary_id=summary_id,
                captured_at=last_time,
                contour=source_contour or ExecutionContour.PAPER,
            )
            for instrument in ledger.instruments:
                prior = snapshots.get(instrument)
                snapshots[instrument] = ledger.build_position_snapshot(
                    instrument=instrument,
                    snapshot_id=prior.snapshot_id
                    if prior
                    else self._next_session_id("position"),
                    captured_at=last_time,
                    contour=source_contour or ExecutionContour.PAPER,
                )
            if last_snapshot:
                last_snapshot = snapshots[last_snapshot.instrument]
        session = ReplaySession(
            session_id=self._next_session_id("session"),
            started_at=timestamp,
            status=ReplaySessionStatus.SUCCEEDED,
            source_contour=source_contour or ExecutionContour.PAPER,
            source_event_count=len(records),
            ended_at=timestamp,
            replayed_event_count=len(records),
            notes="all financial checkpoints reconstructed and verified",
        )
        return ReplayResult(
            session=session,
            order_events=tuple(orders),
            fill_events=tuple(fills),
            recorded_position_snapshots=tuple(recorded_positions),
            recorded_account_summaries=tuple(recorded_accounts),
            position_snapshot=last_snapshot,
            account_summary=last_account,
            position_snapshots=snapshots,
            ledger=ledger,
        )

    def _next_session_id(self, prefix):
        self._session_counter += 1
        return f"replay-{prefix}-{self._session_counter:08d}"


def _assert_fields(expected, actual, fields, kind):
    for name in fields:
        if getattr(expected, name) != getattr(actual, name):
            raise ValueError(f"Replay {kind} parity mismatch on {name}.")


__all__ = ["ReplayResult", "RuntimeReplayer"]
