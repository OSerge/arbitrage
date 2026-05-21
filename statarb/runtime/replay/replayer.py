from __future__ import annotations

"""Replay canonical runtime journals through the same ledger logic."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from statarb.domain.events import ExecutionContour, ReplaySession, ReplaySessionStatus
from statarb.domain.orders import FillEvent, OrderEvent
from statarb.domain.positions import AccountSummary, PositionSnapshot
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


class RuntimeReplayer:
    def __init__(
        self,
        *,
        account_id: str,
        starting_cash: Decimal | int | float | str,
    ) -> None:
        self.account_id = account_id
        self.starting_cash = starting_cash
        self._session_counter = 0

    def replay(
        self,
        records: tuple[EventJournalRecord, ...],
        *,
        replayed_at: datetime | None = None,
    ) -> ReplayResult:
        replay_timestamp = replayed_at or datetime.now(UTC)
        order_events: list[OrderEvent] = []
        fill_events: list[FillEvent] = []
        recorded_position_snapshots: list[PositionSnapshot] = []
        recorded_account_summaries: list[AccountSummary] = []
        source_contour = ExecutionContour.PAPER

        ledger = PaperLedger(
            account_id=self.account_id,
            starting_cash=self.starting_cash,
            contour=ExecutionContour.REPLAY,
        )

        last_derived_snapshot: PositionSnapshot | None = None
        last_derived_account_summary: AccountSummary | None = None
        last_recorded_snapshot: PositionSnapshot | None = None
        last_recorded_account_summary: AccountSummary | None = None
        derived_snapshots_by_instrument: dict[tuple[str, str, str], PositionSnapshot] = {}
        recorded_snapshots_by_instrument: dict[tuple[str, str, str], PositionSnapshot] = {}

        for record in records:
            event = decode_event_record(record)
            source_contour = getattr(event, "contour", source_contour)

            if isinstance(event, OrderEvent):
                order_events.append(event)
                continue

            if isinstance(event, FillEvent):
                fill_events.append(event)
                last_derived_snapshot = ledger.apply_fill(
                    fill_event=event,
                    snapshot_id=self._next_session_id("replay-position-snapshot"),
                    captured_at=event.occurred_at,
                    contour=ExecutionContour.REPLAY,
                )
                last_derived_account_summary = ledger.build_account_summary(
                    summary_id=self._next_session_id("replay-account-summary"),
                    captured_at=event.occurred_at,
                    contour=ExecutionContour.REPLAY,
                )
                derived_snapshots_by_instrument[_position_key(event.instrument)] = last_derived_snapshot
                continue

            if isinstance(event, PositionSnapshot):
                recorded_position_snapshots.append(event)
                last_recorded_snapshot = event
                recorded_snapshots_by_instrument[_position_key(event.instrument)] = event
                continue

            recorded_account_summaries.append(event)
            last_recorded_account_summary = event

        for instrument_key, expected_snapshot in recorded_snapshots_by_instrument.items():
            actual_snapshot = derived_snapshots_by_instrument.get(instrument_key)
            if actual_snapshot is None:
                actual_snapshot = ledger.build_position_snapshot(
                    instrument=expected_snapshot.instrument,
                    snapshot_id=self._next_session_id("replay-position-snapshot"),
                    captured_at=expected_snapshot.captured_at,
                    contour=ExecutionContour.REPLAY,
                )
            _assert_position_parity(expected_snapshot, actual_snapshot)

        if last_recorded_account_summary is not None and last_derived_account_summary is not None:
            _assert_account_parity(last_recorded_account_summary, last_derived_account_summary)

        session = ReplaySession(
            session_id=self._next_session_id("replay-session"),
            started_at=replay_timestamp,
            ended_at=replay_timestamp,
            status=ReplaySessionStatus.SUCCEEDED,
            source_contour=source_contour,
            source_event_count=len(records),
            replayed_event_count=len(records),
            notes="paper-runtime journal replay",
        )

        return ReplayResult(
            session=session,
            order_events=tuple(order_events),
            fill_events=tuple(fill_events),
            recorded_position_snapshots=tuple(recorded_position_snapshots),
            recorded_account_summaries=tuple(recorded_account_summaries),
            position_snapshot=last_recorded_snapshot or last_derived_snapshot,
            account_summary=last_recorded_account_summary or last_derived_account_summary,
        )

    def _next_session_id(self, prefix: str) -> str:
        self._session_counter += 1
        return f"{prefix}-{self._session_counter:06d}"


def _assert_position_parity(
    expected: PositionSnapshot,
    actual: PositionSnapshot,
) -> None:
    if expected.net_quantity != actual.net_quantity:
        raise ValueError("Replay position parity mismatch on net_quantity.")
    if expected.average_price != actual.average_price:
        raise ValueError("Replay position parity mismatch on average_price.")
    if expected.realized_pnl != actual.realized_pnl:
        raise ValueError("Replay position parity mismatch on realized_pnl.")


def _assert_account_parity(
    expected: AccountSummary,
    actual: AccountSummary,
) -> None:
    if expected.equity != actual.equity:
        raise ValueError("Replay account parity mismatch on equity.")
    if expected.available_cash != actual.available_cash:
        raise ValueError("Replay account parity mismatch on available_cash.")


def _position_key(instrument) -> tuple[str, str, str]:
    return (
        instrument.symbol,
        instrument.exchange,
        instrument.instrument_type.value,
    )


__all__ = ["ReplayResult", "RuntimeReplayer"]
