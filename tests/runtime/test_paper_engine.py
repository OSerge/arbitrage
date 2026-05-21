from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from statarb.domain.instruments import InstrumentRef, InstrumentType
from statarb.domain.orders import OrderSide, OrderStatus
from statarb.domain.positions import TargetPosition
from statarb.domain.research import SignalDirection, SignalIntent
from statarb.runtime.paper.engine import PaperExecutionEngine
from statarb.runtime.paper.event_sink import JsonlEventSink
from statarb.runtime.paper.ledger import PaperLedger
from statarb.runtime.paper.risk_checks import PaperRiskPolicy
from statarb.runtime.replay.loader import load_event_journal


def _instrument() -> InstrumentRef:
    return InstrumentRef(
        symbol="SBER",
        exchange="MOEX",
        instrument_type=InstrumentType.EQUITY,
    )


def test_paper_engine_converts_signal_into_fill_and_journal(tmp_path) -> None:
    journal_path = tmp_path / "paper-session.jsonl"
    engine = PaperExecutionEngine(
        ledger=PaperLedger(
            account_id="paper-main",
            starting_cash=Decimal("1000000.00"),
        ),
        risk_policy=PaperRiskPolicy(
            max_order_quantity=Decimal("20"),
            max_abs_position=Decimal("20"),
        ),
        event_sink=JsonlEventSink(journal_path),
        default_signal_quantity=Decimal("10"),
    )
    signal = SignalIntent(
        intent_id="intent-001",
        research_run_id="run-001",
        instrument=_instrument(),
        direction=SignalDirection.LONG,
        conviction=Decimal("0.80"),
        generated_at=datetime(2026, 5, 21, 10, 0, tzinfo=UTC),
        horizon="intraday",
        thesis="mean reversion entry",
    )

    result = engine.execute_signal_intent(
        signal_intent=signal,
        fill_price=Decimal("245.10"),
    )

    assert result.target_position.signal_intent_id == signal.intent_id
    assert result.target_position.target_quantity == Decimal("10")
    assert all(check.passed for check in result.risk_results)

    order_event = result.order_events[-1]
    fill_event = result.fill_events[-1]

    assert order_event.status is OrderStatus.ACCEPTED
    assert order_event.side is OrderSide.BUY
    assert order_event.signal_intent_id == signal.intent_id
    assert order_event.target_position_id == result.target_position.target_position_id
    assert fill_event.order_id == order_event.order_id

    assert result.position_snapshot is not None
    assert result.position_snapshot.net_quantity == Decimal("10")
    assert result.position_snapshot.average_price == Decimal("245.10")

    assert result.account_summary is not None
    assert result.account_summary.available_cash == Decimal("997549.00")

    journal_entries = load_event_journal(journal_path)

    assert [entry.event_type for entry in journal_entries] == [
        "order_event",
        "fill_event",
        "position_snapshot",
        "account_summary",
    ]
    assert journal_entries[0].lineage["signal_intent_id"] == signal.intent_id
    assert (
        journal_entries[2].lineage["target_position_id"]
        == result.target_position.target_position_id
    )


def test_paper_engine_rejects_target_position_that_breaks_risk_limits(tmp_path) -> None:
    journal_path = tmp_path / "risk-rejection.jsonl"
    engine = PaperExecutionEngine(
        ledger=PaperLedger(
            account_id="paper-main",
            starting_cash=Decimal("1000000.00"),
        ),
        risk_policy=PaperRiskPolicy(
            max_order_quantity=Decimal("25"),
            max_abs_position=Decimal("25"),
        ),
        event_sink=JsonlEventSink(journal_path),
    )
    target = TargetPosition(
        target_position_id="target-001",
        instrument=_instrument(),
        target_quantity=Decimal("40"),
        as_of=datetime(2026, 5, 21, 10, 5, tzinfo=UTC),
        signal_intent_id="intent-002",
        research_run_id="run-001",
        reason="portfolio sizing",
    )

    result = engine.execute_target_position(
        target_position=target,
        fill_price=Decimal("245.10"),
    )

    assert any(not check.passed for check in result.risk_results)
    assert result.fill_events == ()
    assert result.position_snapshot is None
    assert result.account_summary is None

    rejection = result.order_events[-1]
    assert rejection.status is OrderStatus.REJECTED
    assert rejection.reason is not None
    assert "max_abs_position" in rejection.reason

    journal_entries = load_event_journal(journal_path)

    assert [entry.event_type for entry in journal_entries] == ["order_event"]
