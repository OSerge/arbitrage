from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from statarb.domain.events import ExecutionContour, ReplaySessionStatus
from statarb.domain.instruments import InstrumentRef, InstrumentType
from statarb.domain.positions import TargetPosition
from statarb.runtime.paper.engine import PaperExecutionEngine
from statarb.runtime.paper.event_sink import JsonlEventSink
from statarb.runtime.paper.ledger import PaperLedger
from statarb.runtime.paper.risk_checks import PaperRiskPolicy
from statarb.runtime.replay.loader import load_event_journal
from statarb.runtime.replay.replayer import RuntimeReplayer


def _instrument() -> InstrumentRef:
    return InstrumentRef(
        symbol="GAZP",
        exchange="MOEX",
        instrument_type=InstrumentType.EQUITY,
    )


def test_replayer_rebuilds_ledger_from_paper_journal(tmp_path) -> None:
    journal_path = tmp_path / "paper-replay.jsonl"
    engine = PaperExecutionEngine(
        ledger=PaperLedger(
            account_id="paper-main",
            starting_cash=Decimal("1000000.00"),
        ),
        risk_policy=PaperRiskPolicy(
            max_order_quantity=Decimal("50"),
            max_abs_position=Decimal("50"),
        ),
        event_sink=JsonlEventSink(journal_path),
    )
    target = TargetPosition(
        target_position_id="target-101",
        instrument=_instrument(),
        target_quantity=Decimal("7"),
        as_of=datetime(2026, 5, 21, 11, 0, tzinfo=UTC),
        signal_intent_id="intent-101",
        research_run_id="run-101",
        reason="paper replay scenario",
    )

    paper_result = engine.execute_target_position(
        target_position=target,
        fill_price=Decimal("310.25"),
    )
    records = load_event_journal(journal_path)
    replay_result = RuntimeReplayer(
        account_id="paper-main",
        starting_cash=Decimal("1000000.00"),
    ).replay(records, replayed_at=datetime(2026, 5, 21, 12, 0, tzinfo=UTC))

    assert replay_result.session.status is ReplaySessionStatus.SUCCEEDED
    assert replay_result.session.source_contour is ExecutionContour.PAPER
    assert replay_result.session.source_event_count == 4
    assert replay_result.session.replayed_event_count == 4

    assert replay_result.position_snapshot is not None
    assert replay_result.position_snapshot.net_quantity == Decimal("7")
    assert replay_result.position_snapshot.average_price == Decimal("310.25")
    assert replay_result.account_summary is not None
    assert replay_result.account_summary.available_cash == Decimal("997828.25")

    assert paper_result.position_snapshot is not None
    assert replay_result.position_snapshot.net_quantity == paper_result.position_snapshot.net_quantity
    assert replay_result.position_snapshot.average_price == paper_result.position_snapshot.average_price
    assert replay_result.account_summary == paper_result.account_summary
    assert records[2].lineage["target_position_id"] == target.target_position_id
