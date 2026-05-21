import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from statarb.domain import events, instruments, orders, positions, research


def _contract(module: object, name: str) -> type:
    contract = getattr(module, name, None)
    assert contract is not None, f"{module.__name__}.{name} is missing"
    return contract


def _jsonable(payload: dict[str, object]) -> None:
    json.dumps(payload)


def test_research_contracts_serialize_lineage_and_boundaries() -> None:
    instrument_ref = _contract(instruments, "InstrumentRef")
    instrument_type = _contract(instruments, "InstrumentType")
    research_run_cls = _contract(research, "ResearchRun")
    research_status = _contract(research, "ResearchRunStatus")
    signal_intent_cls = _contract(research, "SignalIntent")
    signal_direction = _contract(research, "SignalDirection")

    run = research_run_cls(
        run_id="run-001",
        strategy_id="pairs-mean-reversion",
        dataset_id="dataset-2026-05-21",
        status=research_status.SUCCEEDED,
        started_at=datetime(2026, 5, 21, 9, 30, tzinfo=UTC),
        completed_at=datetime(2026, 5, 21, 9, 45, tzinfo=UTC),
        parameters={"lookback": 120, "entry_zscore": 2.1},
        artifact_uris=("parquet://research/run-001/signals.parquet",),
        signal_count=1,
    )
    signal = signal_intent_cls(
        intent_id="intent-001",
        research_run_id=run.run_id,
        instrument=instrument_ref(
            symbol="SBER",
            exchange="MOEX",
            instrument_type=instrument_type.EQUITY,
        ),
        direction=signal_direction.LONG,
        conviction=Decimal("0.80"),
        horizon="intraday",
        generated_at=datetime(2026, 5, 21, 9, 46, tzinfo=UTC),
        thesis="zscore mean reversion",
    )

    run_payload = run.to_dict()
    signal_payload = signal.to_dict()

    assert run_payload["status"] == "succeeded"
    assert run_payload["signal_count"] == 1
    assert signal_payload["research_run_id"] == "run-001"
    assert signal_payload["instrument"]["symbol"] == "SBER"
    assert signal_payload["direction"] == "long"
    assert signal_payload["conviction"] == "0.80"
    assert "target_quantity" not in signal_payload

    _jsonable(run_payload)
    _jsonable(signal_payload)


def test_execution_contracts_share_canonical_shape_for_paper_and_replay() -> None:
    instrument_ref = _contract(instruments, "InstrumentRef")
    instrument_type = _contract(instruments, "InstrumentType")
    contour = _contract(events, "ExecutionContour")
    target_position_cls = _contract(positions, "TargetPosition")
    position_snapshot_cls = _contract(positions, "PositionSnapshot")
    account_summary_cls = _contract(positions, "AccountSummary")
    order_event_cls = _contract(orders, "OrderEvent")
    fill_event_cls = _contract(orders, "FillEvent")
    order_side = _contract(orders, "OrderSide")
    order_type = _contract(orders, "OrderType")
    order_status = _contract(orders, "OrderStatus")

    instrument_obj = instrument_ref(
        symbol="GAZP",
        exchange="MOEX",
        instrument_type=instrument_type.EQUITY,
    )
    target = target_position_cls(
        target_position_id="target-001",
        instrument=instrument_obj,
        target_quantity=Decimal("100"),
        as_of=datetime(2026, 5, 21, 10, 0, tzinfo=UTC),
        signal_intent_id="intent-001",
        research_run_id="run-001",
        reason="portfolio sizing",
    )
    order_event = order_event_cls(
        event_id="order-event-001",
        order_id="order-001",
        instrument=instrument_obj,
        order_type=order_type.LIMIT,
        side=order_side.BUY,
        quantity=Decimal("100"),
        status=order_status.ACCEPTED,
        occurred_at=datetime(2026, 5, 21, 10, 1, tzinfo=UTC),
        contour=contour.PAPER,
        limit_price=Decimal("245.10"),
        target_position_id=target.target_position_id,
        signal_intent_id=target.signal_intent_id,
    )
    fill_event = fill_event_cls(
        event_id="fill-event-001",
        fill_id="fill-001",
        order_id="order-001",
        instrument=instrument_obj,
        side=order_side.BUY,
        fill_quantity=Decimal("40"),
        fill_price=Decimal("245.10"),
        commission=Decimal("1.25"),
        occurred_at=datetime(2026, 5, 21, 10, 1, 5, tzinfo=UTC),
        contour=contour.REPLAY,
        broker_trade_id="trade-raw-001",
    )
    snapshot = position_snapshot_cls(
        snapshot_id="snapshot-001",
        instrument=instrument_obj,
        net_quantity=Decimal("40"),
        average_price=Decimal("245.10"),
        mark_price=Decimal("245.25"),
        realized_pnl=Decimal("0"),
        unrealized_pnl=Decimal("6.00"),
        captured_at=datetime(2026, 5, 21, 10, 1, 10, tzinfo=UTC),
        contour=contour.REPLAY,
    )
    account = account_summary_cls(
        summary_id="account-001",
        account_id="paper-account",
        equity=Decimal("1000000"),
        available_cash=Decimal("990194.75"),
        buying_power=Decimal("1500000"),
        margin_used=Decimal("9805.25"),
        captured_at=datetime(2026, 5, 21, 10, 1, 10, tzinfo=UTC),
        contour=contour.REPLAY,
    )

    target_payload = target.to_dict()
    order_payload = order_event.to_dict()
    fill_payload = fill_event.to_dict()
    snapshot_payload = snapshot.to_dict()
    account_payload = account.to_dict()

    assert target_payload["target_quantity"] == "100"
    assert order_payload["target_position_id"] == "target-001"
    assert order_payload["contour"] == "paper"
    assert fill_payload["contour"] == "replay"
    assert snapshot_payload["instrument"]["symbol"] == "GAZP"
    assert account_payload["equity"] == "1000000"

    _jsonable(target_payload)
    _jsonable(order_payload)
    _jsonable(fill_payload)
    _jsonable(snapshot_payload)
    _jsonable(account_payload)


def test_contracts_reject_invalid_invariants() -> None:
    instrument_ref = _contract(instruments, "InstrumentRef")
    instrument_type = _contract(instruments, "InstrumentType")
    contour = _contract(events, "ExecutionContour")
    replay_session_cls = _contract(events, "ReplaySession")
    replay_status = _contract(events, "ReplaySessionStatus")
    signal_intent_cls = _contract(research, "SignalIntent")
    signal_direction = _contract(research, "SignalDirection")
    order_event_cls = _contract(orders, "OrderEvent")
    order_side = _contract(orders, "OrderSide")
    order_type = _contract(orders, "OrderType")
    order_status = _contract(orders, "OrderStatus")
    fill_event_cls = _contract(orders, "FillEvent")

    instrument_obj = instrument_ref(
        symbol="LKOH",
        exchange="MOEX",
        instrument_type=instrument_type.EQUITY,
    )

    with pytest.raises(ValueError, match="conviction"):
        signal_intent_cls(
            intent_id="intent-bad",
            research_run_id="run-001",
            instrument=instrument_obj,
            direction=signal_direction.SHORT,
            conviction=Decimal("1.20"),
            generated_at=datetime(2026, 5, 21, 11, 0, tzinfo=UTC),
        )

    with pytest.raises(ValueError, match="limit_price"):
        order_event_cls(
            event_id="order-bad",
            order_id="order-002",
            instrument=instrument_obj,
            order_type=order_type.LIMIT,
            side=order_side.SELL,
            quantity=Decimal("10"),
            status=order_status.CREATED,
            occurred_at=datetime(2026, 5, 21, 11, 1, tzinfo=UTC),
            contour=contour.PAPER,
        )

    with pytest.raises(ValueError, match="fill_price"):
        fill_event_cls(
            event_id="fill-bad",
            fill_id="fill-002",
            order_id="order-002",
            instrument=instrument_obj,
            side=order_side.SELL,
            fill_quantity=Decimal("10"),
            fill_price=Decimal("0"),
            occurred_at=datetime(2026, 5, 21, 11, 2, tzinfo=UTC),
            contour=contour.PAPER,
        )

    with pytest.raises(ValueError, match="ended_at"):
        replay_session_cls(
            session_id="replay-bad",
            started_at=datetime(2026, 5, 21, 12, 0, tzinfo=UTC),
            ended_at=datetime(2026, 5, 21, 11, 59, tzinfo=UTC),
            status=replay_status.FAILED,
            source_contour=contour.PAPER,
            source_event_count=3,
        )


def test_replay_session_serializes_process_metadata() -> None:
    contour = _contract(events, "ExecutionContour")
    replay_session_cls = _contract(events, "ReplaySession")
    replay_status = _contract(events, "ReplaySessionStatus")

    session = replay_session_cls(
        session_id="replay-001",
        started_at=datetime(2026, 5, 21, 12, 15, tzinfo=UTC),
        ended_at=datetime(2026, 5, 21, 12, 17, tzinfo=UTC),
        status=replay_status.SUCCEEDED,
        source_contour=contour.BROKER_TEST,
        source_event_count=42,
        replayed_event_count=42,
        notes="alor mapper parity check",
    )

    payload = session.to_dict()

    assert payload["status"] == "succeeded"
    assert payload["source_contour"] == "broker-test"
    assert payload["replayed_event_count"] == 42

    _jsonable(payload)
