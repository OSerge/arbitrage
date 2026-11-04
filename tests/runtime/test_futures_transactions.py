"""Hand-calculated futures accounting and durable transaction regressions."""

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import json

import pytest

from statarb.domain import (
    InstrumentCatalog,
    InstrumentRef,
    InstrumentSpec,
    InstrumentType,
    TargetPosition,
)
from statarb.domain.events import ExecutionContour
from statarb.runtime.paper.engine import PaperExecutionEngine
from statarb.runtime.paper.event_sink import JsonlEventSink
from statarb.runtime.paper.ledger import PaperLedger
from statarb.runtime.paper.risk_checks import PaperRiskPolicy
from statarb.runtime.read_model import build_ops_read_model
from statarb.runtime.replay.loader import (
    UncommittedJournalTailWarning,
    load_event_journal,
)
from statarb.runtime.replay.replayer import RuntimeReplayer

NOW = datetime(2026, 1, 5, 10, tzinfo=UTC)


def future(symbol):
    return InstrumentRef(symbol, "MOEX", InstrumentType.FUTURE)


def specification(instrument, margin="50"):
    return InstrumentSpec(
        instrument,
        "synthetic-v1",
        "unit test fixture",
        NOW - timedelta(days=5),
        Decimal("1"),
        Decimal("10"),
        Decimal("100"),
        Decimal(margin),
        Decimal(margin),
        NOW + timedelta(days=20),
    )


def engine_at(path, *, margin="50"):
    instruments = (future("TESTH6"), future("HEDGEH6"))
    catalog = InstrumentCatalog(specification(inst, margin) for inst in instruments)
    ledger = PaperLedger(
        account_id="futures", starting_cash="1000", instrument_catalog=catalog
    )
    return (
        PaperExecutionEngine(
            ledger=ledger,
            risk_policy=PaperRiskPolicy(),
            event_sink=JsonlEventSink(path),
        ),
        instruments,
        catalog,
    )


def target(instrument, quantity, minute=0):
    return TargetPosition(
        f"target-{instrument.symbol}-{minute}-{quantity}",
        instrument,
        Decimal(quantity),
        NOW + timedelta(minutes=minute),
    )


def test_futures_cash_margin_marks_partial_close_and_reversal(tmp_path):
    engine, (inst, _), _ = engine_at(tmp_path / "futures.jsonl")
    opened = engine.execute_target_position(
        target_position=target(inst, "2"), fill_price="100", commission="2"
    )
    assert engine.ledger.cash_balance == 998  # no purchase of the 2000 RUB notional
    assert opened.account_summary.margin_used == 100
    assert opened.account_summary.available_cash == 898
    marked = engine.mark_to_market(
        prices={inst: "105"}, occurred_at=NOW + timedelta(minutes=1)
    )
    assert marked.equity == 1098  # two contracts * five points * ten RUB/point
    reduced = engine.execute_target_position(
        target_position=target(inst, "1", 2), fill_price="110", commission="1"
    )
    assert reduced.position_snapshot.realized_pnl == 97
    assert reduced.position_snapshot.unrealized_pnl == 100
    assert engine.ledger.cash_balance == 1097
    reversed_ = engine.execute_target_position(
        target_position=target(inst, "-1", 3), fill_price="90", commission="2"
    )
    assert reversed_.position_snapshot.average_price == 90
    assert engine.ledger.cash_balance == 995
    closed = engine.execute_target_position(
        target_position=target(inst, "0", 4), fill_price="80", commission="1"
    )
    assert closed.position_snapshot.net_quantity == 0
    assert closed.account_summary.equity == 1094


def test_pair_collateral_is_checked_for_both_legs_atomically(tmp_path):
    engine, instruments, _ = engine_at(tmp_path / "pair.jsonl", margin="600")
    result = engine.execute_target_positions(
        target_positions=[target(inst, "1") for inst in instruments],
        fill_prices={inst: "100" for inst in instruments},
    )
    assert not result.accepted
    assert not any(leg.fill_events for leg in result.results)
    assert all(engine.ledger.current_quantity(inst) == 0 for inst in instruments)
    assert engine.ledger.cash_balance == 1000
    assert all(
        record.event_type == "order_event"
        for record in load_event_journal(tmp_path / "pair.jsonl")
    )


def test_one_leg_risk_failure_rejects_the_other_leg(tmp_path):
    engine, (left, right), _ = engine_at(tmp_path / "risk.jsonl")
    batch = engine.execute_target_positions(
        target_positions=(target(left, "1"), target(right, "101")),
        fill_prices={left: "100", right: "100"},
    )
    assert not batch.accepted
    assert engine.ledger.current_quantity(left) == 0
    assert engine.ledger.current_quantity(right) == 0


def test_fractional_contracts_and_conflicting_fills_are_rejected(tmp_path):
    engine, (inst, _), _ = engine_at(tmp_path / "conflict.jsonl")
    with pytest.raises(ValueError, match="integer contracts"):
        engine.execute_target_position(
            target_position=target(inst, "0.5"), fill_price="100"
        )
    assert engine.ledger.current_quantity(inst) == 0
    fill = engine.execute_target_position(
        target_position=target(inst, "1"), fill_price="100"
    ).fill_events[0]
    with pytest.raises(ValueError, match="Conflicting duplicate fill"):
        engine.ledger.apply_fill(
            fill_event=replace(fill, event_id="changed", commission=Decimal("5")),
            snapshot_id="changed",
            captured_at=NOW,
        )


def test_risk_reduction_is_chunked_without_raising_limits(tmp_path):
    engine, (inst, _), _ = engine_at(tmp_path / "chunks.jsonl")
    engine.execute_target_position(target_position=target(inst, "4"), fill_price="100")
    engine.risk_policy = PaperRiskPolicy(max_order_quantity=2, max_abs_position=1)
    closed = engine.execute_target_position(
        target_position=target(inst, "0", 1), fill_price="100"
    )
    assert [fill.fill_quantity for fill in closed.fill_events] == [2, 2]
    assert closed.position_snapshot.net_quantity == 0


@pytest.mark.parametrize(
    "event_type,field,value",
    [
        ("position_snapshot", "mark_price", "999"),
        ("position_snapshot", "unrealized_pnl", "7"),
        ("position_snapshot", "realized_pnl", "7"),
        ("position_snapshot", "net_quantity", "7"),
        ("account_summary", "equity", "7"),
        ("account_summary", "available_cash", "7"),
        ("account_summary", "buying_power", "7"),
        ("account_summary", "margin_used", "7"),
        ("account_summary", "account_id", "other-account"),
    ],
)
def test_replay_checks_all_intermediate_financial_fields(
    tmp_path, event_type, field, value
):
    path = tmp_path / "tamper.jsonl"
    engine, (inst, _), catalog = engine_at(path)
    engine.execute_target_position(target_position=target(inst, "1"), fill_price="100")
    engine.execute_target_position(
        target_position=target(inst, "0", 1), fill_price="110"
    )
    records = deepcopy(load_event_journal(path))
    next(record for record in records if record.event_type == event_type).payload[
        field
    ] = value
    with pytest.raises(ValueError, match="parity mismatch"):
        RuntimeReplayer(
            account_id="futures", starting_cash="1000", instrument_catalog=catalog
        ).replay(records)


def test_committed_corruption_fails_and_uncommitted_tail_is_reported(tmp_path):
    path = tmp_path / "tail.jsonl"
    engine, (inst, _), catalog = engine_at(path)
    engine.execute_target_position(target_position=target(inst, "1"), fill_price="100")
    committed = path.read_bytes()
    path.write_bytes(committed + b'{"schema_version":2')
    with pytest.warns(UncommittedJournalTailWarning):
        records = load_event_journal(path)
    assert len(records) == 6
    replay = RuntimeReplayer(
        account_id="futures", starting_cash="1000", instrument_catalog=catalog
    ).replay(records)
    assert replay.position_snapshot.net_quantity == 1
    item = json.loads(committed)
    item["events"][1]["payload"]["fill_price"] = "500"
    path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_event_journal(path)


def test_legacy_v1_equity_journal_is_still_readable(tmp_path):
    path = tmp_path / "old.jsonl"
    inst = InstrumentRef("SBER", "MOEX", InstrumentType.EQUITY)
    engine = PaperExecutionEngine(
        ledger=PaperLedger(account_id="old", starting_cash="1000"),
        risk_policy=PaperRiskPolicy(),
    )
    engine.execute_target_position(target_position=target(inst, "1"), fill_price="100")
    entries = [
        entry.to_dict() for entry in engine.journal if entry.event_type != "price_mark"
    ]
    # The old engine recorded acceptance, fill and two checkpoints, without a terminal order event.
    entries.pop(2)
    path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )
    replay = RuntimeReplayer(account_id="old", starting_cash="1000").replay(
        load_event_journal(path)
    )
    assert replay.position_snapshot.net_quantity == 1
    assert replay.account_summary.equity == 1000


def test_recovery_rebuilds_state_and_uses_new_session_ids(tmp_path):
    path = tmp_path / "restart.jsonl"
    engine, (inst, _), catalog = engine_at(path)
    opened = engine.execute_target_position(
        target_position=target(inst, "1"), fill_price="100"
    )
    with pytest.raises(ValueError, match="Existing journal"):
        PaperExecutionEngine(
            ledger=PaperLedger(account_id="futures", starting_cash="1000"),
            risk_policy=PaperRiskPolicy(),
            event_sink=JsonlEventSink(path),
        )
    recovered = PaperExecutionEngine.recover(
        journal_path=path,
        account_id="futures",
        starting_cash="1000",
        risk_policy=PaperRiskPolicy(),
        instrument_catalog=catalog,
    )
    assert recovered.ledger.current_quantity(inst) == 1
    closed = recovered.execute_target_position(
        target_position=target(inst, "0", 1), fill_price="110"
    )
    assert closed.fill_events[0].fill_id != opened.fill_events[0].fill_id
    replay = RuntimeReplayer(
        account_id="futures", starting_cash="1000", instrument_catalog=catalog
    ).replay(load_event_journal(path))
    assert replay.position_snapshot.net_quantity == 0
    assert replay.account_summary.equity == 1100
    assert build_ops_read_model(path).open_order_ids == ()


def test_stale_concurrent_writer_cannot_publish_an_inconsistent_account(tmp_path):
    path = tmp_path / "writers.jsonl"
    first, (inst, _), _ = engine_at(path)
    stale, _, _ = engine_at(path)
    first.execute_target_position(target_position=target(inst, "1"), fill_price="100")
    with pytest.raises(OSError, match="changed under this writer"):
        stale.execute_target_position(
            target_position=target(inst, "2"), fill_price="100"
        )
    assert stale.ledger.current_quantity(inst) == 0


def test_duplicate_execution_transaction_cannot_charge_twice(tmp_path):
    engine, (inst, _), _ = engine_at(tmp_path / "idempotent.jsonl")
    kwargs = dict(
        target_positions=(target(inst, "1"),),
        fill_prices={inst: "100"},
        commissions={inst: "2"},
        transaction_id="once",
    )
    original = engine.execute_target_positions(**kwargs)
    assert engine.execute_target_positions(**kwargs) == original
    assert engine.ledger.cash_balance == 998
    with pytest.raises(ValueError, match="Conflicting duplicate execution"):
        engine.execute_target_positions(**{**kwargs, "fill_prices": {inst: "101"}})


def test_fsync_failure_rolls_back_the_file_and_published_ledger(tmp_path, monkeypatch):
    import os

    path = tmp_path / "fsync.jsonl"
    engine, (inst, _), _ = engine_at(path)
    real_fsync = os.fsync
    calls = []

    def fail_first(fd):
        calls.append(fd)
        if len(calls) == 1:
            raise OSError("fsync unavailable")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_first)
    with pytest.raises(OSError, match="fsync unavailable"):
        engine.execute_target_position(
            target_position=target(inst, "1"), fill_price="100"
        )
    assert path.read_bytes() == b""
    assert engine.ledger.cash_balance == 1000
    assert engine.ledger.current_quantity(inst) == 0


def test_historical_margin_change_updates_collateral_without_new_trades(tmp_path):
    path = tmp_path / "margin.jsonl"
    inst = future("TESTH6")
    change = NOW + timedelta(minutes=1)
    original = specification(inst)
    catalog = InstrumentCatalog(
        (
            replace(original, valid_until=change),
            replace(
                original,
                version="synthetic-v2",
                valid_from=change,
                initial_margin_long=Decimal("75"),
                initial_margin_short=Decimal("75"),
            ),
        )
    )
    engine = PaperExecutionEngine(
        ledger=PaperLedger(
            account_id="margin", starting_cash="1000", instrument_catalog=catalog
        ),
        risk_policy=PaperRiskPolicy(),
        event_sink=JsonlEventSink(path),
    )
    opened = engine.execute_target_position(
        target_position=target(inst, "2"), fill_price="100"
    )
    marked = engine.mark_to_market(prices={inst: "100"}, occurred_at=change)
    assert opened.account_summary.margin_used == 100
    assert marked.margin_used == 150
    assert marked.available_cash == 850
    replay = RuntimeReplayer(
        account_id="margin", starting_cash="1000", instrument_catalog=catalog
    ).replay(load_event_journal(path))
    assert replay.account_summary == marked


def test_paper_engine_does_not_relabel_a_broker_ledger():
    ledger = PaperLedger(
        account_id="broker", starting_cash="1000", contour=ExecutionContour.BROKER_TEST
    )
    with pytest.raises(ValueError, match="paper-contour ledger"):
        PaperExecutionEngine(ledger=ledger, risk_policy=PaperRiskPolicy())


def test_rotated_recovery_preserves_original_transaction_identity(tmp_path):
    path = tmp_path / "original.jsonl"
    engine, (inst, _), catalog = engine_at(path)
    kwargs = dict(
        target_positions=(target(inst, "1"),),
        fill_prices={inst: "100"},
        transaction_id="original-operation",
    )
    engine.execute_target_positions(**kwargs)
    first = tmp_path / "first-recovery.jsonl"
    PaperExecutionEngine.recover(
        journal_path=path,
        new_journal_path=first,
        account_id="futures",
        starting_cash="1000",
        risk_policy=PaperRiskPolicy(),
        instrument_catalog=catalog,
    )
    recovered = PaperExecutionEngine.recover(
        journal_path=first,
        new_journal_path=tmp_path / "second-recovery.jsonl",
        account_id="futures",
        starting_cash="1000",
        risk_policy=PaperRiskPolicy(),
        instrument_catalog=catalog,
    )
    with pytest.raises(ValueError, match="already committed"):
        recovered.execute_target_positions(**kwargs)
    assert recovered.ledger.current_quantity(inst) == 1
