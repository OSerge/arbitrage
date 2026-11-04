from datetime import UTC, datetime
from decimal import Decimal

import pytest

from statarb.domain import InstrumentRef, InstrumentType, TargetPosition
from statarb.runtime.paper.engine import PaperExecutionEngine
from statarb.runtime.paper.ledger import PaperLedger
from statarb.runtime.paper.risk_checks import PaperRiskPolicy


NOW = datetime(2026, 1, 5, 10, tzinfo=UTC)
STOCK = InstrumentRef("SBER", "MOEX", InstrumentType.EQUITY)


def target(quantity="1"):
    return TargetPosition("regression-target", STOCK, Decimal(quantity), NOW)


def test_insufficient_collateral_does_not_mutate_ledger():
    ledger = PaperLedger(account_id="regression", starting_cash="100")
    engine = PaperExecutionEngine(ledger=ledger, risk_policy=PaperRiskPolicy())
    result = engine.execute_target_position(target_position=target(), fill_price="200")
    assert not result.fill_events
    assert ledger.current_quantity(STOCK) == 0
    assert ledger.cash_balance == 100


def test_duplicate_fill_is_idempotent():
    ledger = PaperLedger(account_id="regression", starting_cash="1000")
    engine = PaperExecutionEngine(ledger=ledger, risk_policy=PaperRiskPolicy())
    fill = engine.execute_target_position(
        target_position=target(), fill_price="100"
    ).fill_events[0]
    ledger.apply_fill(fill_event=fill, snapshot_id="repeat", captured_at=NOW)
    assert ledger.current_quantity(STOCK) == 1
    assert ledger.cash_balance == 900


def test_journal_failure_does_not_mutate_ledger():
    class FailedSink:
        def append(self, *args, **kwargs):
            raise OSError("disk unavailable")

        def append_batch(self, *args, **kwargs):
            raise OSError("disk unavailable")

    ledger = PaperLedger(account_id="regression", starting_cash="1000")
    engine = PaperExecutionEngine(
        ledger=ledger, risk_policy=PaperRiskPolicy(), event_sink=FailedSink()
    )
    with pytest.raises(OSError, match="disk unavailable"):
        engine.execute_target_position(target_position=target(), fill_price="100")
    assert ledger.current_quantity(STOCK) == 0
    assert ledger.cash_balance == 1000
