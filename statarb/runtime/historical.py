"""Historical smoke orchestration over paper execution and replay."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from statarb.bridges.historical import HistoricalSmokeBundle
from statarb.domain.positions import AccountSummary, PositionSnapshot
from statarb.domain.events import _coerce_decimal
from statarb.runtime.paper.engine import PaperExecutionEngine, PaperExecutionResult
from statarb.runtime.paper.event_sink import JsonlEventSink
from statarb.runtime.paper.ledger import PaperLedger
from statarb.runtime.paper.risk_checks import PaperRiskPolicy
from statarb.runtime.replay.loader import EventJournalRecord, load_event_journal
from statarb.runtime.replay.replayer import ReplayResult, RuntimeReplayer


@dataclass(frozen=True, slots=True)
class HistoricalPaperReplaySmokeResult:
    bundle: HistoricalSmokeBundle
    journal_path: Path
    journal_records: tuple[EventJournalRecord, ...]
    fill_prices_by_symbol: Mapping[str, Decimal]
    paper_results: tuple[PaperExecutionResult, ...]
    position_snapshots_by_symbol: Mapping[str, PositionSnapshot]
    replay_position_snapshots_by_symbol: Mapping[str, PositionSnapshot]
    account_summary: AccountSummary | None
    replay_result: ReplayResult


def run_historical_paper_replay_smoke(
    *,
    bundle: HistoricalSmokeBundle,
    journal_path: str | Path,
    account_id: str = "paper-historical-smoke",
    starting_cash: Decimal | int | float | str = Decimal("1000000.00"),
    risk_policy: PaperRiskPolicy | None = None,
) -> HistoricalPaperReplaySmokeResult:
    resolved_journal_path = Path(journal_path)
    fill_prices_by_symbol = _load_fill_prices(bundle)
    engine = PaperExecutionEngine(
        ledger=PaperLedger(
            account_id=account_id,
            starting_cash=starting_cash,
            instrument_catalog=bundle.instrument_catalog,
        ),
        risk_policy=risk_policy or PaperRiskPolicy(),
        event_sink=JsonlEventSink(resolved_journal_path),
    )

    paper_results: list[PaperExecutionResult] = []
    latest_paper_snapshots: dict[str, PositionSnapshot] = {}
    latest_account_summary: AccountSummary | None = None

    batch = engine.execute_target_positions(
        target_positions=bundle.target_positions,
        fill_prices={
            target.instrument: fill_prices_by_symbol[target.instrument.symbol]
            for target in bundle.target_positions
        },
    )
    for target_position, result in zip(
        bundle.target_positions, batch.results, strict=True
    ):
        if result.position_snapshot is None or result.account_summary is None:
            raise ValueError(
                "Historical paper/replay smoke requires accepted paper execution for every target."
            )

        paper_results.append(result)
        latest_paper_snapshots[target_position.instrument.symbol] = (
            result.position_snapshot
        )
        latest_account_summary = result.account_summary

    journal_records = load_event_journal(resolved_journal_path)
    replay_result = RuntimeReplayer(
        account_id=account_id,
        starting_cash=starting_cash,
        instrument_catalog=bundle.instrument_catalog,
    ).replay(journal_records, replayed_at=bundle.as_of)

    return HistoricalPaperReplaySmokeResult(
        bundle=bundle,
        journal_path=resolved_journal_path,
        journal_records=journal_records,
        fill_prices_by_symbol=fill_prices_by_symbol,
        paper_results=tuple(paper_results),
        position_snapshots_by_symbol=MappingProxyType(dict(latest_paper_snapshots)),
        replay_position_snapshots_by_symbol=MappingProxyType(
            {
                instrument.symbol: snapshot
                for instrument, snapshot in replay_result.position_snapshots.items()
            }
        ),
        account_summary=latest_account_summary,
        replay_result=replay_result,
    )


def _load_fill_prices(bundle: HistoricalSmokeBundle) -> Mapping[str, Decimal]:
    if bundle.fill_prices_by_symbol is None or set(bundle.fill_prices_by_symbol) != set(
        bundle.source_pair
    ):
        raise ValueError(
            "Historical smoke requires explicit as-of prices for both legs; implicit CSV price lookup is retired."
        )
    prices = {
        symbol: _coerce_decimal(price, "as_of_price")
        for symbol, price in bundle.fill_prices_by_symbol.items()
    }
    if any(price <= 0 for price in prices.values()):
        raise ValueError("Historical smoke prices must be positive.")
    return MappingProxyType(prices)


__all__ = ["HistoricalPaperReplaySmokeResult", "run_historical_paper_replay_smoke"]
