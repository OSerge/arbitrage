from __future__ import annotations

"""Historical smoke orchestration over paper execution and replay."""

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from statarb.bridges.historical import HistoricalSmokeBundle
from statarb.domain.positions import AccountSummary, PositionSnapshot
from statarb.runtime.paper.engine import PaperExecutionEngine, PaperExecutionResult
from statarb.runtime.paper.event_sink import JsonlEventSink
from statarb.runtime.paper.ledger import PaperLedger
from statarb.runtime.paper.risk_checks import PaperRiskPolicy
from statarb.runtime.replay.loader import EventJournalRecord, load_event_journal
from statarb.runtime.replay.replayer import ReplayResult, RuntimeReplayer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"


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
        ),
        risk_policy=risk_policy or _build_default_risk_policy(bundle),
        event_sink=JsonlEventSink(resolved_journal_path),
    )

    paper_results: list[PaperExecutionResult] = []
    latest_paper_snapshots: dict[str, PositionSnapshot] = {}
    latest_account_summary: AccountSummary | None = None

    for target_position in bundle.target_positions:
        result = engine.execute_target_position(
            target_position=target_position,
            fill_price=fill_prices_by_symbol[target_position.instrument.symbol],
        )
        if result.position_snapshot is None or result.account_summary is None:
            raise ValueError(
                "Historical paper/replay smoke requires accepted paper execution for every target."
            )

        paper_results.append(result)
        latest_paper_snapshots[target_position.instrument.symbol] = result.position_snapshot
        latest_account_summary = result.account_summary

    journal_records = load_event_journal(resolved_journal_path)
    replay_result = RuntimeReplayer(
        account_id=account_id,
        starting_cash=starting_cash,
    ).replay(journal_records, replayed_at=bundle.as_of)

    return HistoricalPaperReplaySmokeResult(
        bundle=bundle,
        journal_path=resolved_journal_path,
        journal_records=journal_records,
        fill_prices_by_symbol=fill_prices_by_symbol,
        paper_results=tuple(paper_results),
        position_snapshots_by_symbol=MappingProxyType(dict(latest_paper_snapshots)),
        replay_position_snapshots_by_symbol=MappingProxyType(
            _latest_snapshots_by_symbol(replay_result.recorded_position_snapshots)
        ),
        account_summary=latest_account_summary,
        replay_result=replay_result,
    )


def _build_default_risk_policy(bundle: HistoricalSmokeBundle) -> PaperRiskPolicy:
    max_abs_target = max(abs(target.target_quantity) for target in bundle.target_positions)
    allowed_quantity = max_abs_target + Decimal("1")
    return PaperRiskPolicy(
        max_order_quantity=allowed_quantity,
        max_abs_position=allowed_quantity,
    )


def _load_fill_prices(bundle: HistoricalSmokeBundle) -> Mapping[str, Decimal]:
    if bundle.fill_prices_by_symbol is not None:
        return MappingProxyType(dict(bundle.fill_prices_by_symbol))
    return MappingProxyType(
        {
            symbol: _load_close_price_for_symbol(symbol=symbol, as_of=bundle.as_of)
            for symbol in bundle.source_pair
        }
    )


def _load_close_price_for_symbol(*, symbol: str, as_of) -> Decimal:
    file_path = _DATA_DIR / f"{symbol}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Historical data file is missing for smoke scenario: {file_path}")

    target_epoch = int(as_of.timestamp())
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["time"]) != target_epoch:
                continue
            return Decimal(row["close"])

    raise ValueError(f"No historical close found for {symbol} at epoch {target_epoch}.")


def _latest_snapshots_by_symbol(
    snapshots: tuple[PositionSnapshot, ...],
) -> dict[str, PositionSnapshot]:
    latest: dict[str, PositionSnapshot] = {}
    for snapshot in snapshots:
        latest[snapshot.instrument.symbol] = snapshot
    return latest


__all__ = ["HistoricalPaperReplaySmokeResult", "run_historical_paper_replay_smoke"]
