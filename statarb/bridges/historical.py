from __future__ import annotations

"""Minimal bridge from legacy historical pair runs into canonical contracts."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from statarb.data.alor_history import LoadedAlorHistoryBarsDataset
from statarb.domain import InstrumentRef, InstrumentType, ResearchRun, ResearchRunStatus, SignalDirection, SignalIntent, TargetPosition

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"


@dataclass(frozen=True, slots=True)
class HistoricalSmokeBundle:
    research_run: ResearchRun
    signal_intents: tuple[SignalIntent, SignalIntent]
    target_positions: tuple[TargetPosition, TargetPosition]
    source_pair: tuple[str, str]
    source_row_count: int
    source_signal: int
    as_of: datetime
    fill_prices_by_symbol: Mapping[str, Decimal] | None = None


def build_default_historical_smoke() -> HistoricalSmokeBundle:
    return build_historical_pair_bridge("SRM5", "SPM5", tail_rows=700)


def build_historical_pair_bridge(
    symbol_1: str,
    symbol_2: str,
    *,
    tail_rows: int,
    lookback: int = 60,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    min_estimation_window: int = 512,
    recompute_frequency: int = 16,
    exchange: str = "MOEX",
    instrument_type: InstrumentType = InstrumentType.FUTURE,
) -> HistoricalSmokeBundle:
    import pandas as pd

    frame_1 = _load_legacy_csv(symbol_1, pd).tail(tail_rows)
    frame_2 = _load_legacy_csv(symbol_2, pd).tail(tail_rows)
    return _build_historical_smoke_bundle(
        symbol_1=symbol_1,
        symbol_2=symbol_2,
        frame_1=frame_1,
        frame_2=frame_2,
        lookback=lookback,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        min_estimation_window=min_estimation_window,
        recompute_frequency=recompute_frequency,
        exchange=exchange,
        instrument_type=instrument_type,
        strategy_id="legacy-pairs-walk-forward",
        dataset_id=f"legacy-csv:{symbol_1}-{symbol_2}:tail-{tail_rows}",
        artifact_uris=(
            f"file://data/{symbol_1}.csv",
            f"file://data/{symbol_2}.csv",
        ),
        thesis_prefix="legacy pair bridge",
        hedge_thesis_prefix="legacy pair bridge hedge leg",
        target_reason="legacy historical smoke bridge",
        hedge_target_reason="legacy historical smoke bridge hedge leg",
        source_label="Legacy",
        parameters={"tail_rows": tail_rows},
    )


def build_historical_pair_bridge_from_alor_datasets(
    *,
    left_dataset: LoadedAlorHistoryBarsDataset,
    right_dataset: LoadedAlorHistoryBarsDataset,
    tail_rows: int | None = None,
    lookback: int = 60,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    min_estimation_window: int = 512,
    recompute_frequency: int = 16,
    instrument_type: InstrumentType = InstrumentType.FUTURE,
) -> HistoricalSmokeBundle:
    import pandas as pd

    symbol_1 = _dataset_symbol(left_dataset)
    symbol_2 = _dataset_symbol(right_dataset)
    exchange = _shared_exchange(left_dataset, right_dataset)
    _require_matching_timeframe(left_dataset, right_dataset)

    frame_1 = _frame_from_alor_dataset(left_dataset, pd)
    frame_2 = _frame_from_alor_dataset(right_dataset, pd)
    if tail_rows is not None:
        frame_1 = frame_1.tail(tail_rows)
        frame_2 = frame_2.tail(tail_rows)

    dataset_id = (
        f"alor-normalized:{left_dataset.manifest.dataset_id}+{right_dataset.manifest.dataset_id}"
    )
    return _build_historical_smoke_bundle(
        symbol_1=symbol_1,
        symbol_2=symbol_2,
        frame_1=frame_1,
        frame_2=frame_2,
        lookback=lookback,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        min_estimation_window=min_estimation_window,
        recompute_frequency=recompute_frequency,
        exchange=exchange,
        instrument_type=instrument_type,
        strategy_id="alor-normalized-pairs-walk-forward",
        dataset_id=dataset_id,
        artifact_uris=(
            f"file://{left_dataset.normalized_artifact.storage_uri}",
            f"file://{right_dataset.normalized_artifact.storage_uri}",
        ),
        thesis_prefix="alor normalized pair bridge",
        hedge_thesis_prefix="alor normalized pair bridge hedge leg",
        target_reason="alor normalized historical smoke bridge",
        hedge_target_reason="alor normalized historical smoke bridge hedge leg",
        source_label="Alor normalized",
        parameters={
            "left_dataset_id": left_dataset.manifest.dataset_id,
            "right_dataset_id": right_dataset.manifest.dataset_id,
            "tail_rows": tail_rows,
            "left_storage_uri": left_dataset.normalized_artifact.storage_uri,
            "right_storage_uri": right_dataset.normalized_artifact.storage_uri,
        },
    )


def _load_legacy_csv(symbol: str, pandas_module: object):
    file_path = _DATA_DIR / f"{symbol}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Legacy dataset file is missing: {file_path}")
    return pandas_module.read_csv(file_path)


def _build_historical_smoke_bundle(
    *,
    symbol_1: str,
    symbol_2: str,
    frame_1,
    frame_2,
    lookback: int,
    entry_threshold: float,
    exit_threshold: float,
    min_estimation_window: int,
    recompute_frequency: int,
    exchange: str,
    instrument_type: InstrumentType,
    strategy_id: str,
    dataset_id: str,
    artifact_uris: tuple[str, str],
    thesis_prefix: str,
    hedge_thesis_prefix: str,
    target_reason: str,
    hedge_target_reason: str,
    source_label: str,
    parameters: Mapping[str, object] | None = None,
) -> HistoricalSmokeBundle:
    from core.analysis import DataAnalyzer
    from core.backtest import Backtester

    analyzer = DataAnalyzer()
    joined = analyzer.join_pair(frame_1, frame_2).dropna()
    if len(joined) <= min_estimation_window:
        raise ValueError(f"{source_label} pair slice is too small for the historical smoke bridge.")

    backtester = Backtester(
        series_1=joined["close_1"].values,
        series_2=joined["close_2"].values,
        lookback=lookback,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        min_estimation_window=min_estimation_window,
        recompute_frequency=recompute_frequency,
    )
    results = backtester.run_full_backtest(analyzer)

    if backtester.signals is None:
        raise ValueError(f"{source_label} backtester did not produce signal state.")

    active_indexes = [index for index, signal in enumerate(backtester.signals) if signal != 0]
    if not active_indexes:
        raise ValueError(f"{source_label} backtester produced no active signal for pair {symbol_1}-{symbol_2}.")

    last_active_index = active_indexes[-1]
    source_signal = int(backtester.signals[last_active_index])
    as_of = _coerce_utc_datetime(joined.index[last_active_index].to_pydatetime())
    pair_slug = f"{symbol_1.lower()}-{symbol_2.lower()}"
    timestamp_slug = as_of.strftime("%Y%m%dT%H%M%SZ")
    beta = Decimal(f"{abs(float(results['cointegration']['beta'])):.6f}")
    fill_prices_by_symbol = MappingProxyType(
        {
            symbol_1: Decimal(str(joined.iloc[last_active_index]["close_1"])),
            symbol_2: Decimal(str(joined.iloc[last_active_index]["close_2"])),
        }
    )

    first_instrument = InstrumentRef(
        symbol=symbol_1,
        exchange=exchange,
        instrument_type=instrument_type,
    )
    second_instrument = InstrumentRef(
        symbol=symbol_2,
        exchange=exchange,
        instrument_type=instrument_type,
    )

    first_direction, second_direction = _directions_for_signal(source_signal)
    first_target_quantity, second_target_quantity = _target_quantities_for_signal(source_signal, beta)
    run_id_prefix = strategy_id.removesuffix("-pairs-walk-forward")

    research_run = ResearchRun(
        run_id=f"{run_id_prefix}-{pair_slug}-{timestamp_slug}",
        strategy_id=strategy_id,
        dataset_id=dataset_id,
        status=ResearchRunStatus.SUCCEEDED,
        started_at=_coerce_utc_datetime(joined.index[0].to_pydatetime()),
        completed_at=as_of,
        parameters={
            "lookback": lookback,
            "entry_threshold": entry_threshold,
            "exit_threshold": exit_threshold,
            "min_estimation_window": min_estimation_window,
            "recompute_frequency": recompute_frequency,
            "cointegration_beta": float(results["cointegration"]["beta"]),
            "cointegration_p_value": float(results["cointegration"]["p_value"]),
            **dict(parameters or {}),
        },
        artifact_uris=artifact_uris,
        signal_count=2,
    )

    first_intent = SignalIntent(
        intent_id=f"intent-{pair_slug}-leg-1-{timestamp_slug}",
        research_run_id=research_run.run_id,
        instrument=first_instrument,
        direction=first_direction,
        conviction=Decimal("1"),
        generated_at=as_of,
        horizon="intraday",
        thesis=f"{thesis_prefix} for {symbol_1}-{symbol_2} signal={source_signal}",
    )
    second_intent = SignalIntent(
        intent_id=f"intent-{pair_slug}-leg-2-{timestamp_slug}",
        research_run_id=research_run.run_id,
        instrument=second_instrument,
        direction=second_direction,
        conviction=Decimal("1"),
        generated_at=as_of,
        horizon="intraday",
        thesis=f"{hedge_thesis_prefix} for {symbol_1}-{symbol_2} signal={source_signal}",
    )

    first_target = TargetPosition(
        target_position_id=f"target-{pair_slug}-leg-1-{timestamp_slug}",
        instrument=first_instrument,
        target_quantity=first_target_quantity,
        as_of=as_of,
        signal_intent_id=first_intent.intent_id,
        research_run_id=research_run.run_id,
        reason=target_reason,
    )
    second_target = TargetPosition(
        target_position_id=f"target-{pair_slug}-leg-2-{timestamp_slug}",
        instrument=second_instrument,
        target_quantity=second_target_quantity,
        as_of=as_of,
        signal_intent_id=second_intent.intent_id,
        research_run_id=research_run.run_id,
        reason=hedge_target_reason,
    )

    return HistoricalSmokeBundle(
        research_run=research_run,
        signal_intents=(first_intent, second_intent),
        target_positions=(first_target, second_target),
        source_pair=(symbol_1, symbol_2),
        source_row_count=len(joined),
        source_signal=source_signal,
        as_of=as_of,
        fill_prices_by_symbol=fill_prices_by_symbol,
    )


def _frame_from_alor_dataset(dataset: LoadedAlorHistoryBarsDataset, pandas_module: object):
    return pandas_module.DataFrame(
        {
            "time": [record.time for record in dataset.records],
            "close": [record.close for record in dataset.records],
        }
    )


def _dataset_symbol(dataset: LoadedAlorHistoryBarsDataset) -> str:
    if not dataset.records:
        raise ValueError("Alor history dataset must contain at least one bar.")
    return dataset.records[0].symbol


def _shared_exchange(
    left_dataset: LoadedAlorHistoryBarsDataset,
    right_dataset: LoadedAlorHistoryBarsDataset,
) -> str:
    left_exchange = _dataset_exchange(left_dataset)
    right_exchange = _dataset_exchange(right_dataset)
    if left_exchange != right_exchange:
        raise ValueError(
            f"Alor pair bridge requires matching exchanges, got {left_exchange!r} and {right_exchange!r}."
        )
    return left_exchange


def _dataset_exchange(dataset: LoadedAlorHistoryBarsDataset) -> str:
    if not dataset.records:
        raise ValueError("Alor history dataset must contain at least one bar.")
    return dataset.records[0].exchange


def _require_matching_timeframe(
    left_dataset: LoadedAlorHistoryBarsDataset,
    right_dataset: LoadedAlorHistoryBarsDataset,
) -> None:
    if not left_dataset.records or not right_dataset.records:
        raise ValueError("Alor pair bridge requires non-empty history datasets.")
    left_timeframe = left_dataset.records[0].timeframe
    right_timeframe = right_dataset.records[0].timeframe
    if left_timeframe != right_timeframe:
        raise ValueError(
            f"Alor pair bridge requires matching timeframes, got {left_timeframe!r} and {right_timeframe!r}."
        )


def _coerce_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _directions_for_signal(source_signal: int) -> tuple[SignalDirection, SignalDirection]:
    if source_signal > 0:
        return (SignalDirection.LONG, SignalDirection.SHORT)
    if source_signal < 0:
        return (SignalDirection.SHORT, SignalDirection.LONG)
    raise ValueError("Historical bridge requires a non-zero legacy signal.")


def _target_quantities_for_signal(
    source_signal: int,
    hedge_ratio: Decimal,
) -> tuple[Decimal, Decimal]:
    if source_signal > 0:
        return (Decimal("1"), -hedge_ratio)
    if source_signal < 0:
        return (-Decimal("1"), hedge_ratio)
    raise ValueError("Historical bridge requires a non-zero legacy signal.")


__all__ = [
    "HistoricalSmokeBundle",
    "build_default_historical_smoke",
    "build_historical_pair_bridge",
    "build_historical_pair_bridge_from_alor_datasets",
]
