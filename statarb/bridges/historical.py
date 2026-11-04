"""Final causal research snapshots; these fixtures are not full financial runs."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import pandas as pd

from statarb.data.bars import prepare_pair_bars
from statarb.domain import (
    InstrumentCatalog,
    InstrumentRef,
    InstrumentSpec,
    InstrumentType,
    ResearchRun,
    ResearchRunStatus,
    SignalDirection,
    SignalIntent,
    TargetPosition,
)
from statarb.research.config import StrategyConfig
from statarb.research.sizing import size_pair
from statarb.research.strategy import run_signal_research

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


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
    instrument_catalog: InstrumentCatalog | None = None


def synthetic_smoke_catalog(symbols=("SRM5", "SPM5")) -> InstrumentCatalog:
    """Explicit unit/GO fixtures for contract plumbing, not historical economics."""
    return InstrumentCatalog(
        InstrumentSpec(
            InstrumentRef(symbol, "MOEX", InstrumentType.FUTURE),
            "synthetic-smoke-v1",
            "SYNTHETIC CONTRACT FIXTURE; not historical MOEX economics",
            datetime(1970, 1, 1, tzinfo=UTC),
            Decimal("1"),
            Decimal("1"),
            Decimal("1"),
            Decimal("100"),
            Decimal("100"),
            datetime(2100, 1, 1, tzinfo=UTC),
        )
        for symbol in symbols
    )


def build_default_historical_smoke():
    return build_historical_pair_bridge(
        "SRM5",
        "SPM5",
        tail_rows=700,
        instrument_catalog=synthetic_smoke_catalog(),
        pair_budget=Decimal("100000"),
        max_hedge_error=Decimal("0.05"),
    )


def build_historical_pair_bridge(
    symbol_1,
    symbol_2,
    *,
    tail_rows,
    lookback=60,
    entry_threshold=2.0,
    exit_threshold=0.5,
    min_estimation_window=512,
    recompute_frequency=16,
    exchange="MOEX",
    instrument_type=InstrumentType.FUTURE,
    instrument_catalog=None,
    pair_budget=None,
    max_hedge_error=None,
    from_time=None,
    to_time=None,
    data_directory=None,
):
    root = Path(data_directory) if data_directory is not None else _DATA_DIR
    frames = [
        pd.read_csv(root / f"{symbol}.csv", float_precision="round_trip")
        for symbol in (symbol_1, symbol_2)
    ]
    return _build_bundle(
        symbol_1,
        symbol_2,
        *frames,
        timeframe_seconds=3600,
        tail_rows=tail_rows,
        lookback=lookback,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        min_estimation_window=min_estimation_window,
        recompute_frequency=recompute_frequency,
        exchange=exchange,
        instrument_type=instrument_type,
        instrument_catalog=instrument_catalog,
        pair_budget=pair_budget,
        max_hedge_error=max_hedge_error,
        from_time=from_time,
        to_time=to_time,
        strategy_id="legacy-pairs-walk-forward",
        dataset_id=f"legacy-csv:{symbol_1}-{symbol_2}:tail-{tail_rows}",
        artifact_uris=tuple(
            f"file://data/{symbol}.csv" for symbol in (symbol_1, symbol_2)
        ),
        parameters={"tail_rows": tail_rows},
    )


def build_historical_pair_bridge_from_alor_datasets(
    *,
    left_dataset,
    right_dataset,
    tail_rows=None,
    lookback=60,
    entry_threshold=2.0,
    exit_threshold=0.5,
    min_estimation_window=512,
    recompute_frequency=16,
    instrument_type=InstrumentType.FUTURE,
    instrument_catalog=None,
    pair_budget=None,
    max_hedge_error=None,
    from_time=None,
    to_time=None,
):
    for dataset in (left_dataset, right_dataset):
        if not dataset.records:
            raise ValueError("Pair bridge requires non-empty history datasets.")
        first = dataset.records[0]
        if any(
            (record.symbol, record.exchange, record.board, record.timeframe)
            != (first.symbol, first.exchange, first.board, first.timeframe)
            for record in dataset.records
        ):
            raise ValueError(
                "History dataset contains inconsistent instrument/timeframe records."
            )
    left, right = left_dataset.records[0], right_dataset.records[0]
    if left.exchange != right.exchange or left.timeframe != right.timeframe:
        raise ValueError("Pair bridge requires matching exchanges and timeframes.")
    try:
        seconds = int(left.timeframe)
    except ValueError as exc:
        raise ValueError(
            "Pair bridge currently requires an explicit timeframe in seconds."
        ) from exc
    frames = [
        pd.DataFrame(
            [
                {
                    name: getattr(record, name)
                    for name in ("time", "open", "high", "low", "close", "volume")
                }
                for record in dataset.records
            ]
        )
        for dataset in (left_dataset, right_dataset)
    ]
    return _build_bundle(
        left.symbol,
        right.symbol,
        *frames,
        timeframe_seconds=seconds,
        tail_rows=tail_rows,
        lookback=lookback,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        min_estimation_window=min_estimation_window,
        recompute_frequency=recompute_frequency,
        exchange=left.exchange,
        instrument_type=instrument_type,
        instrument_catalog=instrument_catalog,
        pair_budget=pair_budget,
        max_hedge_error=max_hedge_error,
        from_time=from_time,
        to_time=to_time,
        strategy_id="alor-normalized-pairs-walk-forward",
        dataset_id=f"alor-normalized:{left_dataset.manifest.dataset_id}+{right_dataset.manifest.dataset_id}",
        artifact_uris=tuple(
            f"file://{dataset.normalized_artifact.storage_uri}"
            for dataset in (left_dataset, right_dataset)
        ),
        parameters={
            "left_dataset_id": left_dataset.manifest.dataset_id,
            "right_dataset_id": right_dataset.manifest.dataset_id,
            "tail_rows": tail_rows,
            "available_at": min(
                datetime.fromisoformat(dataset.manifest.created_at_utc)
                - timedelta(minutes=dataset.manifest.data_delay_minutes or 0)
                for dataset in (left_dataset, right_dataset)
            ),
        },
    )


def _build_bundle(
    symbol_1,
    symbol_2,
    frame_1,
    frame_2,
    *,
    timeframe_seconds,
    tail_rows,
    lookback,
    entry_threshold,
    exit_threshold,
    min_estimation_window,
    recompute_frequency,
    exchange,
    instrument_type,
    instrument_catalog,
    pair_budget,
    max_hedge_error,
    from_time,
    to_time,
    strategy_id,
    dataset_id,
    artifact_uris,
    parameters,
):
    def timestamp(value):
        return (
            datetime.fromtimestamp(value, UTC)
            if isinstance(value, (int, float))
            else value
        )

    bars = prepare_pair_bars(
        frame_1,
        frame_2,
        timeframe_seconds=timeframe_seconds,
        tail_rows=tail_rows,
        from_time=timestamp(from_time),
        to_time=timestamp(to_time),
        available_at=parameters.get("available_at") or datetime.now(UTC),
    )
    config = StrategyConfig(
        lookback=lookback,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        min_estimation_window=min_estimation_window,
        recompute_frequency=recompute_frequency,
    )
    if len(bars) <= config.min_estimation_window:
        raise ValueError(
            "Historical pair slice is too small after filtering for formation and evaluation."
        )
    trace, models = run_signal_research(bars, config)
    final = trace[-1]
    signal, as_of = final.signal, final.generated_at
    instruments = tuple(
        InstrumentRef(symbol, exchange, instrument_type)
        for symbol in (symbol_1, symbol_2)
    )
    prices = tuple(
        Decimal(str(frame.close.iloc[-1])) for frame in (bars.left, bars.right)
    )
    quantities = (Decimal("0"), Decimal("0"))
    if signal:
        if instrument_catalog is None or pair_budget is None or max_hedge_error is None:
            raise ValueError(
                "Active futures targets require historical InstrumentCatalog, pair_budget and max_hedge_error."
            )
        size = size_pair(
            beta=str(final.model.beta),
            direction=signal,
            left_price=prices[0],
            right_price=prices[1],
            left_spec=instrument_catalog.resolve(instruments[0], as_of, trading=True),
            right_spec=instrument_catalog.resolve(instruments[1], as_of, trading=True),
            budget=pair_budget,
            max_hedge_error=max_hedge_error,
            max_abs_position=Decimal("100"),
        )
        if size is None:
            raise ValueError(
                "No integer hedge fits the explicit smoke budget/error constraints."
            )
        quantities = size.left, size.right
    slug = f"{symbol_1.lower()}-{symbol_2.lower()}-{as_of.strftime('%Y%m%dT%H%M%SZ')}"
    run = ResearchRun(
        run_id=f"{strategy_id}-{slug}",
        strategy_id=strategy_id,
        dataset_id=dataset_id,
        status=ResearchRunStatus.SUCCEEDED,
        started_at=bars.index[0].to_pydatetime(),
        completed_at=as_of,
        parameters={
            **parameters,
            **{name: getattr(config, name) for name in config.__dataclass_fields__},
            "snapshot_contract_smoke": True,
            "final_signal": signal,
            "model_count": len(models),
            "model_reason": final.reason,
            "economic_metadata_sources": [
                spec.source for spec in instrument_catalog.specs
            ]
            if instrument_catalog
            else [],
        },
        artifact_uris=artifact_uris,
        signal_count=2,
    )
    intents, targets = [], []
    for leg, (instrument, quantity) in enumerate(
        zip(instruments, quantities, strict=True), start=1
    ):
        direction = (
            SignalDirection.LONG
            if quantity > 0
            else SignalDirection.SHORT
            if quantity < 0
            else SignalDirection.FLAT
        )
        intent = SignalIntent(
            f"intent-{slug}-{leg}",
            run.run_id,
            instrument,
            direction,
            Decimal("1") if signal else Decimal("0"),
            as_of,
            horizon="bar-based",
            thesis=final.reason,
        )
        intents.append(intent)
        targets.append(
            TargetPosition(
                f"target-{slug}-{leg}",
                instrument,
                quantity,
                as_of,
                intent.intent_id,
                run.run_id,
                "final causal research snapshot; contract smoke only",
            )
        )
    return HistoricalSmokeBundle(
        run,
        tuple(intents),
        tuple(targets),
        (symbol_1, symbol_2),
        len(bars),
        signal,
        as_of,
        MappingProxyType(dict(zip((symbol_1, symbol_2), prices, strict=True))),
        instrument_catalog,
    )
