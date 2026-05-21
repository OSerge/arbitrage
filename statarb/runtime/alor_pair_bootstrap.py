from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from statarb.bridges.historical import HistoricalSmokeBundle, build_historical_pair_bridge_from_alor_datasets
from statarb.data.alor_discovery import find_latest_history_dataset_manifest
from statarb.data.alor_history import LoadedAlorHistoryBarsDataset, load_history_bars_dataset
from statarb.domain import InstrumentType
from statarb.runtime.historical import HistoricalPaperReplaySmokeResult, run_historical_paper_replay_smoke


@dataclass(frozen=True, slots=True)
class PairBootstrapResult:
    left_manifest_path: Path
    right_manifest_path: Path
    left_dataset: LoadedAlorHistoryBarsDataset
    right_dataset: LoadedAlorHistoryBarsDataset
    bundle: HistoricalSmokeBundle
    smoke_result: HistoricalPaperReplaySmokeResult | None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "as_of_utc": self.bundle.as_of.isoformat(),
            "exchange": self.left_dataset.records[0].exchange,
            "left_dataset_id": self.left_dataset.manifest.dataset_id,
            "left_manifest_path": str(self.left_manifest_path),
            "left_storage_uri": self.left_dataset.normalized_artifact.storage_uri,
            "pair": list(self.bundle.source_pair),
            "research_run_id": self.bundle.research_run.run_id,
            "right_dataset_id": self.right_dataset.manifest.dataset_id,
            "right_manifest_path": str(self.right_manifest_path),
            "right_storage_uri": self.right_dataset.normalized_artifact.storage_uri,
            "signal": self.bundle.source_signal,
            "source_row_count": self.bundle.source_row_count,
            "timeframe": self.left_dataset.records[0].timeframe,
        }
        if self.smoke_result is None:
            payload["smoke"] = None
            return payload

        payload["smoke"] = {
            "journal_path": str(self.smoke_result.journal_path),
            "paper_result_count": len(self.smoke_result.paper_results),
            "replay_status": self.smoke_result.replay_result.session.status.name,
            "source_event_count": self.smoke_result.replay_result.session.source_event_count,
        }
        return payload


def run_alor_pair_bootstrap(
    *,
    storage_root: Path,
    left_symbol: str,
    right_symbol: str,
    exchange: str = "MOEX",
    timeframe: str = "60",
    instrument_group: str | None = None,
    from_time: int | None = None,
    to_time: int | None = None,
    left_manifest_path: str | Path | None = None,
    right_manifest_path: str | Path | None = None,
    tail_rows: int | None = None,
    lookback: int = 60,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    min_estimation_window: int = 512,
    recompute_frequency: int = 16,
    instrument_type: InstrumentType | str = InstrumentType.FUTURE,
    journal_path: str | Path | None = None,
    run_smoke: bool = True,
) -> PairBootstrapResult:
    resolved_storage_root = Path(storage_root)
    resolved_left_manifest_path = _resolve_manifest_path(
        storage_root=resolved_storage_root,
        symbol=left_symbol,
        exchange=exchange,
        timeframe=timeframe,
        instrument_group=instrument_group,
        from_time=from_time,
        to_time=to_time,
        manifest_path=left_manifest_path,
    )
    resolved_right_manifest_path = _resolve_manifest_path(
        storage_root=resolved_storage_root,
        symbol=right_symbol,
        exchange=exchange,
        timeframe=timeframe,
        instrument_group=instrument_group,
        from_time=from_time,
        to_time=to_time,
        manifest_path=right_manifest_path,
    )
    left_dataset = load_history_bars_dataset(
        storage_root=resolved_storage_root,
        manifest_path=resolved_left_manifest_path,
    )
    right_dataset = load_history_bars_dataset(
        storage_root=resolved_storage_root,
        manifest_path=resolved_right_manifest_path,
    )
    _require_symbol_match(dataset=left_dataset, expected_symbol=left_symbol)
    _require_symbol_match(dataset=right_dataset, expected_symbol=right_symbol)

    bundle = build_historical_pair_bridge_from_alor_datasets(
        left_dataset=left_dataset,
        right_dataset=right_dataset,
        tail_rows=tail_rows,
        lookback=lookback,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        min_estimation_window=min_estimation_window,
        recompute_frequency=recompute_frequency,
        instrument_type=InstrumentType.coerce(instrument_type),
    )
    smoke_result = (
        run_historical_paper_replay_smoke(
            bundle=bundle,
            journal_path=journal_path or _default_journal_path(resolved_storage_root, bundle),
        )
        if run_smoke
        else None
    )
    return PairBootstrapResult(
        left_manifest_path=resolved_left_manifest_path,
        right_manifest_path=resolved_right_manifest_path,
        left_dataset=left_dataset,
        right_dataset=right_dataset,
        bundle=bundle,
        smoke_result=smoke_result,
    )


def _resolve_manifest_path(
    *,
    storage_root: Path,
    symbol: str,
    exchange: str,
    timeframe: str,
    instrument_group: str | None,
    from_time: int | None,
    to_time: int | None,
    manifest_path: str | Path | None,
) -> Path:
    if manifest_path is not None:
        return Path(manifest_path)

    match = find_latest_history_dataset_manifest(
        storage_root=storage_root,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        instrument_group=instrument_group,
        from_time=from_time,
        to_time=to_time,
    )
    if match is None:
        raise FileNotFoundError(
            "No matching Alor history dataset manifest found for "
            f"symbol={symbol!r}, exchange={exchange!r}, timeframe={timeframe!r}, "
            f"instrument_group={instrument_group!r}, from_time={from_time!r}, to_time={to_time!r}."
        )
    return match.manifest_path


def _require_symbol_match(
    *,
    dataset: LoadedAlorHistoryBarsDataset,
    expected_symbol: str,
) -> None:
    if not dataset.records:
        raise ValueError("Pair bootstrap requires non-empty datasets.")
    observed_symbol = dataset.records[0].symbol
    if observed_symbol != expected_symbol:
        raise ValueError(
            f"Pair bootstrap expected symbol {expected_symbol!r}, got dataset for {observed_symbol!r}."
        )


def _default_journal_path(storage_root: Path, bundle: HistoricalSmokeBundle) -> Path:
    pair_slug = "-".join(symbol.lower() for symbol in bundle.source_pair)
    timestamp_slug = bundle.as_of.strftime("%Y%m%dT%H%M%SZ")
    return storage_root / "artifacts" / "historical-smoke" / f"{pair_slug}-{timestamp_slug}.jsonl"


__all__ = ["PairBootstrapResult", "run_alor_pair_bootstrap"]
