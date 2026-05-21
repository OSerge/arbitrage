from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from statarb.adapters.alor.http_market_data import AlorHistoryBar
from statarb.data.alor_storage import AlorDatasetManifest, StoredArtifact


@dataclass(frozen=True, slots=True)
class LoadedAlorHistoryBarsDataset:
    manifest: AlorDatasetManifest
    manifest_path: Path
    normalized_artifact: StoredArtifact
    records: tuple[AlorHistoryBar, ...]


def load_history_bars_dataset(
    *,
    storage_root: Path,
    manifest_path: str | Path,
) -> LoadedAlorHistoryBarsDataset:
    resolved_manifest_path = Path(manifest_path)
    manifest_payload = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    manifest = AlorDatasetManifest(
        dataset_id=str(manifest_payload["dataset_id"]),
        dataset_kind=str(manifest_payload["dataset_kind"]),
        storage_uri=str(manifest_payload["storage_uri"]),
        raw_storage_uri=str(manifest_payload["raw_storage_uri"]),
        source=str(manifest_payload["source"]),
        endpoint=str(manifest_payload["endpoint"]),
        authorized=bool(manifest_payload["authorized"]),
        data_delay_minutes=_optional_int(manifest_payload.get("data_delay_minutes")),
        response_format=str(manifest_payload["response_format"]),
        request_params=dict(manifest_payload["request_params"]),
        row_count=int(manifest_payload["row_count"]),
        min_event_time_utc=_optional_text(manifest_payload.get("min_event_time_utc")),
        max_event_time_utc=_optional_text(manifest_payload.get("max_event_time_utc")),
        created_at_utc=str(manifest_payload["created_at_utc"]),
        source_doc_version=str(manifest_payload["source_doc_version"]),
    )
    if manifest.dataset_kind != "bars":
        raise ValueError(
            f"History bars loader only supports dataset_kind='bars', got {manifest.dataset_kind!r}."
        )

    normalized_path = Path(storage_root) / manifest.storage_uri
    frame = pd.read_parquet(normalized_path).sort_values("bar_start_utc").reset_index(drop=True)
    records = tuple(_record_from_row(row) for row in frame.to_dict(orient="records"))
    return LoadedAlorHistoryBarsDataset(
        manifest=manifest,
        manifest_path=resolved_manifest_path,
        normalized_artifact=StoredArtifact(
            path=normalized_path,
            storage_uri=manifest.storage_uri,
        ),
        records=records,
    )


def _record_from_row(row: dict[str, object]) -> AlorHistoryBar:
    return AlorHistoryBar(
        symbol=str(row["symbol"]),
        exchange=str(row["exchange"]),
        board=_optional_text(row.get("board")),
        timeframe=str(row["timeframe"]),
        time=_timestamp_to_epoch_seconds(row["bar_start_utc"]),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=_numeric_value(row["volume"]),
    )


def _timestamp_to_epoch_seconds(value: object) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp())


def _numeric_value(value: object) -> int | float:
    if isinstance(value, (int, float)):
        return value
    raise ValueError(f"Expected numeric history value, got {type(value).__name__}.")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    raise ValueError(f"Expected integer manifest value, got {type(value).__name__}.")


def _optional_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


__all__ = ["LoadedAlorHistoryBarsDataset", "load_history_bars_dataset"]
