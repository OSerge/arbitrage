from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import pandas as pd

from statarb.adapters.alor.http_market_data import (
    RAW_RESPONSE_FILENAME,
    AlorAvailableBoardSnapshot,
    AlorHistoryBar,
    AlorObjectFormat,
    AlorQuoteSnapshot,
    AlorSecuritySnapshot,
    AvailableBoardsRequest,
    HistoricalBarsRequest,
    RecommendedStorageLayout,
    normalize_available_boards_response,
    normalize_history_response,
    normalize_securities_response,
    recommended_history_storage_layout,
    recommended_snapshot_storage_layout,
)

PARQUET_PART_FILENAME = "part-00000.parquet"
ALOR_MARKET_DATA_CONTRACT_VERSION = "2026-05-21"
NOT_APPLICABLE_RESPONSE_FORMAT = "N/A"


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    path: Path
    storage_uri: str
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class AlorDatasetManifest:
    dataset_id: str
    dataset_kind: str
    storage_uri: str
    raw_storage_uri: str
    source: str
    endpoint: str
    authorized: bool
    data_delay_minutes: int | None
    response_format: str
    request_params: Mapping[str, Any]
    row_count: int
    min_event_time_utc: str | None
    max_event_time_utc: str | None
    created_at_utc: str
    source_doc_version: str = ALOR_MARKET_DATA_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_kind": self.dataset_kind,
            "storage_uri": self.storage_uri,
            "raw_storage_uri": self.raw_storage_uri,
            "source": self.source,
            "endpoint": self.endpoint,
            "authorized": self.authorized,
            "data_delay_minutes": self.data_delay_minutes,
            "response_format": self.response_format,
            "request_params": dict(self.request_params),
            "row_count": self.row_count,
            "min_event_time_utc": self.min_event_time_utc,
            "max_event_time_utc": self.max_event_time_utc,
            "created_at_utc": self.created_at_utc,
            "source_doc_version": self.source_doc_version,
        }


@dataclass(frozen=True, slots=True)
class StoredDatasetBatch:
    raw_artifact: StoredArtifact
    normalized_artifact: StoredArtifact
    manifest_path: Path
    manifest_uri: str
    manifest: AlorDatasetManifest


def persist_raw_alor_payload(
    *,
    storage_root: Path,
    relative_uri: str,
    payload: Any,
) -> StoredArtifact:
    path = storage_root / relative_uri
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = _serialize_payload(payload)
    path.write_bytes(gzip.compress(serialized))
    return StoredArtifact(
        path=path,
        storage_uri=relative_uri,
        sha256=hashlib.sha256(serialized).hexdigest(),
    )


def write_history_bars_dataset(
    *,
    storage_root: Path,
    records: Sequence[AlorHistoryBar],
    request: HistoricalBarsRequest,
    fetched_at_utc: datetime,
    request_id: str | None = None,
    raw_storage_uri: str,
    authorized: bool = False,
    data_delay_minutes: int | None = 15,
    response_format: AlorObjectFormat | str | None = None,
) -> StoredDatasetBatch:
    dataset_request_id = request_id or uuid4().hex
    layout = recommended_history_storage_layout(
        exchange=request.exchange,
        timeframe=request.tf,
        observed_at_utc=fetched_at_utc,
    )
    rows = [
        {
            "source": "alor",
            "endpoint": "md/v2/history",
            "authorized": authorized,
            "data_delay_minutes": data_delay_minutes,
            "response_format": _coerce_format(response_format or request.format),
            "fetched_at_utc": fetched_at_utc.astimezone(UTC),
            "exchange": row.exchange,
            "symbol": row.symbol,
            "board": row.board or request.instrument_group,
            "timeframe": row.timeframe,
            "bar_start_utc": _timestamp_to_utc(row.time),
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
            "split_adjust": request.split_adjust,
            "untraded": request.untraded,
        }
        for row in records
    ]
    event_times = [row["bar_start_utc"] for row in rows]
    return _write_dataset_batch(
        storage_root=storage_root,
        layout=layout,
        request_id=dataset_request_id,
        dataset_kind="bars",
        endpoint="md/v2/history",
        rows=rows,
        raw_storage_uri=raw_storage_uri,
        request_params=_json_ready(asdict(request)),
        authorized=authorized,
        data_delay_minutes=data_delay_minutes,
        response_format=_coerce_format(response_format or request.format),
        created_at_utc=fetched_at_utc,
        event_times=event_times,
    )


def write_security_snapshot_dataset(
    *,
    storage_root: Path,
    records: Sequence[AlorSecuritySnapshot],
    observed_at_utc: datetime,
    request_id: str | None = None,
    raw_storage_uri: str,
    request_params: Mapping[str, Any],
    authorized: bool = False,
    data_delay_minutes: int | None = 15,
    response_format: AlorObjectFormat | str | None = None,
) -> StoredDatasetBatch:
    dataset_request_id = request_id or uuid4().hex
    exchange = records[0].exchange if records else _optional_text(request_params.get("exchange"))
    layout = recommended_snapshot_storage_layout(
        dataset="instruments",
        observed_at_utc=observed_at_utc,
        exchange=exchange,
    )
    rows = [
        {
            "source": "alor",
            "endpoint": "md/v2/Securities",
            "authorized": authorized,
            "data_delay_minutes": data_delay_minutes,
            "response_format": _coerce_format(response_format),
            "fetched_at_utc": observed_at_utc.astimezone(UTC),
            "exchange": row.exchange,
            "symbol": row.symbol,
            "board": row.board,
            "primary_board": row.primary_board,
            "market": row.market,
            "instrument_type": row.instrument_type,
            "short_name": row.short_name,
            "description": row.description,
            "lot_size": row.lot_size,
            "min_step": row.min_step,
            "face_value": row.face_value,
            "currency": row.currency,
            "isin": row.isin,
            "trading_status": row.trading_status,
            "trading_status_info": row.trading_status_info,
        }
        for row in records
    ]
    event_times = [observed_at_utc.astimezone(UTC)] * len(rows)
    return _write_dataset_batch(
        storage_root=storage_root,
        layout=layout,
        request_id=dataset_request_id,
        dataset_kind="instruments",
        endpoint="md/v2/Securities",
        rows=rows,
        raw_storage_uri=raw_storage_uri,
        request_params=_json_ready(dict(request_params)),
        authorized=authorized,
        data_delay_minutes=data_delay_minutes,
        response_format=_coerce_format(response_format),
        created_at_utc=observed_at_utc,
        event_times=event_times,
    )


def write_available_boards_dataset(
    *,
    storage_root: Path,
    records: Sequence[AlorAvailableBoardSnapshot],
    observed_at_utc: datetime,
    request_id: str | None = None,
    raw_storage_uri: str,
    request_params: Mapping[str, Any],
    authorized: bool = False,
    data_delay_minutes: int | None = 15,
) -> StoredDatasetBatch:
    dataset_request_id = request_id or uuid4().hex
    exchange = records[0].exchange if records else _optional_text(request_params.get("exchange"))
    layout = recommended_snapshot_storage_layout(
        dataset="available_boards",
        observed_at_utc=observed_at_utc,
        exchange=exchange,
    )
    rows = [
        {
            "source": "alor",
            "endpoint": "md/v2/Securities/{exchange}/{symbol}/availableBoards",
            "authorized": authorized,
            "data_delay_minutes": data_delay_minutes,
            "response_format": NOT_APPLICABLE_RESPONSE_FORMAT,
            "fetched_at_utc": observed_at_utc.astimezone(UTC),
            "exchange": row.exchange,
            "symbol": row.symbol,
            "board": row.board,
            "instrument_group": row.instrument_group,
            "primary_board": row.primary_board,
            "market": row.market,
            "is_primary": row.is_primary,
        }
        for row in records
    ]
    event_times = [observed_at_utc.astimezone(UTC)] * len(rows)
    return _write_dataset_batch(
        storage_root=storage_root,
        layout=layout,
        request_id=dataset_request_id,
        dataset_kind="available_boards",
        endpoint="md/v2/Securities/{exchange}/{symbol}/availableBoards",
        rows=rows,
        raw_storage_uri=raw_storage_uri,
        request_params=_json_ready(dict(request_params)),
        authorized=authorized,
        data_delay_minutes=data_delay_minutes,
        response_format=NOT_APPLICABLE_RESPONSE_FORMAT,
        created_at_utc=observed_at_utc,
        event_times=event_times,
    )


def write_quote_snapshot_dataset(
    *,
    storage_root: Path,
    records: Sequence[AlorQuoteSnapshot],
    observed_at_utc: datetime,
    request_id: str | None = None,
    raw_storage_uri: str,
    request_params: Mapping[str, Any],
    authorized: bool = False,
    data_delay_minutes: int | None = 15,
    response_format: AlorObjectFormat | str | None = None,
) -> StoredDatasetBatch:
    dataset_request_id = request_id or uuid4().hex
    exchange = records[0].exchange if records else None
    layout = recommended_snapshot_storage_layout(
        dataset="quotes",
        observed_at_utc=observed_at_utc,
        exchange=exchange,
    )
    rows = [
        {
            "source": "alor",
            "endpoint": "md/v2/Securities/{symbols}/quotes",
            "authorized": authorized,
            "data_delay_minutes": data_delay_minutes,
            "response_format": _coerce_format(response_format),
            "fetched_at_utc": observed_at_utc.astimezone(UTC),
            "exchange": row.exchange,
            "symbol": row.symbol,
            "quote_time_utc": _timestamp_to_utc(row.last_price_timestamp)
            if row.last_price_timestamp is not None
            else observed_at_utc.astimezone(UTC),
            "open": row.open_price,
            "high": row.high_price,
            "low": row.low_price,
            "last": row.last_price,
            "volume": row.volume,
            "open_interest": row.open_interest,
            "bid": row.bid,
            "ask": row.ask,
            "bid_volume": row.bid_volume,
            "ask_volume": row.ask_volume,
            "lot_size": row.lot_size,
            "instrument_type": row.instrument_type,
        }
        for row in records
    ]
    event_times = [row["quote_time_utc"] for row in rows]
    return _write_dataset_batch(
        storage_root=storage_root,
        layout=layout,
        request_id=dataset_request_id,
        dataset_kind="quotes",
        endpoint="md/v2/Securities/{symbols}/quotes",
        rows=rows,
        raw_storage_uri=raw_storage_uri,
        request_params=_json_ready(dict(request_params)),
        authorized=authorized,
        data_delay_minutes=data_delay_minutes,
        response_format=_coerce_format(response_format),
        created_at_utc=observed_at_utc,
        event_times=event_times,
    )


def ingest_history_payload(
    *,
    storage_root: Path,
    request: HistoricalBarsRequest,
    payload: Mapping[str, Any],
    fetched_at_utc: datetime,
    request_id: str | None = None,
    authorized: bool = False,
    data_delay_minutes: int | None = 15,
    response_format: AlorObjectFormat | str | None = None,
) -> StoredDatasetBatch:
    dataset_request_id = request_id or uuid4().hex
    layout = recommended_history_storage_layout(
        exchange=request.exchange,
        timeframe=request.tf,
        observed_at_utc=fetched_at_utc,
    )
    raw_artifact = persist_raw_alor_payload(
        storage_root=storage_root,
        relative_uri=_raw_storage_uri(layout=layout, request_id=dataset_request_id),
        payload=payload,
    )
    records = normalize_history_response(payload, request=request)
    return write_history_bars_dataset(
        storage_root=storage_root,
        records=records,
        request=request,
        fetched_at_utc=fetched_at_utc,
        request_id=dataset_request_id,
        raw_storage_uri=raw_artifact.storage_uri,
        authorized=authorized,
        data_delay_minutes=data_delay_minutes,
        response_format=response_format,
    )


def ingest_security_payload(
    *,
    storage_root: Path,
    payload: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    fetched_at_utc: datetime,
    request_params: Mapping[str, Any],
    request_id: str | None = None,
    authorized: bool = False,
    data_delay_minutes: int | None = 15,
    response_format: AlorObjectFormat | str | None = None,
) -> StoredDatasetBatch:
    dataset_request_id = request_id or uuid4().hex
    layout = recommended_snapshot_storage_layout(
        dataset="instruments",
        observed_at_utc=fetched_at_utc,
        exchange=_optional_text(request_params.get("exchange")),
    )
    raw_artifact = persist_raw_alor_payload(
        storage_root=storage_root,
        relative_uri=_raw_storage_uri(layout=layout, request_id=dataset_request_id),
        payload=payload,
    )
    records = normalize_securities_response(payload)
    return write_security_snapshot_dataset(
        storage_root=storage_root,
        records=records,
        observed_at_utc=fetched_at_utc,
        request_id=dataset_request_id,
        raw_storage_uri=raw_artifact.storage_uri,
        request_params=request_params,
        authorized=authorized,
        data_delay_minutes=data_delay_minutes,
        response_format=response_format,
    )


def ingest_available_boards_payload(
    *,
    storage_root: Path,
    request: AvailableBoardsRequest,
    payload: Sequence[Mapping[str, Any] | str] | Mapping[str, Any],
    fetched_at_utc: datetime,
    request_id: str | None = None,
    authorized: bool = False,
    data_delay_minutes: int | None = 15,
) -> StoredDatasetBatch:
    dataset_request_id = request_id or uuid4().hex
    layout = recommended_snapshot_storage_layout(
        dataset="available_boards",
        observed_at_utc=fetched_at_utc,
        exchange=request.exchange,
    )
    raw_artifact = persist_raw_alor_payload(
        storage_root=storage_root,
        relative_uri=_raw_storage_uri(layout=layout, request_id=dataset_request_id),
        payload=payload,
    )
    records = normalize_available_boards_response(payload, request=request)
    return write_available_boards_dataset(
        storage_root=storage_root,
        records=records,
        observed_at_utc=fetched_at_utc,
        request_id=dataset_request_id,
        raw_storage_uri=raw_artifact.storage_uri,
        request_params={"exchange": request.exchange, "symbol": request.symbol},
        authorized=authorized,
        data_delay_minutes=data_delay_minutes,
    )


def _write_dataset_batch(
    *,
    storage_root: Path,
    layout: RecommendedStorageLayout,
    request_id: str,
    dataset_kind: str,
    endpoint: str,
    rows: Sequence[Mapping[str, Any]],
    raw_storage_uri: str,
    request_params: Mapping[str, Any],
    authorized: bool,
    data_delay_minutes: int | None,
    response_format: str,
    created_at_utc: datetime,
    event_times: Sequence[datetime],
) -> StoredDatasetBatch:
    normalized_uri = _normalized_storage_uri(layout=layout, request_id=request_id)
    normalized_path = storage_root / normalized_uri
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_parquet(normalized_path, index=False, engine="pyarrow")

    manifest = AlorDatasetManifest(
        dataset_id=f"alor-{dataset_kind}-{request_id}",
        dataset_kind=dataset_kind,
        storage_uri=normalized_uri,
        raw_storage_uri=raw_storage_uri,
        source="alor",
        endpoint=endpoint,
        authorized=authorized,
        data_delay_minutes=data_delay_minutes,
        response_format=response_format,
        request_params=request_params,
        row_count=len(rows),
        min_event_time_utc=_iso_or_none(min(event_times, default=None)),
        max_event_time_utc=_iso_or_none(max(event_times, default=None)),
        created_at_utc=created_at_utc.astimezone(UTC).isoformat(),
    )
    manifest_uri = _manifest_storage_uri(layout=layout, request_id=request_id)
    manifest_path = storage_root / manifest_uri
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return StoredDatasetBatch(
        raw_artifact=StoredArtifact(
            path=storage_root / raw_storage_uri,
            storage_uri=raw_storage_uri,
        ),
        normalized_artifact=StoredArtifact(
            path=normalized_path,
            storage_uri=normalized_uri,
        ),
        manifest_path=manifest_path,
        manifest_uri=manifest_uri,
        manifest=manifest,
    )


def _serialize_payload(payload: Any) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode("utf-8")
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _timestamp_to_utc(timestamp: int | float | None) -> datetime:
    if timestamp is None:
        raise ValueError("Timestamp is required for UTC conversion.")
    return datetime.fromtimestamp(timestamp, tz=UTC)


def _iso_or_none(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat()


def _coerce_format(value: AlorObjectFormat | str | None) -> str:
    return AlorObjectFormat.coerce(value).value


def _normalized_storage_uri(*, layout: RecommendedStorageLayout, request_id: str) -> str:
    return f"{layout.normalized_prefix}/request_id={request_id}/{PARQUET_PART_FILENAME}"


def _manifest_storage_uri(*, layout: RecommendedStorageLayout, request_id: str) -> str:
    return f"{layout.normalized_prefix}/request_id={request_id}/manifest.json"


def _raw_storage_uri(*, layout: RecommendedStorageLayout, request_id: str) -> str:
    return f"{layout.raw_prefix}/request_id={request_id}/{RAW_RESPONSE_FILENAME}"


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


__all__ = [
    "ALOR_MARKET_DATA_CONTRACT_VERSION",
    "AlorDatasetManifest",
    "NOT_APPLICABLE_RESPONSE_FORMAT",
    "PARQUET_PART_FILENAME",
    "StoredArtifact",
    "StoredDatasetBatch",
    "ingest_available_boards_payload",
    "ingest_history_payload",
    "ingest_security_payload",
    "persist_raw_alor_payload",
    "write_available_boards_dataset",
    "write_history_bars_dataset",
    "write_quote_snapshot_dataset",
    "write_security_snapshot_dataset",
]
