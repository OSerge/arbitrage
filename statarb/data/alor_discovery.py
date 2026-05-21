from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from statarb.data.alor_storage import AlorDatasetManifest


@dataclass(frozen=True, slots=True)
class DiscoveredAlorHistoryDatasetManifest:
    manifest: AlorDatasetManifest
    manifest_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at_utc": self.manifest.created_at_utc,
            "dataset_id": self.manifest.dataset_id,
            "exchange": _request_param_text(self.manifest, "exchange"),
            "instrument_group": _request_param_text(self.manifest, "instrument_group"),
            "manifest_path": str(self.manifest_path),
            "manifest_uri": _manifest_uri_from_path(self.manifest_path),
            "max_event_time_utc": self.manifest.max_event_time_utc,
            "min_event_time_utc": self.manifest.min_event_time_utc,
            "normalized_storage_uri": self.manifest.storage_uri,
            "row_count": self.manifest.row_count,
            "symbol": _request_param_text(self.manifest, "symbol"),
            "timeframe": _request_param_text(self.manifest, "tf"),
        }


def discover_history_dataset_manifests(
    *,
    storage_root: Path,
    symbol: str | None = None,
    exchange: str | None = None,
    timeframe: str | None = None,
    instrument_group: str | None = None,
    from_time: int | None = None,
    to_time: int | None = None,
    request_params: Mapping[str, object] | None = None,
) -> tuple[DiscoveredAlorHistoryDatasetManifest, ...]:
    discovered: list[DiscoveredAlorHistoryDatasetManifest] = []
    normalized_root = Path(storage_root) / "data/normalized/vendor=alor/dataset=bars"
    if not normalized_root.exists():
        return ()

    for manifest_path in normalized_root.rglob("manifest.json"):
        manifest = _load_manifest(manifest_path)
        if not _matches_history_dataset(
            manifest,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            instrument_group=instrument_group,
            from_time=from_time,
            to_time=to_time,
            request_params=request_params,
        ):
            continue
        discovered.append(
            DiscoveredAlorHistoryDatasetManifest(
                manifest=manifest,
                manifest_path=manifest_path,
            )
        )

    discovered.sort(key=_sort_key, reverse=True)
    return tuple(discovered)


def find_latest_history_dataset_manifest(
    *,
    storage_root: Path,
    symbol: str | None = None,
    exchange: str | None = None,
    timeframe: str | None = None,
    instrument_group: str | None = None,
    from_time: int | None = None,
    to_time: int | None = None,
    request_params: Mapping[str, object] | None = None,
) -> DiscoveredAlorHistoryDatasetManifest | None:
    matches = discover_history_dataset_manifests(
        storage_root=storage_root,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        instrument_group=instrument_group,
        from_time=from_time,
        to_time=to_time,
        request_params=request_params,
    )
    return matches[0] if matches else None


def _matches_history_dataset(
    manifest: AlorDatasetManifest,
    *,
    symbol: str | None,
    exchange: str | None,
    timeframe: str | None,
    instrument_group: str | None,
    from_time: int | None,
    to_time: int | None,
    request_params: Mapping[str, object] | None,
) -> bool:
    if manifest.dataset_kind != "bars":
        return False
    if manifest.source != "alor":
        return False
    if symbol is not None and manifest.request_params.get("symbol") != symbol:
        return False
    if exchange is not None and manifest.request_params.get("exchange") != exchange:
        return False
    if timeframe is not None and str(manifest.request_params.get("tf")) != timeframe:
        return False
    if instrument_group is not None and manifest.request_params.get("instrument_group") != instrument_group:
        return False
    if request_params is not None:
        for key, value in request_params.items():
            if manifest.request_params.get(key) != value:
                return False
    if from_time is not None:
        min_event_time = _parse_optional_datetime(manifest.min_event_time_utc)
        if min_event_time is None or min_event_time > _epoch_to_utc_datetime(from_time):
            return False
    if to_time is not None:
        max_event_time = _parse_optional_datetime(manifest.max_event_time_utc)
        if max_event_time is None or max_event_time < _epoch_to_utc_datetime(to_time):
            return False
    return True


def _load_manifest(manifest_path: Path) -> AlorDatasetManifest:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return AlorDatasetManifest(
        dataset_id=str(payload["dataset_id"]),
        dataset_kind=str(payload["dataset_kind"]),
        storage_uri=str(payload["storage_uri"]),
        raw_storage_uri=str(payload["raw_storage_uri"]),
        source=str(payload["source"]),
        endpoint=str(payload["endpoint"]),
        authorized=bool(payload["authorized"]),
        data_delay_minutes=_optional_int(payload.get("data_delay_minutes")),
        response_format=str(payload["response_format"]),
        request_params=dict(payload["request_params"]),
        row_count=int(payload["row_count"]),
        min_event_time_utc=_optional_text(payload.get("min_event_time_utc")),
        max_event_time_utc=_optional_text(payload.get("max_event_time_utc")),
        created_at_utc=str(payload["created_at_utc"]),
        source_doc_version=str(payload["source_doc_version"]),
    )


def _sort_key(match: DiscoveredAlorHistoryDatasetManifest) -> tuple[datetime, datetime, str]:
    return (
        _parse_optional_datetime(match.manifest.created_at_utc) or datetime.fromtimestamp(0, tz=UTC),
        _parse_optional_datetime(match.manifest.max_event_time_utc) or datetime.fromtimestamp(0, tz=UTC),
        match.manifest.dataset_id,
    )


def _manifest_uri_from_path(manifest_path: Path) -> str:
    parts = manifest_path.parts
    data_index = parts.index("data")
    return "/".join(parts[data_index:])


def _request_param_text(manifest: AlorDatasetManifest, key: str) -> str | None:
    return _optional_text(manifest.request_params.get(key))


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    raise ValueError(f"Expected integer manifest value, got {type(value).__name__}.")


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _parse_optional_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _epoch_to_utc_datetime(value: int) -> datetime:
    return datetime.fromtimestamp(value, tz=UTC)


__all__ = [
    "DiscoveredAlorHistoryDatasetManifest",
    "discover_history_dataset_manifests",
    "find_latest_history_dataset_manifest",
]
