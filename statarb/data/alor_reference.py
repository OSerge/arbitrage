from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from statarb.data.alor_storage import AlorDatasetManifest


@dataclass(frozen=True, slots=True)
class AlorSymbolBoardEnrichment:
    symbol: str
    exchange: str
    preferred_board: str | None
    preferred_instrument_group: str | None
    primary_board: str | None
    available_boards: tuple[str, ...]
    market: str | None
    instrument_type: str | None
    security_manifest_path: Path | None
    available_boards_manifest_path: Path | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "preferred_board": self.preferred_board,
            "preferred_instrument_group": self.preferred_instrument_group,
            "primary_board": self.primary_board,
            "available_boards": list(self.available_boards),
            "market": self.market,
            "instrument_type": self.instrument_type,
            "security_manifest_path": None
            if self.security_manifest_path is None
            else str(self.security_manifest_path),
            "available_boards_manifest_path": None
            if self.available_boards_manifest_path is None
            else str(self.available_boards_manifest_path),
        }


def resolve_symbol_board_enrichment(
    *,
    storage_root: Path,
    symbol: str,
    exchange: str,
) -> AlorSymbolBoardEnrichment | None:
    resolved_storage_root = Path(storage_root)
    security_row, security_manifest_path = _find_latest_security_row(
        storage_root=resolved_storage_root,
        symbol=symbol,
        exchange=exchange,
    )
    available_board_rows, available_boards_manifest_path = _find_latest_available_boards_rows(
        storage_root=resolved_storage_root,
        symbol=symbol,
        exchange=exchange,
    )

    if security_row is None and not available_board_rows:
        return None

    market = _row_value(security_row, "market") if security_row is not None else None
    instrument_type = _row_value(security_row, "instrument_type") if security_row is not None else None
    security_board = _row_value(security_row, "board") if security_row is not None else None
    security_primary_board = _row_value(security_row, "primary_board") if security_row is not None else None

    primary_from_available = next(
        (
            row.board
            for row in available_board_rows
            if row.is_primary is True or (row.primary_board is not None and row.primary_board == row.board)
        ),
        None,
    )
    primary_board = security_primary_board or primary_from_available

    board_candidates = [
        *(row.board for row in available_board_rows if row.board is not None),
        security_primary_board,
        security_board,
    ]
    available_boards = _dedupe_non_empty(board_candidates)
    preferred_board = next(
        (
            candidate
            for candidate in (primary_board, primary_from_available, security_board, *available_boards)
            if candidate is not None
        ),
        None,
    )
    ordered_boards = _promote_to_front(available_boards, preferred_board)

    instrument_group_by_board = {
        row.board: row.instrument_group
        for row in available_board_rows
        if row.board is not None and row.instrument_group is not None
    }
    preferred_instrument_group = (
        instrument_group_by_board.get(preferred_board) if preferred_board is not None else None
    ) or preferred_board

    return AlorSymbolBoardEnrichment(
        symbol=symbol,
        exchange=exchange,
        preferred_board=preferred_board,
        preferred_instrument_group=preferred_instrument_group,
        primary_board=primary_board,
        available_boards=ordered_boards,
        market=market,
        instrument_type=instrument_type,
        security_manifest_path=security_manifest_path,
        available_boards_manifest_path=available_boards_manifest_path,
    )


@dataclass(frozen=True, slots=True)
class _AvailableBoardRow:
    board: str | None
    instrument_group: str | None
    primary_board: str | None
    is_primary: bool | None


def _find_latest_security_row(
    *,
    storage_root: Path,
    symbol: str,
    exchange: str,
) -> tuple[pd.Series | None, Path | None]:
    for manifest_path, manifest in _iter_dataset_manifests(storage_root=storage_root, dataset="instruments"):
        frame = pd.read_parquet(storage_root / manifest.storage_uri)
        matches = frame[
            (frame["symbol"].astype(str) == symbol) & (frame["exchange"].astype(str) == exchange)
        ]
        if matches.empty:
            continue
        preferred_matches = matches[
            matches["primary_board"].notna() & (matches["board"].astype(str) == matches["primary_board"].astype(str))
        ]
        chosen = preferred_matches.iloc[0] if not preferred_matches.empty else matches.iloc[0]
        return chosen, manifest_path
    return None, None


def _find_latest_available_boards_rows(
    *,
    storage_root: Path,
    symbol: str,
    exchange: str,
) -> tuple[tuple[_AvailableBoardRow, ...], Path | None]:
    for manifest_path, manifest in _iter_dataset_manifests(
        storage_root=storage_root,
        dataset="available_boards",
    ):
        frame = pd.read_parquet(storage_root / manifest.storage_uri)
        matches = frame[
            (frame["symbol"].astype(str) == symbol) & (frame["exchange"].astype(str) == exchange)
        ]
        if matches.empty:
            continue
        rows = tuple(
            _AvailableBoardRow(
                board=_row_value(row, "board"),
                instrument_group=_row_value(row, "instrument_group"),
                primary_board=_row_value(row, "primary_board"),
                is_primary=_row_bool(row, "is_primary"),
            )
            for _, row in matches.iterrows()
        )
        return rows, manifest_path
    return (), None


def _iter_dataset_manifests(
    *,
    storage_root: Path,
    dataset: str,
) -> tuple[tuple[Path, AlorDatasetManifest], ...]:
    root = storage_root / "data" / "normalized" / "vendor=alor" / f"dataset={dataset}"
    if not root.exists():
        return ()

    manifests = []
    for manifest_path in root.rglob("manifest.json"):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = AlorDatasetManifest(
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
        manifests.append((manifest_path, manifest))

    manifests.sort(key=lambda item: (_parse_optional_datetime(item[1].created_at_utc), item[1].dataset_id), reverse=True)
    return tuple(manifests)


def _row_value(row: pd.Series | None, column: str) -> str | None:
    if row is None:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return str(value)


def _row_bool(row: pd.Series, column: str) -> bool | None:
    value = row[column]
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    raise ValueError(f"Expected boolean value in column {column!r}, got {type(value).__name__}.")


def _dedupe_non_empty(values: list[str | None]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _promote_to_front(values: tuple[str, ...], preferred: str | None) -> tuple[str, ...]:
    if preferred is None or preferred not in values:
        return values
    return (preferred, *(value for value in values if value != preferred))


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    raise ValueError(f"Expected integer manifest value, got {type(value).__name__}.")


def _parse_optional_datetime(value: str | None) -> datetime:
    if value is None:
        return datetime.fromtimestamp(0, tz=UTC)
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


__all__ = ["AlorSymbolBoardEnrichment", "resolve_symbol_board_enrichment"]
