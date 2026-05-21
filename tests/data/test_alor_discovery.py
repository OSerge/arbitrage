from __future__ import annotations

import json
from datetime import UTC, datetime

from statarb.adapters.alor.http_market_data import HistoricalBarsRequest
from statarb.data.alor_discovery import (
    discover_history_dataset_manifests,
    find_latest_history_dataset_manifest,
)
from statarb.data.alor_fetch import main
from statarb.data.alor_storage import ingest_history_payload


def test_find_latest_history_dataset_manifest_returns_newest_match(tmp_path, capsys) -> None:
    first_bar = int(datetime(2026, 5, 20, 10, 0, tzinfo=UTC).timestamp())
    second_bar = int(datetime(2026, 5, 20, 11, 0, tzinfo=UTC).timestamp())

    older = _ingest_history_dataset(
        tmp_path=tmp_path,
        symbol="SBER",
        timeframe="60",
        instrument_group="TQBR",
        first_bar=first_bar,
        second_bar=second_bar,
        observed_at=datetime(2026, 5, 21, 11, 0, tzinfo=UTC),
        request_id="sber-older",
    )
    newer = _ingest_history_dataset(
        tmp_path=tmp_path,
        symbol="SBER",
        timeframe="60",
        instrument_group="TQBR",
        first_bar=first_bar,
        second_bar=second_bar,
        observed_at=datetime(2026, 5, 21, 12, 0, tzinfo=UTC),
        request_id="sber-newer",
    )
    _ingest_history_dataset(
        tmp_path=tmp_path,
        symbol="GAZP",
        timeframe="60",
        instrument_group="TQBR",
        first_bar=first_bar,
        second_bar=second_bar,
        observed_at=datetime(2026, 5, 21, 12, 30, tzinfo=UTC),
        request_id="gazp-ignore",
    )

    matches = discover_history_dataset_manifests(
        storage_root=tmp_path,
        symbol="SBER",
        exchange="MOEX",
        timeframe="60",
    )

    assert [match.manifest.dataset_id for match in matches] == [
        newer.manifest.dataset_id,
        older.manifest.dataset_id,
    ]

    latest = find_latest_history_dataset_manifest(
        storage_root=tmp_path,
        symbol="SBER",
        exchange="MOEX",
        timeframe="60",
    )

    assert latest is not None
    assert latest.manifest_path == newer.manifest_path

    exit_code = main(
        [
            "discover",
            "--storage-root",
            str(tmp_path),
            "--symbol",
            "SBER",
            "--exchange",
            "MOEX",
            "--timeframe",
            "60",
            "--latest",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["dataset_id"] == newer.manifest.dataset_id
    assert summary["manifest_path"] == str(newer.manifest_path)


def test_discover_history_dataset_manifests_filters_by_request_metadata_and_coverage(tmp_path) -> None:
    first_bar = int(datetime(2026, 5, 20, 10, 0, tzinfo=UTC).timestamp())
    second_bar = int(datetime(2026, 5, 20, 11, 0, tzinfo=UTC).timestamp())
    third_bar = int(datetime(2026, 5, 20, 12, 0, tzinfo=UTC).timestamp())

    covered = _ingest_history_dataset(
        tmp_path=tmp_path,
        symbol="SBER",
        timeframe="60",
        instrument_group="TQBR",
        first_bar=first_bar,
        second_bar=third_bar,
        observed_at=datetime(2026, 5, 21, 12, 0, tzinfo=UTC),
        request_id="covers-window",
    )
    _ingest_history_dataset(
        tmp_path=tmp_path,
        symbol="SBER",
        timeframe="60",
        instrument_group="TQTF",
        first_bar=first_bar,
        second_bar=third_bar,
        observed_at=datetime(2026, 5, 21, 12, 5, tzinfo=UTC),
        request_id="wrong-board",
    )
    _ingest_history_dataset(
        tmp_path=tmp_path,
        symbol="SBER",
        timeframe="60",
        instrument_group="TQBR",
        first_bar=second_bar,
        second_bar=third_bar,
        observed_at=datetime(2026, 5, 21, 12, 10, tzinfo=UTC),
        request_id="too-short",
    )

    matches = discover_history_dataset_manifests(
        storage_root=tmp_path,
        symbol="SBER",
        exchange="MOEX",
        timeframe="60",
        instrument_group="TQBR",
        from_time=first_bar,
        to_time=third_bar,
    )

    assert [match.manifest_path for match in matches] == [covered.manifest_path]


def _ingest_history_dataset(
    *,
    tmp_path,
    symbol: str,
    timeframe: str,
    instrument_group: str,
    first_bar: int,
    second_bar: int,
    observed_at: datetime,
    request_id: str,
):
    payload = {
        "history": [
            {
                "time": timestamp,
                "open": 300.0 + index,
                "high": 301.0 + index,
                "low": 299.5 + index,
                "close": 300.5 + index,
                "volume": 1000 + index,
            }
            for index, timestamp in enumerate(_bar_range(first_bar, second_bar), start=1)
        ]
    }
    request = HistoricalBarsRequest(
        symbol=symbol,
        exchange="MOEX",
        instrument_group=instrument_group,
        tf=timeframe,
        from_time=first_bar,
        to_time=second_bar,
    )
    return ingest_history_payload(
        storage_root=tmp_path,
        request=request,
        payload=payload,
        fetched_at_utc=observed_at,
        request_id=request_id,
    )


def _bar_range(first_bar: int, second_bar: int) -> range:
    return range(first_bar, second_bar + 3600, 3600)
