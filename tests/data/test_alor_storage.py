from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime

import pandas as pd

from statarb.adapters.alor.http_market_data import (
    HistoricalBarsRequest,
    normalize_quotes_response,
    normalize_securities_response,
)
from statarb.data.alor_storage import (
    ingest_history_payload,
    persist_raw_alor_payload,
    write_quote_snapshot_dataset,
    write_security_snapshot_dataset,
)


def test_ingest_history_payload_writes_raw_parquet_and_manifest(tmp_path) -> None:
    observed_at = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    first_bar = int(datetime(2026, 5, 20, 10, 0, tzinfo=UTC).timestamp())
    second_bar = int(datetime(2026, 5, 20, 11, 0, tzinfo=UTC).timestamp())
    request = HistoricalBarsRequest(
        symbol="SBER",
        exchange="MOEX",
        instrument_group="TQBR",
        tf="60",
        from_time=first_bar,
        to_time=second_bar,
    )
    payload = {
        "history": [
            {"time": first_bar, "open": 318.5, "high": 319.1, "low": 318.2, "close": 318.9, "volume": 14500},
            {"time": second_bar, "open": 318.9, "high": 320.0, "low": 318.8, "close": 319.8, "volume": 18750},
        ],
        "next": second_bar + 3600,
        "prev": first_bar - 3600,
    }

    batch = ingest_history_payload(
        storage_root=tmp_path,
        request=request,
        payload=payload,
        fetched_at_utc=observed_at,
        request_id="history-001",
    )

    assert batch.raw_artifact.storage_uri == (
        "data/raw/vendor=alor/dataset=bars/load_date=2026-05-21/request_id=history-001/response.json.gz"
    )
    assert batch.normalized_artifact.storage_uri == (
        "data/normalized/vendor=alor/dataset=bars/exchange=MOEX/timeframe=60/date=2026-05-21/request_id=history-001/part-00000.parquet"
    )

    with gzip.open(batch.raw_artifact.path, "rt", encoding="utf-8") as handle:
        assert json.load(handle) == payload

    frame = pd.read_parquet(batch.normalized_artifact.path)
    assert list(frame["symbol"]) == ["SBER", "SBER"]
    assert list(frame["board"]) == ["TQBR", "TQBR"]
    assert frame.loc[0, "bar_start_utc"].isoformat() == datetime.fromtimestamp(first_bar, tz=UTC).isoformat()
    assert frame.loc[1, "close"] == 319.8
    assert bool(frame.loc[0, "authorized"]) is False
    assert frame.loc[0, "data_delay_minutes"] == 15

    manifest_payload = json.loads(batch.manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["dataset_id"] == "alor-bars-history-001"
    assert manifest_payload["dataset_kind"] == "bars"
    assert manifest_payload["storage_uri"] == batch.normalized_artifact.storage_uri
    assert manifest_payload["raw_storage_uri"] == batch.raw_artifact.storage_uri
    assert manifest_payload["request_params"]["symbol"] == "SBER"
    assert manifest_payload["row_count"] == 2
    assert manifest_payload["min_event_time_utc"] == datetime.fromtimestamp(first_bar, tz=UTC).isoformat()
    assert manifest_payload["max_event_time_utc"] == datetime.fromtimestamp(second_bar, tz=UTC).isoformat()


def test_snapshot_writers_use_typed_alor_records_and_manifest_metadata(tmp_path) -> None:
    observed_at = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    quote_time = int(datetime(2026, 5, 21, 11, 45, tzinfo=UTC).timestamp())

    securities_payload = [
        {
            "symbol": "SBER",
            "exchange": "MOEX",
            "market": "FOND",
            "board": "TQBR",
            "primaryBoard": "TQBR",
            "type": "Stock",
            "shortName": "Sberbank",
            "description": "Sber ordinary shares",
            "lotSize": 10,
            "minStep": 0.01,
            "faceValue": 3,
            "currency": "RUB",
            "ISIN": "RU0009029540",
            "tradingStatus": 17,
            "tradingStatusInfo": "normal session",
        }
    ]
    quotes_payload = [
        {
            "symbol": "SBER",
            "exchange": "MOEX",
            "description": "Sberbank",
            "currency": "RUB",
            "lastPriceTimestamp": quote_time,
            "openPrice": 318.1,
            "highPrice": 320.2,
            "lowPrice": 317.8,
            "lastPrice": 319.4,
            "volume": 2540000,
            "openInterest": 100,
            "bid": 319.39,
            "ask": 319.41,
            "bidVol": 120,
            "askVol": 80,
            "lotSize": 10,
            "type": "Stock",
        }
    ]

    securities_raw = persist_raw_alor_payload(
        storage_root=tmp_path,
        relative_uri=(
            "data/raw/vendor=alor/dataset=instruments/load_date=2026-05-21/request_id=securities-001/response.json.gz"
        ),
        payload=securities_payload,
    )
    quotes_raw = persist_raw_alor_payload(
        storage_root=tmp_path,
        relative_uri=(
            "data/raw/vendor=alor/dataset=quotes/load_date=2026-05-21/request_id=quotes-001/response.json.gz"
        ),
        payload=quotes_payload,
    )

    securities_batch = write_security_snapshot_dataset(
        storage_root=tmp_path,
        records=normalize_securities_response(securities_payload),
        observed_at_utc=observed_at,
        request_id="securities-001",
        raw_storage_uri=securities_raw.storage_uri,
        request_params={"query": "SBER", "exchange": "MOEX"},
    )
    quotes_batch = write_quote_snapshot_dataset(
        storage_root=tmp_path,
        records=normalize_quotes_response(quotes_payload),
        observed_at_utc=observed_at,
        request_id="quotes-001",
        raw_storage_uri=quotes_raw.storage_uri,
        request_params={"symbols": ["MOEX:SBER"]},
    )

    securities_frame = pd.read_parquet(securities_batch.normalized_artifact.path)
    assert securities_frame.loc[0, "source"] == "alor"
    assert securities_frame.loc[0, "symbol"] == "SBER"
    assert securities_frame.loc[0, "primary_board"] == "TQBR"
    assert securities_frame.loc[0, "fetched_at_utc"].isoformat() == observed_at.isoformat()

    quotes_frame = pd.read_parquet(quotes_batch.normalized_artifact.path)
    assert quotes_frame.loc[0, "symbol"] == "SBER"
    assert quotes_frame.loc[0, "quote_time_utc"].isoformat() == datetime.fromtimestamp(quote_time, tz=UTC).isoformat()
    assert quotes_frame.loc[0, "last"] == 319.4
    assert quotes_frame.loc[0, "bid_volume"] == 120

    quotes_manifest = json.loads(quotes_batch.manifest_path.read_text(encoding="utf-8"))
    assert quotes_manifest["dataset_kind"] == "quotes"
    assert quotes_manifest["request_params"] == {"symbols": ["MOEX:SBER"]}
    assert quotes_manifest["min_event_time_utc"] == datetime.fromtimestamp(quote_time, tz=UTC).isoformat()
