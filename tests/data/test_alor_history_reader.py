from __future__ import annotations

import statarb.data as data_module
from datetime import UTC, datetime

from statarb.adapters.alor.http_market_data import HistoricalBarsRequest, normalize_history_response
from statarb.data.alor_storage import ingest_history_payload


def test_load_history_bars_dataset_reads_typed_records_from_manifest(tmp_path) -> None:
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

    loader = getattr(data_module, "load_history_bars_dataset", None)
    assert loader is not None, "statarb.data.load_history_bars_dataset export is missing"

    loaded = loader(storage_root=tmp_path, manifest_path=batch.manifest_path)

    assert loaded.manifest == batch.manifest
    assert loaded.normalized_artifact.storage_uri == batch.normalized_artifact.storage_uri
    assert loaded.records == normalize_history_response(payload, request=request)
