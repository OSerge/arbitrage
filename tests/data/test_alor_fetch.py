from __future__ import annotations

import gzip
import importlib
import json
import sys
from datetime import UTC, datetime

import pytest

from statarb.adapters.alor.http_market_data import HistoricalBarsRequest
from statarb.config import AlorContour
from statarb.data.alor_fetch import AlorHttpResponse, fetch_public_history, main


class StubHttpClient:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.calls: list[tuple[object, float]] = []

    def execute(self, request, *, timeout_seconds: float) -> AlorHttpResponse:
        self.calls.append((request, timeout_seconds))
        return AlorHttpResponse(status_code=200, payload=self.payload)


def test_fetch_public_history_persists_open_history_dataset(tmp_path) -> None:
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
    client = StubHttpClient(payload)

    result = fetch_public_history(
        storage_root=tmp_path,
        request=request,
        contour=AlorContour.LIVE,
        http_client=client,
        fetched_at_utc=observed_at,
        request_id="history-public-001",
        timeout_seconds=7.5,
    )

    assert result.http_request.url == "https://api.alor.ru/md/v2/history"
    assert result.http_request.headers == {"Accept": "application/json"}
    assert result.http_request.params["symbol"] == "SBER"
    assert result.http_status_code == 200
    assert len(client.calls) == 1
    assert client.calls[0][0] == result.http_request
    assert client.calls[0][1] == 7.5

    with gzip.open(result.batch.raw_artifact.path, "rt", encoding="utf-8") as handle:
        assert json.load(handle) == payload

    manifest_payload = json.loads(result.batch.manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["dataset_id"] == "alor-bars-history-public-001"
    assert manifest_payload["authorized"] is False
    assert manifest_payload["data_delay_minutes"] == 15
    assert manifest_payload["request_params"]["instrument_group"] == "TQBR"
    assert manifest_payload["row_count"] == 2


def test_fetch_public_history_rejects_non_object_payload(tmp_path) -> None:
    request = HistoricalBarsRequest(
        symbol="SBER",
        exchange="MOEX",
        tf="D",
        from_time=1704067200,
        to_time=1704153600,
    )
    client = StubHttpClient([{"time": 1704067200}])

    with pytest.raises(ValueError, match="JSON object"):
        fetch_public_history(
            storage_root=tmp_path,
            request=request,
            http_client=client,
            fetched_at_utc=datetime(2026, 5, 21, 12, 0, tzinfo=UTC),
        )


def test_main_history_command_prints_dataset_summary(tmp_path, capsys) -> None:
    client = StubHttpClient(
        {
            "history": [
                {
                    "time": 1704067200,
                    "open": 300.0,
                    "high": 301.0,
                    "low": 299.5,
                    "close": 300.5,
                    "volume": 1000,
                }
            ]
        }
    )

    exit_code = main(
        [
            "history",
            "--symbol",
            "SBER",
            "--exchange",
            "MOEX",
            "--timeframe",
            "D",
            "--from",
            "2024-01-01T00:00:00Z",
            "--to",
            "2024-01-02T00:00:00Z",
            "--storage-root",
            str(tmp_path),
            "--request-id",
            "cli-001",
            "--contour",
            "live",
        ],
        http_client=client,
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["dataset_id"] == "alor-bars-cli-001"
    assert summary["row_count"] == 1
    assert summary["storage_uri"].endswith("/request_id=cli-001/part-00000.parquet")
    assert summary["raw_storage_uri"].endswith("/request_id=cli-001/response.json.gz")


def test_importing_statarb_data_does_not_preload_cli_module() -> None:
    sys.modules.pop("statarb.data.alor_fetch", None)
    sys.modules.pop("statarb.data", None)

    data_module = importlib.import_module("statarb.data")

    assert "statarb.data.alor_fetch" not in sys.modules

    fetch_public_history_export = getattr(data_module, "fetch_public_history")

    assert callable(fetch_public_history_export)
    assert "statarb.data.alor_fetch" in sys.modules
