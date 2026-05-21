from __future__ import annotations

import gzip
import importlib
import json
import sys
from datetime import UTC, datetime

import pytest

from statarb.adapters.alor.http_market_data import (
    AvailableBoardsRequest,
    HistoricalBarsRequest,
    normalize_available_boards_response,
    normalize_securities_response,
)
from statarb.config import AlorContour
from statarb.data.alor_fetch import AlorHttpResponse, fetch_public_history, main
from statarb.data.alor_storage import (
    persist_raw_alor_payload,
    write_available_boards_dataset,
    write_security_snapshot_dataset,
)


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


def test_main_securities_command_prints_dataset_summary(tmp_path, capsys) -> None:
    client = StubHttpClient(
        [
            {
                "symbol": "SBER",
                "exchange": "MOEX",
                "market": "FOND",
                "board": "TQBR",
                "primaryBoard": "TQBR",
                "type": "Stock",
                "shortName": "Sberbank",
            }
        ]
    )

    exit_code = main(
        [
            "securities",
            "--query",
            "SBER",
            "--exchange",
            "MOEX",
            "--storage-root",
            str(tmp_path),
            "--request-id",
            "securities-cli-001",
            "--contour",
            "live",
        ],
        http_client=client,
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["dataset_id"] == "alor-instruments-securities-cli-001"
    assert summary["row_count"] == 1
    assert summary["symbol_count"] == 1
    assert summary["storage_uri"].endswith("/request_id=securities-cli-001/part-00000.parquet")


def test_main_resolve_symbol_command_prints_preferred_board(tmp_path, capsys) -> None:
    observed_at = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    securities_payload = [
        {
            "symbol": "SBER",
            "exchange": "MOEX",
            "market": "FOND",
            "board": "SMAL",
            "primaryBoard": "TQBR",
            "type": "Stock",
            "shortName": "Sberbank",
        }
    ]
    available_boards_payload = [
        {
            "board": "TQBR",
            "primaryBoard": "TQBR",
            "instrumentGroup": "TQBR",
            "isPrimary": True,
        },
        {
            "board": "SMAL",
            "primaryBoard": "TQBR",
        },
    ]

    securities_raw = persist_raw_alor_payload(
        storage_root=tmp_path,
        relative_uri=(
            "data/raw/vendor=alor/dataset=instruments/load_date=2026-05-21/request_id=resolve-securities/response.json.gz"
        ),
        payload=securities_payload,
    )
    write_security_snapshot_dataset(
        storage_root=tmp_path,
        records=normalize_securities_response(securities_payload),
        observed_at_utc=observed_at,
        request_id="resolve-securities",
        raw_storage_uri=securities_raw.storage_uri,
        request_params={"query": "SBER", "exchange": "MOEX"},
    )

    boards_raw = persist_raw_alor_payload(
        storage_root=tmp_path,
        relative_uri=(
            "data/raw/vendor=alor/dataset=available_boards/load_date=2026-05-21/request_id=resolve-boards/response.json.gz"
        ),
        payload=available_boards_payload,
    )
    write_available_boards_dataset(
        storage_root=tmp_path,
        records=normalize_available_boards_response(
            available_boards_payload,
            request=AvailableBoardsRequest(exchange="MOEX", symbol="SBER"),
        ),
        observed_at_utc=observed_at,
        request_id="resolve-boards",
        raw_storage_uri=boards_raw.storage_uri,
        request_params={"exchange": "MOEX", "symbol": "SBER"},
    )

    exit_code = main(
        [
            "resolve-symbol",
            "--storage-root",
            str(tmp_path),
            "--symbol",
            "SBER",
            "--exchange",
            "MOEX",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["symbol"] == "SBER"
    assert summary["preferred_board"] == "TQBR"
    assert summary["preferred_instrument_group"] == "TQBR"
    assert sorted(summary["available_boards"]) == ["SMAL", "TQBR"]


def test_importing_statarb_data_does_not_preload_cli_module() -> None:
    sys.modules.pop("statarb.data.alor_fetch", None)
    sys.modules.pop("statarb.data", None)

    data_module = importlib.import_module("statarb.data")

    assert "statarb.data.alor_fetch" not in sys.modules

    fetch_public_history_export = getattr(data_module, "fetch_public_history")

    assert callable(fetch_public_history_export)
    assert "statarb.data.alor_fetch" in sys.modules
