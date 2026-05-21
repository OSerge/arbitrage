from __future__ import annotations

from datetime import UTC, datetime

from statarb.adapters.alor.http_market_data import (
    AvailableBoardsRequest,
    normalize_available_boards_response,
    normalize_securities_response,
)
from statarb.data.alor_reference import resolve_symbol_board_enrichment
from statarb.data.alor_storage import (
    persist_raw_alor_payload,
    write_available_boards_dataset,
    write_security_snapshot_dataset,
)


def test_resolve_symbol_board_enrichment_prefers_primary_board_from_local_snapshots(tmp_path) -> None:
    observed_at = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    securities_payload = [
        {
            "symbol": "SBER",
            "exchange": "MOEX",
            "market": "FOND",
            "board": "TQTF",
            "primaryBoard": "TQBR",
            "type": "Stock",
            "shortName": "Sberbank",
        }
    ]
    available_boards_payload = [
        {
            "board": "TQTF",
            "primaryBoard": "TQBR",
            "instrumentGroup": "TQTF",
            "isPrimary": False,
        },
        {
            "board": "TQBR",
            "primaryBoard": "TQBR",
            "instrumentGroup": "TQBR",
            "isPrimary": True,
        },
        "SMAL",
    ]

    securities_raw = persist_raw_alor_payload(
        storage_root=tmp_path,
        relative_uri=(
            "data/raw/vendor=alor/dataset=instruments/load_date=2026-05-21/request_id=securities-resolution/response.json.gz"
        ),
        payload=securities_payload,
    )
    write_security_snapshot_dataset(
        storage_root=tmp_path,
        records=normalize_securities_response(securities_payload),
        observed_at_utc=observed_at,
        request_id="securities-resolution",
        raw_storage_uri=securities_raw.storage_uri,
        request_params={"query": "SBER", "exchange": "MOEX"},
    )

    boards_request = AvailableBoardsRequest(exchange="MOEX", symbol="SBER")
    boards_raw = persist_raw_alor_payload(
        storage_root=tmp_path,
        relative_uri=(
            "data/raw/vendor=alor/dataset=available_boards/load_date=2026-05-21/request_id=boards-resolution/response.json.gz"
        ),
        payload=available_boards_payload,
    )
    write_available_boards_dataset(
        storage_root=tmp_path,
        records=normalize_available_boards_response(available_boards_payload, request=boards_request),
        observed_at_utc=observed_at,
        request_id="boards-resolution",
        raw_storage_uri=boards_raw.storage_uri,
        request_params={"exchange": "MOEX", "symbol": "SBER"},
    )

    resolution = resolve_symbol_board_enrichment(
        storage_root=tmp_path,
        symbol="SBER",
        exchange="MOEX",
    )

    assert resolution is not None
    assert resolution.preferred_board == "TQBR"
    assert resolution.preferred_instrument_group == "TQBR"
    assert resolution.primary_board == "TQBR"
    assert set(resolution.available_boards) == {"TQBR", "TQTF", "SMAL"}
    assert resolution.market == "FOND"
    assert resolution.instrument_type == "Stock"
    assert resolution.security_manifest_path is not None
    assert resolution.available_boards_manifest_path is not None


def test_resolve_symbol_board_enrichment_falls_back_to_security_snapshot(tmp_path) -> None:
    observed_at = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    securities_payload = [
        {
            "symbol": "Si-6.25",
            "exchange": "MOEX",
            "market": "FORTS",
            "board": "RFUD",
            "primaryBoard": "RFUD",
            "type": "Futures",
            "shortName": "SiM5",
        }
    ]

    securities_raw = persist_raw_alor_payload(
        storage_root=tmp_path,
        relative_uri=(
            "data/raw/vendor=alor/dataset=instruments/load_date=2026-05-21/request_id=securities-fallback/response.json.gz"
        ),
        payload=securities_payload,
    )
    write_security_snapshot_dataset(
        storage_root=tmp_path,
        records=normalize_securities_response(securities_payload),
        observed_at_utc=observed_at,
        request_id="securities-fallback",
        raw_storage_uri=securities_raw.storage_uri,
        request_params={"query": "Si-6.25", "exchange": "MOEX"},
    )

    resolution = resolve_symbol_board_enrichment(
        storage_root=tmp_path,
        symbol="Si-6.25",
        exchange="MOEX",
    )

    assert resolution is not None
    assert resolution.preferred_board == "RFUD"
    assert resolution.preferred_instrument_group == "RFUD"
    assert resolution.available_boards == ("RFUD",)
    assert resolution.available_boards_manifest_path is None
