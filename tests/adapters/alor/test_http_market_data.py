from __future__ import annotations

from datetime import UTC, datetime

from statarb.adapters.alor.http_market_data import (
    AvailableBoardsRequest,
    HistoricalBarsRequest,
    QuotesSnapshotRequest,
    SecuritiesCatalogRequest,
    normalize_available_boards_response,
    normalize_history_response,
    normalize_quotes_response,
    normalize_securities_response,
    recommended_history_storage_layout,
    recommended_snapshot_storage_layout,
)
from statarb.config import AlorContour


def test_public_securities_request_uses_open_http_shape() -> None:
    request = SecuritiesCatalogRequest(
        query="SBER",
        exchange="MOEX",
        sector="FOND",
        include_non_base_boards=True,
    )

    http_request = request.to_http_request(contour=AlorContour.TEST)

    assert http_request.url == "https://apidev.alor.ru/md/v2/Securities"
    assert http_request.headers == {"Accept": "application/json"}
    assert http_request.params == {
        "query": "SBER",
        "limit": 25,
        "offset": 0,
        "sector": "FOND",
        "exchange": "MOEX",
        "includeNonBaseBoards": True,
        "format": "Heavy",
    }


def test_public_quotes_request_builds_symbol_pairs_path() -> None:
    request = QuotesSnapshotRequest.from_pairs([("MOEX", "SBER"), ("MOEX", "GAZP")])

    http_request = request.to_http_request(contour=AlorContour.TEST)

    assert (
        http_request.url
        == "https://apidev.alor.ru/md/v2/Securities/MOEX:SBER,MOEX:GAZP/quotes"
    )
    assert http_request.headers == {"Accept": "application/json"}
    assert http_request.params == {"format": "Heavy"}


def test_available_boards_request_and_normalization_use_symbol_context() -> None:
    request = AvailableBoardsRequest(exchange="MOEX", symbol="SBER")

    http_request = request.to_http_request(contour=AlorContour.TEST)
    rows = normalize_available_boards_response(
        [
            {
                "board": "TQBR",
                "primaryBoard": "TQBR",
                "instrumentGroup": "TQBR",
                "market": "FOND",
                "isPrimary": True,
            },
            {
                "boardCode": "SMAL",
                "primary_board": "TQBR",
                "boardGroup": "SMAL",
            },
        ],
        request=request,
    )

    assert http_request.url == "https://apidev.alor.ru/md/v2/Securities/MOEX/SBER/availableBoards"
    assert http_request.headers == {"Accept": "application/json"}
    assert http_request.params == {}

    assert rows[0].exchange == "MOEX"
    assert rows[0].symbol == "SBER"
    assert rows[0].board == "TQBR"
    assert rows[0].instrument_group == "TQBR"
    assert rows[0].primary_board == "TQBR"
    assert rows[0].is_primary is True

    assert rows[1].board == "SMAL"
    assert rows[1].instrument_group == "SMAL"
    assert rows[1].primary_board == "TQBR"


def test_normalize_history_response_uses_board_from_request_context() -> None:
    request = HistoricalBarsRequest(
        symbol="SBER",
        exchange="MOEX",
        instrument_group="TQBR",
        tf="60",
        from_time=1701068400,
        to_time=1701069000,
    )

    rows = normalize_history_response(
        {
            "h": [
                {
                    "t": 1701068400,
                    "o": 287.41,
                    "h": 287.90,
                    "l": 287.00,
                    "c": 287.79,
                    "v": 174466,
                }
            ]
        },
        request=request,
    )

    assert len(rows) == 1
    assert rows[0].symbol == "SBER"
    assert rows[0].exchange == "MOEX"
    assert rows[0].board == "TQBR"
    assert rows[0].timeframe == "60"
    assert rows[0].close == 287.79
    assert rows[0].volume == 174466


def test_normalize_security_and_quote_shapes_support_public_payload_variants() -> None:
    securities = normalize_securities_response(
        [
            {
                "symbol": "CNY-6.25",
                "exchange": "MOEX",
                "market": "FORTS",
                "shortName": "CRM5",
                "description": "CNY futures",
                "board": "RFUD",
                "primaryBoard": "RFUD",
                "type": "Futures",
                "lotSize": 1,
                "minStep": 0.001,
                "faceValue": 1000,
                "currency": "RUB",
                "ISIN": None,
                "tradingStatus": 17,
                "tradingStatusInfo": "normal session",
            },
            {
                "sym": "Si-6.25",
                "ex": "MOEX",
                "n": "SiM5",
                "desc": "Si futures",
                "bd": "RFUD",
                "pbd": "RFUD",
                "t": "Futures",
                "lot": 1,
                "stp": 1,
                "fv": 1000,
                "cur": "RUB",
                "st": 17,
                "sti": "normal session",
            },
        ]
    )
    quotes = normalize_quotes_response(
        [
            {
                "symbol": "SBRF-12.25",
                "exchange": "MOEX",
                "description": "Sber futures",
                "currency": "RUB",
                "lastPriceTimestamp": 1761980644,
                "openPrice": 29928,
                "highPrice": 30050,
                "lowPrice": 29685,
                "lastPrice": 30019,
                "volume": 31796,
                "openInterest": 305614,
                "ask": 30019,
                "bid": 30018,
                "askVol": 4,
                "bidVol": 10,
                "lotSize": 1,
                "type": "Future",
            }
        ]
    )

    assert securities[0].board == "RFUD"
    assert securities[0].primary_board == "RFUD"
    assert securities[1].short_name == "SiM5"
    assert securities[1].trading_status == 17

    assert quotes[0].symbol == "SBRF-12.25"
    assert quotes[0].open_interest == 305614
    assert quotes[0].bid == 30018
    assert quotes[0].ask_volume == 4


def test_recommended_storage_layouts_use_coarse_partitions() -> None:
    observed_at = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)

    history_layout = recommended_history_storage_layout(
        exchange="MOEX",
        timeframe="60",
        observed_at_utc=observed_at,
    )
    snapshot_layout = recommended_snapshot_storage_layout(
        dataset="quotes",
        observed_at_utc=observed_at,
        exchange="MOEX",
    )

    assert (
        history_layout.normalized_prefix
        == "data/normalized/vendor=alor/dataset=bars/exchange=MOEX/timeframe=60/date=2026-05-21"
    )
    assert history_layout.raw_prefix == "data/raw/vendor=alor/dataset=bars/load_date=2026-05-21"
    assert (
        snapshot_layout.normalized_prefix
        == "data/normalized/vendor=alor/dataset=quotes/as_of_date=2026-05-21/exchange=MOEX"
    )
    assert "symbol=" not in history_layout.normalized_prefix
    assert "symbol=" not in snapshot_layout.normalized_prefix
