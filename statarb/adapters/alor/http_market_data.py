from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Mapping, Sequence

from statarb.adapters.alor.endpoints import AlorEndpointCatalog, get_endpoint_catalog
from statarb.config import AlorContour


class AlorObjectFormat(StrEnum):
    SIMPLE = "Simple"
    SLIM = "Slim"
    HEAVY = "Heavy"

    @classmethod
    def coerce(cls, value: "AlorObjectFormat | str | None") -> "AlorObjectFormat":
        if value is None:
            return cls.HEAVY
        if isinstance(value, cls):
            return value
        return cls(value)


@dataclass(frozen=True, slots=True)
class HttpRequestShape:
    method: str
    url: str
    headers: Mapping[str, str]
    params: Mapping[str, str | int | bool]


RAW_MARKET_DATA_ROOT = "data/raw/vendor=alor"
NORMALIZED_MARKET_DATA_ROOT = "data/normalized/vendor=alor"
RAW_RESPONSE_FILENAME = "response.json.gz"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class RecommendedStorageLayout:
    dataset: str
    normalized_partition: tuple[str, ...]
    raw_partition: tuple[str, ...]
    raw_filename: str = RAW_RESPONSE_FILENAME
    manifest_filename: str = MANIFEST_FILENAME

    @property
    def normalized_prefix(self) -> str:
        return "/".join(
            [
                NORMALIZED_MARKET_DATA_ROOT,
                f"dataset={self.dataset}",
                *self.normalized_partition,
            ]
        )

    @property
    def raw_prefix(self) -> str:
        return "/".join(
            [
                RAW_MARKET_DATA_ROOT,
                f"dataset={self.dataset}",
                *self.raw_partition,
            ]
        )


@dataclass(frozen=True, slots=True)
class SecuritiesCatalogRequest:
    query: str | None = None
    limit: int = 25
    offset: int = 0
    sector: str | None = None
    cfi_code: str | None = None
    exchange: str | None = None
    instrument_group: str | None = None
    include_non_base_boards: bool = False
    format: AlorObjectFormat | str = AlorObjectFormat.HEAVY

    def to_http_request(
        self,
        *,
        access_token: str | None = None,
        contour: AlorContour | str | None = None,
        endpoints: AlorEndpointCatalog | None = None,
    ) -> HttpRequestShape:
        catalog = endpoints or get_endpoint_catalog(contour)
        params: dict[str, str | int | bool] = {
            "limit": self.limit,
            "offset": self.offset,
            "includeNonBaseBoards": self.include_non_base_boards,
            "format": AlorObjectFormat.coerce(self.format).value,
        }
        if self.query is not None:
            params["query"] = self.query
        if self.sector is not None:
            params["sector"] = self.sector
        if self.cfi_code is not None:
            params["cficode"] = self.cfi_code
        if self.exchange is not None:
            params["exchange"] = self.exchange
        if self.instrument_group is not None:
            params["instrumentGroup"] = self.instrument_group
        return HttpRequestShape(
            method="GET",
            url=catalog.build_api_url("/md/v2/Securities"),
            headers=_request_headers(access_token),
            params=params,
        )


@dataclass(frozen=True, slots=True)
class HistoricalBarsRequest:
    symbol: str
    exchange: str = "MOEX"
    instrument_group: str | None = None
    tf: str = "D"
    from_time: int | None = None
    to_time: int | None = None
    count_back: int | None = None
    untraded: bool | None = None
    split_adjust: bool = True
    format: AlorObjectFormat | str = AlorObjectFormat.HEAVY

    def to_http_request(
        self,
        *,
        access_token: str | None = None,
        contour: AlorContour | str | None = None,
        endpoints: AlorEndpointCatalog | None = None,
    ) -> HttpRequestShape:
        if self.from_time is None or self.to_time is None:
            raise ValueError("Historical bars request requires both from_time and to_time.")
        params: dict[str, str | int | bool] = {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "tf": self.tf,
            "from": self.from_time,
            "to": self.to_time,
            "splitAdjust": self.split_adjust,
            "format": AlorObjectFormat.coerce(self.format).value,
        }
        if self.instrument_group is not None:
            params["instrumentGroup"] = self.instrument_group
        if self.count_back is not None:
            params["countBack"] = self.count_back
        if self.untraded is not None:
            params["untraded"] = self.untraded
        catalog = endpoints or get_endpoint_catalog(contour)
        return HttpRequestShape(
            method="GET",
            url=catalog.build_api_url("/md/v2/history"),
            headers=_request_headers(access_token),
            params=params,
        )


@dataclass(frozen=True, slots=True)
class AvailableBoardsRequest:
    exchange: str
    symbol: str

    def to_http_request(
        self,
        *,
        access_token: str | None = None,
        contour: AlorContour | str | None = None,
        endpoints: AlorEndpointCatalog | None = None,
    ) -> HttpRequestShape:
        catalog = endpoints or get_endpoint_catalog(contour)
        return HttpRequestShape(
            method="GET",
            url=catalog.build_api_url(
                f"/md/v2/Securities/{self.exchange}/{self.symbol}/availableBoards"
            ),
            headers=_request_headers(access_token),
            params={},
        )


@dataclass(frozen=True, slots=True)
class QuotesSnapshotRequest:
    symbols: tuple[str, ...]
    format: AlorObjectFormat | str = AlorObjectFormat.HEAVY

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("Quotes snapshot request requires at least one symbol pair.")
        if any(":" not in item for item in self.symbols):
            raise ValueError("Quotes symbols must use the EXCHANGE:SYMBOL shape.")

    @classmethod
    def from_pairs(
        cls,
        pairs: Sequence[tuple[str, str]],
        *,
        format: AlorObjectFormat | str = AlorObjectFormat.HEAVY,
    ) -> "QuotesSnapshotRequest":
        return cls(
            symbols=tuple(f"{exchange}:{symbol}" for exchange, symbol in pairs),
            format=format,
        )

    def to_http_request(
        self,
        *,
        access_token: str | None = None,
        contour: AlorContour | str | None = None,
        endpoints: AlorEndpointCatalog | None = None,
    ) -> HttpRequestShape:
        catalog = endpoints or get_endpoint_catalog(contour)
        symbol_path = ",".join(self.symbols)
        return HttpRequestShape(
            method="GET",
            url=catalog.build_api_url(f"/md/v2/Securities/{symbol_path}/quotes"),
            headers=_request_headers(access_token),
            params={"format": AlorObjectFormat.coerce(self.format).value},
        )


@dataclass(frozen=True, slots=True)
class AlorHistoryBar:
    symbol: str
    exchange: str
    board: str | None
    timeframe: str
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: int | float


@dataclass(frozen=True, slots=True)
class AlorSecuritySnapshot:
    symbol: str
    exchange: str
    market: str | None
    board: str | None
    primary_board: str | None
    instrument_type: str | None
    short_name: str | None
    description: str | None
    lot_size: int | float | None
    min_step: int | float | None
    face_value: int | float | None
    currency: str | None
    isin: str | None
    trading_status: int | None
    trading_status_info: str | None


@dataclass(frozen=True, slots=True)
class AlorQuoteSnapshot:
    symbol: str
    exchange: str
    description: str | None
    currency: str | None
    last_price_timestamp: int | None
    open_price: int | float | None
    high_price: int | float | None
    low_price: int | float | None
    last_price: int | float | None
    volume: int | float | None
    open_interest: int | None
    bid: int | float | None
    ask: int | float | None
    bid_volume: int | float | None
    ask_volume: int | float | None
    lot_size: int | float | None
    instrument_type: str | None


def normalize_history_response(
    payload: Mapping[str, Any],
    *,
    request: HistoricalBarsRequest,
) -> tuple[AlorHistoryBar, ...]:
    history_items = payload.get("history")
    if history_items is None:
        history_items = payload.get("h")
    rows = _as_sequence(history_items, field_name="history")
    return tuple(
        AlorHistoryBar(
            symbol=request.symbol,
            exchange=request.exchange,
            board=request.instrument_group,
            timeframe=request.tf,
            time=int(_required_value(item, "time", "t")),
            open=float(_required_value(item, "open", "o")),
            high=float(_required_value(item, "high", "h")),
            low=float(_required_value(item, "low", "l")),
            close=float(_required_value(item, "close", "c")),
            volume=_numeric_value(item, "volume", "v"),
        )
        for item in (_as_mapping(entry, field_name="history entry") for entry in rows)
    )


def normalize_securities_response(
    payload: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> tuple[AlorSecuritySnapshot, ...]:
    rows = _as_sequence(payload, field_name="securities payload")
    return tuple(
        AlorSecuritySnapshot(
            symbol=str(_required_value(item, "symbol", "sym")),
            exchange=str(_required_value(item, "exchange", "ex")),
            market=_optional_text(item, "market"),
            board=_optional_text(item, "board", "bd"),
            primary_board=_optional_text(item, "primaryBoard", "primary_board", "pbd"),
            instrument_type=_optional_text(item, "type", "t"),
            short_name=_optional_text(item, "shortName", "shortname", "n"),
            description=_optional_text(item, "description", "desc"),
            lot_size=_optional_numeric(item, "lotSize", "lotsize", "lot"),
            min_step=_optional_numeric(item, "minStep", "minstep", "stp"),
            face_value=_optional_numeric(item, "faceValue", "facevalue", "fv"),
            currency=_optional_text(item, "currency", "cur"),
            isin=_optional_text(item, "ISIN", "isin"),
            trading_status=_optional_int(item, "tradingStatus", "st"),
            trading_status_info=_optional_text(item, "tradingStatusInfo", "sti"),
        )
        for item in (_as_mapping(entry, field_name="security entry") for entry in rows)
    )


def normalize_quotes_response(
    payload: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> tuple[AlorQuoteSnapshot, ...]:
    rows = _as_sequence(payload, field_name="quotes payload")
    return tuple(
        AlorQuoteSnapshot(
            symbol=str(_required_value(item, "symbol", "sym")),
            exchange=str(_required_value(item, "exchange", "ex")),
            description=_optional_text(item, "description", "desc"),
            currency=_optional_text(item, "currency"),
            last_price_timestamp=_optional_int(item, "lastPriceTimestamp", "last_price_timestamp", "tst"),
            open_price=_optional_numeric(item, "openPrice", "open_price", "o"),
            high_price=_optional_numeric(item, "highPrice", "high_price", "h"),
            low_price=_optional_numeric(item, "lowPrice", "low_price", "l"),
            last_price=_optional_numeric(item, "lastPrice", "last_price", "c"),
            volume=_optional_numeric(item, "volume", "v"),
            open_interest=_optional_int(item, "openInterest", "open_interest", "oi"),
            bid=_optional_numeric(item, "bid"),
            ask=_optional_numeric(item, "ask"),
            bid_volume=_optional_numeric(item, "bidVol", "bid_vol", "bv"),
            ask_volume=_optional_numeric(item, "askVol", "ask_vol", "av"),
            lot_size=_optional_numeric(item, "lotSize", "lotsize", "lot"),
            instrument_type=_optional_text(item, "type", "t"),
        )
        for item in (_as_mapping(entry, field_name="quote entry") for entry in rows)
    )


def recommended_history_storage_layout(
    *,
    exchange: str,
    timeframe: str,
    observed_at_utc: datetime,
) -> RecommendedStorageLayout:
    day = observed_at_utc.astimezone(UTC).date().isoformat()
    return RecommendedStorageLayout(
        dataset="bars",
        normalized_partition=(
            f"exchange={exchange}",
            f"timeframe={timeframe}",
            f"date={day}",
        ),
        raw_partition=(f"load_date={day}",),
    )


def recommended_snapshot_storage_layout(
    *,
    dataset: str,
    observed_at_utc: datetime,
    exchange: str | None = None,
) -> RecommendedStorageLayout:
    day = observed_at_utc.astimezone(UTC).date().isoformat()
    normalized_partition = [f"as_of_date={day}"]
    if exchange is not None:
        normalized_partition.append(f"exchange={exchange}")
    return RecommendedStorageLayout(
        dataset=dataset,
        normalized_partition=tuple(normalized_partition),
        raw_partition=(f"load_date={day}",),
    )


def _request_headers(access_token: str | None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    return headers


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{field_name} must be an object-like mapping.")


def _as_sequence(value: Any, *, field_name: str) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise ValueError(f"{field_name} must be a sequence.")


def _required_value(payload: Mapping[str, Any], *candidates: str) -> Any:
    for candidate in candidates:
        value = payload.get(candidate)
        if value is not None:
            return value
    joined = ", ".join(candidates)
    raise ValueError(f"Payload is missing required field among: {joined}.")


def _optional_text(payload: Mapping[str, Any], *candidates: str) -> str | None:
    value = _first_present_value(payload, *candidates)
    return None if value is None else str(value)


def _optional_numeric(payload: Mapping[str, Any], *candidates: str) -> int | float | None:
    value = _first_present_value(payload, *candidates)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    raise ValueError(f"Expected numeric value for one of: {', '.join(candidates)}.")


def _optional_int(payload: Mapping[str, Any], *candidates: str) -> int | None:
    value = _first_present_value(payload, *candidates)
    if value is None:
        return None
    if isinstance(value, int):
        return value
    raise ValueError(f"Expected integer value for one of: {', '.join(candidates)}.")


def _numeric_value(payload: Mapping[str, Any], *candidates: str) -> int | float:
    value = _required_value(payload, *candidates)
    if isinstance(value, (int, float)):
        return value
    raise ValueError(f"Expected numeric value for one of: {', '.join(candidates)}.")


def _first_present_value(payload: Mapping[str, Any], *candidates: str) -> Any | None:
    for candidate in candidates:
        if candidate in payload and payload[candidate] is not None:
            return payload[candidate]
    return None


__all__ = [
    "AlorHistoryBar",
    "AlorObjectFormat",
    "AlorQuoteSnapshot",
    "AlorSecuritySnapshot",
    "AvailableBoardsRequest",
    "HistoricalBarsRequest",
    "HttpRequestShape",
    "MANIFEST_FILENAME",
    "NORMALIZED_MARKET_DATA_ROOT",
    "QuotesSnapshotRequest",
    "RAW_MARKET_DATA_ROOT",
    "RAW_RESPONSE_FILENAME",
    "RecommendedStorageLayout",
    "SecuritiesCatalogRequest",
    "normalize_history_response",
    "normalize_quotes_response",
    "normalize_securities_response",
    "recommended_history_storage_layout",
    "recommended_snapshot_storage_layout",
]
