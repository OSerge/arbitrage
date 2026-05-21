from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from statarb.adapters.alor.endpoints import AlorEndpointCatalog, get_endpoint_catalog
from statarb.config import AlorContour


@dataclass(frozen=True, slots=True)
class HttpRequestShape:
    method: str
    url: str
    headers: Mapping[str, str]
    params: Mapping[str, str | int]


@dataclass(frozen=True, slots=True)
class SecuritiesCatalogRequest:
    sector: str = "FORTS"
    exchange: str = "MOEX"
    instrument_group: str = "RFUD"
    limit: int = 50
    offset: int = 0

    def to_http_request(
        self,
        *,
        access_token: str,
        contour: AlorContour | str | None = None,
        endpoints: AlorEndpointCatalog | None = None,
    ) -> HttpRequestShape:
        catalog = endpoints or get_endpoint_catalog(contour)
        return HttpRequestShape(
            method="GET",
            url=catalog.build_api_url("/md/v2/Securities"),
            headers=_bearer_headers(access_token),
            params={
                "sector": self.sector,
                "exchange": self.exchange,
                "instrumentGroup": self.instrument_group,
                "limit": self.limit,
                "offset": self.offset,
            },
        )


@dataclass(frozen=True, slots=True)
class HistoricalBarsRequest:
    symbol: str
    exchange: str = "MOEX"
    tf: str = "D"
    from_time: str | None = None
    to_time: str | None = None

    def to_http_request(
        self,
        *,
        access_token: str,
        contour: AlorContour | str | None = None,
        endpoints: AlorEndpointCatalog | None = None,
    ) -> HttpRequestShape:
        params: dict[str, str | int] = {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "tf": self.tf,
        }
        if self.from_time is not None:
            params["from"] = self.from_time
        if self.to_time is not None:
            params["to"] = self.to_time
        catalog = endpoints or get_endpoint_catalog(contour)
        return HttpRequestShape(
            method="GET",
            url=catalog.build_api_url("/md/v2/history"),
            headers=_bearer_headers(access_token),
            params=params,
        )


def _bearer_headers(access_token: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }


__all__ = [
    "HistoricalBarsRequest",
    "HttpRequestShape",
    "SecuritiesCatalogRequest",
]
