from __future__ import annotations

from dataclasses import dataclass

from statarb.adapters.alor.endpoints import AlorEndpointCatalog, get_endpoint_catalog
from statarb.adapters.alor.http_market_data import AlorObjectFormat, HttpRequestShape
from statarb.config import AlorContour


@dataclass(frozen=True, slots=True)
class PortfolioRequest:
    portfolio: str
    exchange: str = "MOEX"
    format: AlorObjectFormat | str = AlorObjectFormat.SIMPLE
    json_response: bool = True

    def __post_init__(self) -> None:
        if not self.portfolio.strip():
            raise ValueError("Portfolio request requires a portfolio identifier.")
        if not self.exchange.strip():
            raise ValueError("Portfolio request requires an exchange.")


@dataclass(frozen=True, slots=True)
class PortfolioPositionsRequest(PortfolioRequest):
    without_currency: bool = False

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
            url=catalog.build_api_url(f"/md/v2/Clients/{self.exchange}/{self.portfolio}/positions"),
            headers=_authorized_headers(access_token),
            params={
                "format": AlorObjectFormat.coerce(self.format).value,
                "withoutCurrency": self.without_currency,
                "jsonResponse": self.json_response,
            },
        )


@dataclass(frozen=True, slots=True)
class PortfolioSummaryRequest(PortfolioRequest):
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
            url=catalog.build_api_url(f"/md/v2/Clients/{self.exchange}/{self.portfolio}/summary"),
            headers=_authorized_headers(access_token),
            params={
                "format": AlorObjectFormat.coerce(self.format).value,
                "jsonResponse": self.json_response,
            },
        )


@dataclass(frozen=True, slots=True)
class PortfolioOrdersRequest(PortfolioRequest):
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
            url=catalog.build_api_url(f"/md/v2/Clients/{self.exchange}/{self.portfolio}/orders"),
            headers=_authorized_headers(access_token),
            params={
                "format": AlorObjectFormat.coerce(self.format).value,
                "jsonResponse": self.json_response,
            },
        )


@dataclass(frozen=True, slots=True)
class PortfolioTradesRequest(PortfolioRequest):
    with_repo: bool = False

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
            url=catalog.build_api_url(f"/md/v2/Clients/{self.exchange}/{self.portfolio}/trades"),
            headers=_authorized_headers(access_token),
            params={
                "format": AlorObjectFormat.coerce(self.format).value,
                "withRepo": self.with_repo,
                "jsonResponse": self.json_response,
            },
        )


def _authorized_headers(access_token: str) -> dict[str, str]:
    normalized = access_token.strip()
    if not normalized:
        raise ValueError("Private Alor portfolio requests require an access token.")
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {normalized}",
    }


__all__ = [
    "PortfolioOrdersRequest",
    "PortfolioPositionsRequest",
    "PortfolioRequest",
    "PortfolioSummaryRequest",
    "PortfolioTradesRequest",
]
