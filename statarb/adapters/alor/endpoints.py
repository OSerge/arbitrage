from __future__ import annotations

from dataclasses import dataclass

from statarb.config import AlorContour


@dataclass(frozen=True, slots=True)
class AlorEndpointCatalog:
    contour: AlorContour
    auth_base_url: str
    api_base_url: str
    ws_url: str
    cws_url: str

    def build_auth_url(self, path: str) -> str:
        return _join_url(self.auth_base_url, path)

    def build_api_url(self, path: str) -> str:
        return _join_url(self.api_base_url, path)


TEST_ENDPOINT_CATALOG = AlorEndpointCatalog(
    contour=AlorContour.TEST,
    auth_base_url="https://oauthdev.alor.ru",
    api_base_url="https://apidev.alor.ru",
    ws_url="wss://apidev.alor.ru/ws",
    cws_url="wss://apidev.alor.ru/cws",
)

LIVE_ENDPOINT_CATALOG = AlorEndpointCatalog(
    contour=AlorContour.LIVE,
    auth_base_url="https://oauth.alor.ru",
    api_base_url="https://api.alor.ru",
    ws_url="wss://api.alor.ru/ws",
    cws_url="wss://api.alor.ru/cws",
)

ENDPOINTS_BY_CONTOUR: dict[AlorContour, AlorEndpointCatalog] = {
    TEST_ENDPOINT_CATALOG.contour: TEST_ENDPOINT_CATALOG,
    LIVE_ENDPOINT_CATALOG.contour: LIVE_ENDPOINT_CATALOG,
}


def get_endpoint_catalog(contour: AlorContour | str | None) -> AlorEndpointCatalog:
    return ENDPOINTS_BY_CONTOUR[AlorContour.coerce(contour)]


def _join_url(base_url: str, path: str) -> str:
    if not path:
        return base_url.rstrip("/")
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


__all__ = [
    "AlorEndpointCatalog",
    "ENDPOINTS_BY_CONTOUR",
    "LIVE_ENDPOINT_CATALOG",
    "TEST_ENDPOINT_CATALOG",
    "get_endpoint_catalog",
]
