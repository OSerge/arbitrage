"""Alor adapter shells for the agent-operated MVP."""

from statarb.adapters.alor.auth import (
    AccessToken,
    AlorAuthConfig,
    AuthRefreshPolicy,
    TokenExchangeRequest,
    TokenLifecycleManager,
)
from statarb.adapters.alor.cws_orders import (
    CwsAuthorizeRequest,
    CwsCommandEnvelope,
    GuidFactory,
)
from statarb.adapters.alor.endpoints import (
    AlorEndpointCatalog,
    LIVE_ENDPOINT_CATALOG,
    TEST_ENDPOINT_CATALOG,
    get_endpoint_catalog,
)
from statarb.adapters.alor.http_market_data import (
    HistoricalBarsRequest,
    HttpRequestShape,
    SecuritiesCatalogRequest,
)
from statarb.adapters.alor.mapper import (
    AlorPayloadMapper,
    CanonicalEventKind,
    CanonicalEventPlaceholder,
    MappingResult,
    MappingStatus,
    PlaceholderAlorMapper,
)
from statarb.adapters.alor.ws_market_data import (
    MarketDataSubscription,
    OrderBookSubscription,
    QuotesSubscription,
    WsRequestEnvelope,
)
from statarb.adapters.alor.ws_portfolio import (
    OrdersSubscription,
    PortfolioSubscription,
    PositionsSubscription,
    SummariesSubscription,
    TradesSubscription,
)

__all__ = [
    "AccessToken",
    "AlorAuthConfig",
    "AlorEndpointCatalog",
    "AlorPayloadMapper",
    "AuthRefreshPolicy",
    "CanonicalEventKind",
    "CanonicalEventPlaceholder",
    "CwsAuthorizeRequest",
    "CwsCommandEnvelope",
    "GuidFactory",
    "HistoricalBarsRequest",
    "HttpRequestShape",
    "LIVE_ENDPOINT_CATALOG",
    "MappingResult",
    "MappingStatus",
    "MarketDataSubscription",
    "OrderBookSubscription",
    "OrdersSubscription",
    "PlaceholderAlorMapper",
    "PortfolioSubscription",
    "PositionsSubscription",
    "QuotesSubscription",
    "SecuritiesCatalogRequest",
    "SummariesSubscription",
    "TEST_ENDPOINT_CATALOG",
    "TokenExchangeRequest",
    "TokenLifecycleManager",
    "TradesSubscription",
    "WsRequestEnvelope",
    "get_endpoint_catalog",
]
