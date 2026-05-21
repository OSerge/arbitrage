from __future__ import annotations

import pytest

from statarb.adapters.alor.http_portfolio import (
    PortfolioOrdersRequest,
    PortfolioPositionsRequest,
    PortfolioSummaryRequest,
    PortfolioTradesRequest,
)
from statarb.config import AlorContour


def test_private_positions_request_builds_authorized_http_shape() -> None:
    request = PortfolioPositionsRequest(
        portfolio="D39004",
        exchange="MOEX",
        without_currency=True,
    )

    http_request = request.to_http_request(
        contour=AlorContour.TEST,
        access_token="access-token",
    )

    assert http_request.method == "GET"
    assert http_request.url == "https://apidev.alor.ru/md/v2/Clients/MOEX/D39004/positions"
    assert http_request.headers == {
        "Accept": "application/json",
        "Authorization": "Bearer access-token",
    }
    assert http_request.params == {
        "format": "Simple",
        "withoutCurrency": True,
        "jsonResponse": True,
    }


def test_private_portfolio_requests_require_access_token() -> None:
    with pytest.raises(ValueError, match="access token"):
        PortfolioSummaryRequest(
            portfolio="D39004",
            exchange="MOEX",
        ).to_http_request(contour=AlorContour.TEST, access_token="")


def test_private_summary_orders_and_trades_requests_use_expected_paths() -> None:
    summary_request = PortfolioSummaryRequest(
        portfolio="D39004",
        exchange="MOEX",
    ).to_http_request(contour=AlorContour.TEST, access_token="access-token")
    orders_request = PortfolioOrdersRequest(
        portfolio="D39004",
        exchange="MOEX",
    ).to_http_request(contour=AlorContour.TEST, access_token="access-token")
    trades_request = PortfolioTradesRequest(
        portfolio="D39004",
        exchange="MOEX",
        with_repo=True,
    ).to_http_request(contour=AlorContour.TEST, access_token="access-token")

    assert summary_request.url.endswith("/md/v2/Clients/MOEX/D39004/summary")
    assert summary_request.params == {
        "format": "Simple",
        "jsonResponse": True,
    }
    assert orders_request.url.endswith("/md/v2/Clients/MOEX/D39004/orders")
    assert orders_request.params == {
        "format": "Simple",
        "jsonResponse": True,
    }
    assert trades_request.url.endswith("/md/v2/Clients/MOEX/D39004/trades")
    assert trades_request.params == {
        "format": "Simple",
        "withRepo": True,
        "jsonResponse": True,
    }
