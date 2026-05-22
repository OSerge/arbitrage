from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from statarb.config import AlorContour

from statarb.adapters.alor.auth import (
    AccessToken,
    AlorAuthConfig,
    AuthRefreshPolicy,
    TokenLifecycleManager,
)
from statarb.adapters.alor.endpoints import get_endpoint_catalog


def test_endpoint_catalog_selects_test_and_live_contours() -> None:
    test_catalog = get_endpoint_catalog(AlorContour.TEST)
    live_catalog = get_endpoint_catalog(AlorContour.LIVE)

    assert test_catalog.contour is AlorContour.TEST
    assert test_catalog.auth_base_url == "https://oauthdev.alor.ru"
    assert test_catalog.api_base_url == "https://apidev.alor.ru"
    assert test_catalog.ws_url == "wss://apidev.alor.ru/ws"
    assert test_catalog.cws_url == "wss://apidev.alor.ru/cws"

    assert live_catalog.contour is AlorContour.LIVE
    assert live_catalog.auth_base_url == "https://oauth.alor.ru"
    assert live_catalog.api_base_url == "https://api.alor.ru"
    assert live_catalog.ws_url == "wss://api.alor.ru/ws"
    assert live_catalog.cws_url == "wss://api.alor.ru/cws"


def test_token_manager_builds_refresh_request_for_selected_contour() -> None:
    manager = TokenLifecycleManager(
        config=AlorAuthConfig(
            contour=AlorContour.TEST,
            refresh_token="refresh-token",
            client_name="adapter-smoke-check",
        )
    )

    request = manager.build_refresh_request()

    assert request.method == "POST"
    assert request.url == "https://oauthdev.alor.ru/refresh"
    assert request.headers == {"Content-Type": "application/json"}
    assert request.json_body == {"token": "refresh-token"}


def test_token_manager_uses_proactive_refresh_threshold() -> None:
    issued_at = datetime(2026, 5, 21, 10, 0, tzinfo=UTC)
    manager = TokenLifecycleManager(
        config=AlorAuthConfig(
            contour=AlorContour.TEST,
            refresh_token="refresh-token",
        ),
        refresh_policy=AuthRefreshPolicy(
            access_token_ttl=timedelta(minutes=30),
            refresh_margin=timedelta(minutes=5),
        ),
    )
    manager.store_access_token(
        AccessToken(
            value="access-token",
            issued_at=issued_at,
            expires_at=issued_at + timedelta(minutes=30),
        )
    )

    assert manager.should_refresh(now=issued_at + timedelta(minutes=24)) is False
    assert manager.should_refresh(now=issued_at + timedelta(minutes=25)) is True
    assert manager.should_refresh(now=issued_at + timedelta(minutes=29, seconds=30)) is True


def test_token_manager_requires_refresh_token() -> None:
    with pytest.raises(ValueError, match="refresh token"):
        TokenLifecycleManager(
            config=AlorAuthConfig(
                contour=AlorContour.TEST,
                refresh_token="",
            )
        )
