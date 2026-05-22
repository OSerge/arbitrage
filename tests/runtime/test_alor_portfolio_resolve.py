from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from typing import Any

import pytest

from statarb import load_settings
from statarb.adapters.alor.jwt_portfolios import AlorJwtPortfolioError
from statarb.runtime.alor_portfolio_resolve import discover_jwt_portfolios, main


def _encode_payload(payload: dict[str, object]) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    return f"header.{encoded.rstrip('=')}.signature"


class StubAuthClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def exchange(self, request) -> dict[str, Any]:
        return self.payload


def test_discover_jwt_portfolios_lists_claim_without_auto_select(tmp_path) -> None:
    settings = load_settings(
        env_file=tmp_path / "missing.env",
        environ={
            "STATARB_PROFILE": "local-dev",
            "ALOR_CONTOUR": "test",
            "ALOR_TEST_REFRESH_TOKEN": "test-refresh",
        },
    )
    token = _encode_payload({"sub": "LoginSample", "portfolios": "D39004 G00123"})
    result = discover_jwt_portfolios(
        settings,
        auth_client=StubAuthClient({"AccessToken": token, "expiresIn": 1800}),
        now=datetime(2026, 5, 23, tzinfo=UTC),
    )

    summary = result.to_dict()
    assert summary["jwt_portfolios"] == ["D39004", "G00123"]
    assert summary["configured_portfolio"] is None
    assert summary["resolved_portfolio"] is None
    assert summary["resolution_policy"] == "explicit-only"


def test_discover_jwt_portfolios_validates_configured_portfolio(tmp_path) -> None:
    settings = load_settings(
        env_file=tmp_path / "missing.env",
        environ={
            "STATARB_PROFILE": "local-dev",
            "ALOR_CONTOUR": "test",
            "ALOR_TEST_REFRESH_TOKEN": "test-refresh",
            "ALOR_TEST_PORTFOLIO": "G00123",
        },
    )
    token = _encode_payload({"sub": "LoginSample", "portfolios": "D39004 G00123"})
    result = discover_jwt_portfolios(
        settings,
        auth_client=StubAuthClient({"AccessToken": token, "expiresIn": 1800}),
    )

    assert result.resolved_portfolio == "G00123"


def test_main_validate_raises_when_portfolio_missing(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "STATARB_PROFILE=local-dev",
                "ALOR_CONTOUR=test",
                "ALOR_TEST_REFRESH_TOKEN=test-refresh",
            ]
        ),
        encoding="utf-8",
    )
    token = _encode_payload({"portfolios": "D39004 G00123"})

    with pytest.raises(AlorJwtPortfolioError, match="Portfolio validation requires"):
        main(
            ["validate", "--env-file", str(env_file)],
            auth_client=StubAuthClient({"AccessToken": token, "expiresIn": 1800}),
        )
