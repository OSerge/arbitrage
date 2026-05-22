from __future__ import annotations

import base64
import json

import pytest

from statarb.adapters.alor.jwt_portfolios import (
    AlorJwtPortfolioError,
    decode_jwt_payload,
    extract_portfolios_from_payload,
    parse_jwt_portfolios,
    resolve_configured_portfolio,
)


def _encode_payload(payload: dict[str, object]) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    return f"header.{encoded.rstrip('=')}.signature"


def test_parse_jwt_portfolios_reads_space_separated_claim() -> None:
    token = _encode_payload({"portfolios": "D39004 G00123 H99999"})

    assert parse_jwt_portfolios(token) == ("D39004", "G00123", "H99999")


def test_parse_jwt_portfolios_reads_array_claim() -> None:
    token = _encode_payload({"portfolios": ["D39004", "G00123"]})

    assert parse_jwt_portfolios(token) == ("D39004", "G00123")


def test_resolve_configured_portfolio_requires_explicit_value() -> None:
    token = _encode_payload({"portfolios": "D39004 G00123"})

    with pytest.raises(AlorJwtPortfolioError, match="Portfolio is not configured"):
        resolve_configured_portfolio(configured=None, access_token=token)


def test_resolve_configured_portfolio_rejects_unknown_portfolio() -> None:
    token = _encode_payload({"portfolios": "D39004 G00123"})

    with pytest.raises(AlorJwtPortfolioError, match="not present in JWT portfolios claim"):
        resolve_configured_portfolio(configured="UNKNOWN", access_token=token)


def test_resolve_configured_portfolio_accepts_configured_value() -> None:
    token = _encode_payload({"portfolios": "D39004 G00123"})

    assert resolve_configured_portfolio(configured="G00123", access_token=token) == "G00123"


def test_decode_jwt_payload_rejects_invalid_token() -> None:
    with pytest.raises(AlorJwtPortfolioError, match="not a valid JWT"):
        decode_jwt_payload("not-a-jwt")


def test_extract_portfolios_from_payload_rejects_missing_claim() -> None:
    with pytest.raises(AlorJwtPortfolioError, match="missing the portfolios claim"):
        extract_portfolios_from_payload({})
