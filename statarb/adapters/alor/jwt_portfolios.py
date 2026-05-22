from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Mapping


class AlorJwtPortfolioError(ValueError):
    """Raised when an Alor JWT access token cannot yield portfolio identifiers."""


def parse_jwt_portfolios(access_token: str) -> tuple[str, ...]:
    """Return portfolio identifiers embedded in an Alor JWT access token.

    Alor documents portfolio discovery by decoding the JWT payload and reading
    the ``portfolios`` claim as a space-separated string.
    """
    payload = decode_jwt_payload(access_token)
    return extract_portfolios_from_payload(payload)


def decode_jwt_payload(access_token: str) -> Mapping[str, Any]:
    normalized = access_token.strip()
    if not normalized:
        raise AlorJwtPortfolioError("Access token is empty.")

    parts = normalized.split(".")
    if len(parts) != 3:
        raise AlorJwtPortfolioError("Access token is not a valid JWT.")

    try:
        decoded = _decode_base64url(parts[1])
        payload = json.loads(decoded)
    except (ValueError, json.JSONDecodeError) as exc:
        raise AlorJwtPortfolioError("Access token JWT payload is not decodable JSON.") from exc

    if not isinstance(payload, Mapping):
        raise AlorJwtPortfolioError("Access token JWT payload must be a JSON object.")

    return payload


def extract_portfolios_from_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    raw_portfolios = payload.get("portfolios")
    if raw_portfolios is None:
        raise AlorJwtPortfolioError("Access token JWT payload is missing the portfolios claim.")

    if isinstance(raw_portfolios, str):
        portfolios = tuple(
            portfolio
            for portfolio in raw_portfolios.split()
            if portfolio.strip()
        )
    elif isinstance(raw_portfolios, list):
        portfolios = tuple(
            str(portfolio).strip()
            for portfolio in raw_portfolios
            if str(portfolio).strip()
        )
    else:
        raise AlorJwtPortfolioError(
            "Access token JWT portfolios claim must be a string or array."
        )

    if not portfolios:
        raise AlorJwtPortfolioError("Access token JWT portfolios claim is empty.")

    return portfolios


def resolve_configured_portfolio(
    *,
    configured: str | None,
    access_token: str,
) -> str:
    """Require an explicit portfolio and validate it against the JWT claim."""
    if configured is None:
        jwt_portfolios = parse_jwt_portfolios(access_token)
        joined = ", ".join(jwt_portfolios)
        raise AlorJwtPortfolioError(
            "Portfolio is not configured. Set ALOR_TEST_PORTFOLIO or ALOR_LIVE_PORTFOLIO "
            f"explicitly. JWT portfolios claim contains: {joined}."
        )

    normalized = configured.strip()
    if not normalized:
        raise AlorJwtPortfolioError(
            "Portfolio configuration is empty. Set ALOR_TEST_PORTFOLIO or ALOR_LIVE_PORTFOLIO "
            "to an explicit portfolio identifier."
        )

    jwt_portfolios = parse_jwt_portfolios(access_token)
    if normalized not in jwt_portfolios:
        joined = ", ".join(jwt_portfolios)
        raise AlorJwtPortfolioError(
            f"Configured portfolio {normalized!r} is not present in JWT portfolios claim: {joined}."
        )

    return normalized


def _decode_base64url(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.urlsafe_b64decode(f"{value}{padding}")
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid base64url segment.") from exc
    return decoded.decode("utf-8")


__all__ = [
    "AlorJwtPortfolioError",
    "decode_jwt_payload",
    "extract_portfolios_from_payload",
    "parse_jwt_portfolios",
    "resolve_configured_portfolio",
]
