from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from statarb.adapters.alor.auth import (
    AlorAuthConfig,
    TokenExchangeRequest,
    TokenLifecycleManager,
)
from statarb.adapters.alor.jwt_portfolios import (
    AlorJwtPortfolioError,
    decode_jwt_payload,
    parse_jwt_portfolios,
    resolve_configured_portfolio,
)
from statarb.config import AlorContour, RuntimeSettings, load_settings
from statarb.runtime.alor_test_read_only import (
    DEFAULT_CLIENT_NAME,
    RequestsAlorAuthClient,
)


@dataclass(frozen=True, slots=True)
class JwtPortfolioDiscoveryResult:
    contour: AlorContour
    configured_portfolio: str | None
    jwt_portfolios: tuple[str, ...]
    resolved_portfolio: str | None
    jwt_subject: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "contour": self.contour.value,
            "configured_portfolio": self.configured_portfolio,
            "jwt_portfolios": list(self.jwt_portfolios),
            "resolved_portfolio": self.resolved_portfolio,
            "jwt_subject": self.jwt_subject,
            "resolution_policy": "explicit-only",
        }


class AlorAuthClient(Protocol):
    def exchange(self, request: TokenExchangeRequest) -> Mapping[str, Any]: ...


def discover_jwt_portfolios(
    settings: RuntimeSettings,
    *,
    auth_client: AlorAuthClient,
    now: datetime | None = None,
) -> JwtPortfolioDiscoveryResult:
    _require_refresh_token(settings)
    token_manager = TokenLifecycleManager(
        config=AlorAuthConfig.from_settings(
            settings.alor, client_name=DEFAULT_CLIENT_NAME
        )
    )
    issued_at = _coerce_utc_datetime(now or datetime.now(UTC))
    auth_payload = auth_client.exchange(token_manager.build_refresh_request())
    access_token = token_manager.accept_exchange_payload(auth_payload, now=issued_at)
    jwt_portfolios = parse_jwt_portfolios(access_token.value)
    payload = decode_jwt_payload(access_token.value)
    subject = payload.get("sub")
    configured = settings.alor.portfolio
    resolved: str | None = None
    if configured is not None:
        resolved = resolve_configured_portfolio(
            configured=configured,
            access_token=access_token.value,
        )
    return JwtPortfolioDiscoveryResult(
        contour=settings.alor.contour,
        configured_portfolio=configured,
        jwt_portfolios=jwt_portfolios,
        resolved_portfolio=resolved,
        jwt_subject=str(subject) if subject is not None else None,
    )


def main(
    argv: list[str] | None = None,
    *,
    auth_client: AlorAuthClient | None = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = load_settings(env_file=args.env_file)
    if args.command == "list-jwt":
        result = discover_jwt_portfolios(
            settings,
            auth_client=auth_client or RequestsAlorAuthClient(),
        )
        _print_summary(result.to_dict())
        return 0
    if args.command == "validate":
        result = discover_jwt_portfolios(
            settings,
            auth_client=auth_client or RequestsAlorAuthClient(),
        )
        if result.resolved_portfolio is None:
            raise AlorJwtPortfolioError(
                "Portfolio validation requires an explicit ALOR_TEST_PORTFOLIO or ALOR_LIVE_PORTFOLIO."
            )
        _print_summary(result.to_dict())
        return 0
    parser.error(f"Unsupported command: {args.command!r}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m statarb.runtime.alor_portfolio_resolve",
        description=(
            "Discover Alor JWT portfolios and validate explicit portfolio configuration. "
            "This helper never auto-selects a portfolio when multiple candidates exist."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_jwt = subparsers.add_parser(
        "list-jwt",
        help="Refresh an access token and print portfolios from the JWT claim.",
    )
    validate = subparsers.add_parser(
        "validate",
        help="Fail unless configured portfolio is present in the JWT portfolios claim.",
    )
    for command in (list_jwt, validate):
        command.add_argument("--env-file")
    return parser


def _require_refresh_token(settings: RuntimeSettings) -> None:
    if settings.alor.refresh_token is None:
        raise ValueError("Alor portfolio resolution requires a refresh token.")


def _coerce_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _print_summary(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
