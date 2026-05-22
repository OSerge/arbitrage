from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import requests

from statarb.adapters.alor.auth import AlorAuthConfig, TokenExchangeRequest, TokenLifecycleManager
from statarb.adapters.alor.http_market_data import (
    AlorObjectFormat,
    HttpRequestShape,
    SecuritiesCatalogRequest,
)
from statarb.adapters.alor.http_portfolio import (
    PortfolioOrdersRequest,
    PortfolioPositionsRequest,
    PortfolioSummaryRequest,
    PortfolioTradesRequest,
)
from statarb.adapters.alor.ws_portfolio import (
    OrdersSubscription,
    PositionsSubscription,
    SummariesSubscription,
    TradesSubscription,
)
from statarb.config import AlorContour, RuntimeSettings, load_settings

ACCESS_TOKEN_PLACEHOLDER = "<ACCESS_TOKEN>"
REFRESH_TOKEN_PLACEHOLDER = "<REFRESH_TOKEN>"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_CLIENT_NAME = "statarb-read-only-smoke"
PARTIAL_PUBLIC_SECURITIES_LIMIT = 3


@dataclass(frozen=True, slots=True)
class JsonHttpResponse:
    status_code: int
    payload: Any


class AlorAuthClient(Protocol):
    def exchange(self, request: TokenExchangeRequest) -> Mapping[str, Any]: ...


class AlorJsonHttpClient(Protocol):
    def execute(self, request: HttpRequestShape, *, timeout_seconds: float) -> JsonHttpResponse: ...


class RequestsAlorAuthClient:
    def __init__(self, *, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def exchange(self, request: TokenExchangeRequest) -> Mapping[str, Any]:
        response = self._session.request(
            method=request.method,
            url=request.url,
            headers=dict(request.headers),
            json=dict(request.json_body),
            timeout=request.timeout_seconds,
        )
        response.raise_for_status()
        payload = _decode_json_payload(response)
        if not isinstance(payload, Mapping):
            raise ValueError("Alor auth exchange expected a JSON object payload.")
        return payload


class RequestsAlorJsonHttpClient:
    def __init__(self, *, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def execute(self, request: HttpRequestShape, *, timeout_seconds: float) -> JsonHttpResponse:
        response = self._session.request(
            method=request.method,
            url=request.url,
            headers=dict(request.headers),
            params=dict(request.params),
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        return JsonHttpResponse(
            status_code=response.status_code,
            payload=_decode_json_payload(response),
        )


@dataclass(frozen=True, slots=True)
class PlannedHttpSurface:
    surface: str
    request: HttpRequestShape

    def to_dict(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "method": self.request.method,
            "url": self.request.url,
            "headers": dict(self.request.headers),
            "params": dict(self.request.params),
        }


@dataclass(frozen=True, slots=True)
class PlannedWsSurface:
    surface: str
    url: str
    payload: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "url": self.url,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class ExecutedHttpSurface:
    surface: str
    request: HttpRequestShape
    status_code: int
    payload: Any

    def to_dict(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "method": self.request.method,
            "url": self.request.url,
            "status_code": self.status_code,
            "payload": self.payload,
        }


@dataclass(frozen=True, slots=True)
class ReadOnlySmokePlan:
    contour: AlorContour
    exchange: str
    portfolio: str | None
    auth_request: TokenExchangeRequest
    http_surfaces: tuple[PlannedHttpSurface, ...]
    ws_surfaces: tuple[PlannedWsSurface, ...]

    def to_dict(self) -> dict[str, object]:
        smoke_scope = "portfolio_scoped" if self.portfolio else "public_only"
        return {
            "command_path": False,
            "contour": self.contour.value,
            "exchange": self.exchange,
            "portfolio": self.portfolio,
            "smoke_scope": smoke_scope,
            "auth": {
                "method": self.auth_request.method,
                "url": self.auth_request.url,
                "headers": dict(self.auth_request.headers),
                "json_body": _redact_refresh_body(self.auth_request.json_body),
                "timeout_seconds": self.auth_request.timeout_seconds,
            },
            "http": [surface.to_dict() for surface in self.http_surfaces],
            "ws": [surface.to_dict() for surface in self.ws_surfaces],
        }


@dataclass(frozen=True, slots=True)
class ReadOnlySmokeResult:
    plan: ReadOnlySmokePlan
    access_token_issued_at: datetime
    access_token_expires_at: datetime
    http_results: tuple[ExecutedHttpSurface, ...]

    def to_dict(self) -> dict[str, object]:
        payload = self.plan.to_dict()
        payload["access_token"] = {
            "issued_at": self.access_token_issued_at.isoformat(),
            "expires_at": self.access_token_expires_at.isoformat(),
            "expires_in_seconds": int(
                (self.access_token_expires_at - self.access_token_issued_at).total_seconds()
            ),
        }
        payload["http"] = [result.to_dict() for result in self.http_results]
        return payload


def build_read_only_smoke_plan(
    settings: RuntimeSettings,
    *,
    exchange: str = "MOEX",
    object_format: AlorObjectFormat | str = AlorObjectFormat.SIMPLE,
    include_orders: bool = True,
    include_trades: bool = True,
    skip_history: bool = False,
    with_repo: bool = False,
    without_currency: bool = True,
) -> ReadOnlySmokePlan:
    _require_test_read_only_settings(settings)
    token_manager = TokenLifecycleManager(
        config=AlorAuthConfig.from_settings(settings.alor, client_name=DEFAULT_CLIENT_NAME)
    )
    return ReadOnlySmokePlan(
        contour=settings.alor.contour,
        exchange=exchange,
        portfolio=settings.alor.portfolio,
        auth_request=token_manager.build_refresh_request(),
        http_surfaces=_build_http_surfaces(
            contour=settings.alor.contour,
            portfolio=settings.alor.portfolio,
            exchange=exchange,
            access_token=ACCESS_TOKEN_PLACEHOLDER,
            object_format=object_format,
            include_orders=include_orders,
            include_trades=include_trades,
            with_repo=with_repo,
            without_currency=without_currency,
        ),
        ws_surfaces=_build_ws_surfaces(
            contour=settings.alor.contour,
            portfolio=settings.alor.portfolio,
            exchange=exchange,
            access_token=ACCESS_TOKEN_PLACEHOLDER,
            include_orders=include_orders,
            include_trades=include_trades,
            skip_history=skip_history,
            with_repo=with_repo,
        ),
    )


def run_read_only_smoke(
    settings: RuntimeSettings,
    *,
    auth_client: AlorAuthClient,
    http_client: AlorJsonHttpClient,
    now: datetime | None = None,
    exchange: str = "MOEX",
    object_format: AlorObjectFormat | str = AlorObjectFormat.SIMPLE,
    include_orders: bool = True,
    include_trades: bool = True,
    skip_history: bool = False,
    with_repo: bool = False,
    without_currency: bool = True,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> ReadOnlySmokeResult:
    _require_test_read_only_settings(settings)
    token_manager = TokenLifecycleManager(
        config=AlorAuthConfig.from_settings(settings.alor, client_name=DEFAULT_CLIENT_NAME)
    )
    issued_at = _coerce_utc_datetime(now or datetime.now(UTC))
    auth_payload = auth_client.exchange(token_manager.build_refresh_request())
    access_token = token_manager.accept_exchange_payload(auth_payload, now=issued_at)
    http_surfaces = _build_http_surfaces(
        contour=settings.alor.contour,
        portfolio=settings.alor.portfolio,
        exchange=exchange,
        access_token=access_token.value,
        object_format=object_format,
        include_orders=include_orders,
        include_trades=include_trades,
        with_repo=with_repo,
        without_currency=without_currency,
    )
    http_results = tuple(
        ExecutedHttpSurface(
            surface=surface.surface,
            request=surface.request,
            status_code=response.status_code,
            payload=response.payload,
        )
        for surface in http_surfaces
        for response in (http_client.execute(surface.request, timeout_seconds=timeout_seconds),)
    )
    return ReadOnlySmokeResult(
        plan=build_read_only_smoke_plan(
            settings,
            exchange=exchange,
            object_format=object_format,
            include_orders=include_orders,
            include_trades=include_trades,
            skip_history=skip_history,
            with_repo=with_repo,
            without_currency=without_currency,
        ),
        access_token_issued_at=access_token.issued_at,
        access_token_expires_at=access_token.expires_at,
        http_results=http_results,
    )


def main(
    argv: list[str] | None = None,
    *,
    auth_client: AlorAuthClient | None = None,
    http_client: AlorJsonHttpClient | None = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = load_settings(env_file=args.env_file)
    object_format = AlorObjectFormat.coerce(args.object_format)
    if args.command == "plan":
        plan = build_read_only_smoke_plan(
            settings,
            exchange=args.exchange,
            object_format=object_format,
            include_orders=not args.skip_orders,
            include_trades=not args.skip_trades,
            skip_history=args.skip_history,
            with_repo=args.with_repo,
            without_currency=not args.include_currency,
        )
        _print_summary(plan.to_dict())
        return 0
    if args.command == "smoke":
        result = run_read_only_smoke(
            settings,
            auth_client=auth_client or RequestsAlorAuthClient(),
            http_client=http_client or RequestsAlorJsonHttpClient(),
            exchange=args.exchange,
            object_format=object_format,
            include_orders=not args.skip_orders,
            include_trades=not args.skip_trades,
            skip_history=args.skip_history,
            with_repo=args.with_repo,
            without_currency=not args.include_currency,
            timeout_seconds=args.timeout_seconds,
        )
        _print_summary(result.to_dict())
        return 0
    parser.error(f"Unsupported command: {args.command!r}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m statarb.runtime.alor_test_read_only",
        description=(
            "Prepare or run a safe read-only Alor test contour smoke-check. "
            "Without a configured portfolio, the helper falls back to a public-only partial smoke."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser(
        "plan",
        help=(
            "Print the read-only auth/http/ws plan without network I/O. "
            "Without portfolio, prints a public-only partial plan."
        ),
    )
    smoke = subparsers.add_parser(
        "smoke",
        help=(
            "Refresh an access token and execute read-only HTTP requests. "
            "Without portfolio, only the public securities probe is executed."
        ),
    )
    for command in (plan, smoke):
        command.add_argument("--env-file")
        command.add_argument("--exchange", default="MOEX")
        command.add_argument(
            "--object-format",
            default=AlorObjectFormat.SIMPLE.value,
            choices=[member.value for member in AlorObjectFormat],
        )
        command.add_argument("--skip-orders", action="store_true")
        command.add_argument("--skip-trades", action="store_true")
        command.add_argument("--skip-history", action="store_true")
        command.add_argument("--with-repo", action="store_true")
        command.add_argument("--include-currency", action="store_true")
    smoke.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    return parser


def _build_http_surfaces(
    *,
    contour: AlorContour,
    portfolio: str | None,
    exchange: str,
    access_token: str,
    object_format: AlorObjectFormat | str,
    include_orders: bool,
    include_trades: bool,
    with_repo: bool,
    without_currency: bool,
) -> tuple[PlannedHttpSurface, ...]:
    if portfolio is None:
        return (
            PlannedHttpSurface(
                surface="securities",
                request=SecuritiesCatalogRequest(
                    exchange=exchange,
                    limit=PARTIAL_PUBLIC_SECURITIES_LIMIT,
                    format=object_format,
                ).to_http_request(contour=contour, access_token=None),
            ),
        )
    surfaces = [
        PlannedHttpSurface(
            surface="positions",
            request=PortfolioPositionsRequest(
                portfolio=portfolio,
                exchange=exchange,
                format=object_format,
                without_currency=without_currency,
            ).to_http_request(contour=contour, access_token=access_token),
        ),
        PlannedHttpSurface(
            surface="summary",
            request=PortfolioSummaryRequest(
                portfolio=portfolio,
                exchange=exchange,
                format=object_format,
            ).to_http_request(contour=contour, access_token=access_token),
        ),
    ]
    if include_orders:
        surfaces.append(
            PlannedHttpSurface(
                surface="orders",
                request=PortfolioOrdersRequest(
                    portfolio=portfolio,
                    exchange=exchange,
                    format=object_format,
                ).to_http_request(contour=contour, access_token=access_token),
            )
        )
    if include_trades:
        surfaces.append(
            PlannedHttpSurface(
                surface="trades",
                request=PortfolioTradesRequest(
                    portfolio=portfolio,
                    exchange=exchange,
                    format=object_format,
                    with_repo=with_repo,
                ).to_http_request(contour=contour, access_token=access_token),
            )
        )
    return tuple(surfaces)


def _build_ws_surfaces(
    *,
    contour: AlorContour,
    portfolio: str | None,
    exchange: str,
    access_token: str,
    include_orders: bool,
    include_trades: bool,
    skip_history: bool,
    with_repo: bool,
) -> tuple[PlannedWsSurface, ...]:
    if portfolio is None:
        return ()
    ws_url = "wss://apidev.alor.ru/ws" if contour is AlorContour.TEST else "wss://api.alor.ru/ws"
    surfaces = [
        PlannedWsSurface(
            surface="positions",
            url=ws_url,
            payload=PositionsSubscription(
                guid="positions-read-only",
                portfolio=portfolio,
                exchange=exchange,
                skip_history=skip_history,
            ).to_payload(access_token=access_token),
        ),
        PlannedWsSurface(
            surface="summary",
            url=ws_url,
            payload=SummariesSubscription(
                guid="summary-read-only",
                portfolio=portfolio,
                exchange=exchange,
                skip_history=skip_history,
            ).to_payload(access_token=access_token),
        ),
    ]
    if include_orders:
        surfaces.append(
            PlannedWsSurface(
                surface="orders",
                url=ws_url,
                payload=OrdersSubscription(
                    guid="orders-read-only",
                    portfolio=portfolio,
                    exchange=exchange,
                    skip_history=skip_history,
                ).to_payload(access_token=access_token),
            )
        )
    if include_trades:
        surfaces.append(
            PlannedWsSurface(
                surface="trades",
                url=ws_url,
                payload=TradesSubscription(
                    guid="trades-read-only",
                    portfolio=portfolio,
                    exchange=exchange,
                    skip_history=skip_history,
                    with_repo=with_repo,
                ).to_payload(access_token=access_token),
            )
        )
    return tuple(surfaces)


def _decode_json_payload(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise ValueError(
            f"Alor returned a non-JSON response for {response.request.method} {response.request.url}."
        ) from exc


def _redact_refresh_body(payload: Mapping[str, str]) -> dict[str, str]:
    redacted = dict(payload)
    if "token" in redacted:
        redacted["token"] = REFRESH_TOKEN_PLACEHOLDER
    if "refreshToken" in redacted:
        redacted["refreshToken"] = REFRESH_TOKEN_PLACEHOLDER
    return redacted


def _require_test_read_only_settings(settings: RuntimeSettings) -> None:
    if settings.alor.contour is not AlorContour.TEST:
        raise ValueError("Alor read-only smoke is limited to the test contour.")
    if settings.alor.refresh_token is None:
        raise ValueError(
            "Alor read-only smoke requires a refresh token. Portfolio is optional only for the public-only partial smoke."
        )


def _coerce_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _print_summary(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
