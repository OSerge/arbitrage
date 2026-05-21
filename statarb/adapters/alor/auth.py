from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping

from statarb.config import AlorContour, AlorSettings

from statarb.adapters.alor.endpoints import AlorEndpointCatalog, get_endpoint_catalog


DEFAULT_ACCESS_TOKEN_TTL = timedelta(minutes=30)
DEFAULT_REFRESH_MARGIN = timedelta(minutes=5)


@dataclass(frozen=True, slots=True)
class AlorAuthConfig:
    contour: AlorContour
    refresh_token: str
    client_name: str | None = None

    def __post_init__(self) -> None:
        if not self.refresh_token.strip():
            raise ValueError("Alor auth config requires a refresh token.")

    @classmethod
    def from_settings(
        cls,
        settings: AlorSettings,
        *,
        client_name: str | None = None,
    ) -> "AlorAuthConfig":
        if settings.refresh_token is None:
            raise ValueError("Alor settings do not contain a refresh token.")
        return cls(
            contour=settings.contour,
            refresh_token=settings.refresh_token,
            client_name=client_name,
        )


@dataclass(frozen=True, slots=True)
class AuthRefreshPolicy:
    access_token_ttl: timedelta = DEFAULT_ACCESS_TOKEN_TTL
    refresh_margin: timedelta = DEFAULT_REFRESH_MARGIN

    def __post_init__(self) -> None:
        if self.access_token_ttl <= timedelta(0):
            raise ValueError("Access token TTL must be positive.")
        if self.refresh_margin < timedelta(0):
            raise ValueError("Refresh margin cannot be negative.")
        if self.refresh_margin >= self.access_token_ttl:
            raise ValueError("Refresh margin must be shorter than the token TTL.")


@dataclass(frozen=True, slots=True)
class AccessToken:
    value: str
    issued_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("Access token value cannot be empty.")
        if self.expires_at <= self.issued_at:
            raise ValueError("Access token expiry must be after issue time.")

    @property
    def expires_in(self) -> timedelta:
        return self.expires_at - self.issued_at


@dataclass(frozen=True, slots=True)
class TokenExchangeRequest:
    method: str
    url: str
    headers: Mapping[str, str]
    json_body: Mapping[str, str]
    timeout_seconds: float = 10.0


class TokenLifecycleManager:
    def __init__(
        self,
        *,
        config: AlorAuthConfig,
        refresh_policy: AuthRefreshPolicy | None = None,
        endpoints: AlorEndpointCatalog | None = None,
    ) -> None:
        self.config = config
        self.refresh_policy = refresh_policy or AuthRefreshPolicy()
        self.endpoints = endpoints or get_endpoint_catalog(config.contour)
        self._current_token: AccessToken | None = None

    @property
    def current_token(self) -> AccessToken | None:
        return self._current_token

    def build_refresh_request(self) -> TokenExchangeRequest:
        json_body = {"refreshToken": self.config.refresh_token}
        if self.config.client_name is not None:
            json_body["clientName"] = self.config.client_name
        return TokenExchangeRequest(
            method="POST",
            url=self.endpoints.build_auth_url("/refresh"),
            headers={"Content-Type": "application/json"},
            json_body=json_body,
        )

    def store_access_token(self, token: AccessToken) -> AccessToken:
        self._current_token = token
        return token

    def accept_exchange_payload(
        self,
        payload: Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> AccessToken:
        issued_at = now or datetime.now(UTC)
        access_token = _coerce_string(payload, "AccessToken", "accessToken", "token")
        expires_in_seconds = _coerce_positive_int(
            payload,
            "ExpiresIn",
            "expiresIn",
            default=int(self.refresh_policy.access_token_ttl.total_seconds()),
        )
        token = AccessToken(
            value=access_token,
            issued_at=issued_at,
            expires_at=issued_at + timedelta(seconds=expires_in_seconds),
        )
        self._current_token = token
        return token

    def should_refresh(self, *, now: datetime | None = None) -> bool:
        if self._current_token is None:
            return True
        checked_at = now or datetime.now(UTC)
        return checked_at >= self._current_token.expires_at - self.refresh_policy.refresh_margin


def _coerce_string(payload: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    joined = ", ".join(keys)
    raise ValueError(f"Token payload is missing a string value for one of: {joined}.")


def _coerce_positive_int(
    payload: Mapping[str, Any],
    *keys: str,
    default: int,
) -> int:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str) and value.strip().isdigit():
            parsed = int(value.strip())
            if parsed > 0:
                return parsed
    return default


__all__ = [
    "AccessToken",
    "AlorAuthConfig",
    "AuthRefreshPolicy",
    "DEFAULT_ACCESS_TOKEN_TTL",
    "DEFAULT_REFRESH_MARGIN",
    "TokenExchangeRequest",
    "TokenLifecycleManager",
]
