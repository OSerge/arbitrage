from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from statarb import load_settings
from statarb.runtime.alor_test_read_only import build_read_only_smoke_plan, main, run_read_only_smoke


class StubAuthClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[object] = []

    def exchange(self, request) -> dict[str, Any]:
        self.calls.append(request)
        return self.payload


class StubHttpClient:
    def __init__(self, payloads_by_suffix: dict[str, Any]) -> None:
        self.payloads_by_suffix = payloads_by_suffix
        self.calls: list[tuple[object, float]] = []

    def execute(self, request, *, timeout_seconds: float):
        self.calls.append((request, timeout_seconds))
        for suffix, payload in self.payloads_by_suffix.items():
            if request.url.endswith(suffix):
                from statarb.runtime.alor_test_read_only import JsonHttpResponse

                return JsonHttpResponse(status_code=200, payload=payload)
        raise AssertionError(f"Unexpected request URL: {request.url}")


def test_build_read_only_smoke_plan_requires_test_contour_and_refresh_token(tmp_path: Path) -> None:
    live_settings = load_settings(
        env_file=tmp_path / "missing.env",
        environ={
            "STATARB_PROFILE": "local-dev",
            "ALOR_CONTOUR": "live",
            "ALOR_LIVE_REFRESH_TOKEN": "live-refresh",
            "ALOR_LIVE_PORTFOLIO": "D39004",
            "STATARB_ENABLE_LIVE_TRADING": "true",
        }
    )
    missing_refresh_settings = load_settings(
        env_file=tmp_path / "missing.env",
        environ={
            "STATARB_PROFILE": "local-dev",
            "ALOR_CONTOUR": "test",
            "ALOR_TEST_PORTFOLIO": "D39004",
        }
    )

    with pytest.raises(ValueError, match="test contour"):
        build_read_only_smoke_plan(live_settings)
    with pytest.raises(ValueError, match="refresh token"):
        build_read_only_smoke_plan(missing_refresh_settings)


def test_build_read_only_smoke_plan_without_portfolio_uses_public_only_partial_scope(tmp_path: Path) -> None:
    settings = load_settings(
        env_file=tmp_path / "missing.env",
        environ={
            "STATARB_PROFILE": "local-dev",
            "ALOR_CONTOUR": "test",
            "ALOR_TEST_REFRESH_TOKEN": "test-refresh",
        }
    )

    summary = build_read_only_smoke_plan(settings).to_dict()

    assert summary["portfolio"] is None
    assert summary["smoke_scope"] == "public_only"
    assert [item["surface"] for item in summary["http"]] == ["securities"]
    assert summary["ws"] == []


def test_run_read_only_smoke_exchanges_token_and_hits_read_only_account_surfaces(tmp_path: Path) -> None:
    settings = load_settings(
        env_file=tmp_path / "missing.env",
        environ={
            "STATARB_PROFILE": "local-dev",
            "ALOR_CONTOUR": "test",
            "ALOR_TEST_REFRESH_TOKEN": "test-refresh",
            "ALOR_TEST_PORTFOLIO": "D39004",
        }
    )
    auth_client = StubAuthClient(
        {
            "AccessToken": "access-token",
            "ExpiresIn": 1800,
        }
    )
    http_client = StubHttpClient(
        {
            "/positions": [{"symbol": "SBER", "qty": 1}],
            "/summary": {"portfolio": "D39004", "cash": 100000.0},
            "/orders": [{"id": "order-1", "status": "working"}],
            "/trades": [{"id": "trade-1", "symbol": "SBER"}],
        }
    )

    result = run_read_only_smoke(
        settings,
        auth_client=auth_client,
        http_client=http_client,
        now=datetime(2026, 5, 21, 12, 0, tzinfo=UTC),
    )
    summary = result.to_dict()

    assert auth_client.calls[0].url == "https://oauthdev.alor.ru/refresh"
    assert summary["command_path"] is False
    assert summary["contour"] == "test"
    assert summary["portfolio"] == "D39004"
    assert summary["smoke_scope"] == "portfolio_scoped"
    assert summary["access_token"]["expires_at"] == "2026-05-21T12:30:00+00:00"
    assert [item["surface"] for item in summary["http"]] == [
        "positions",
        "summary",
        "orders",
        "trades",
    ]
    assert [item["surface"] for item in summary["ws"]] == [
        "positions",
        "summary",
        "orders",
        "trades",
    ]
    assert summary["ws"][0]["payload"]["token"] == "<ACCESS_TOKEN>"
    assert http_client.calls[0][0].headers["Authorization"] == "Bearer access-token"
    assert http_client.calls[0][1] == 10.0


def test_run_read_only_smoke_without_portfolio_executes_public_only_partial_probe(
    tmp_path: Path,
) -> None:
    settings = load_settings(
        env_file=tmp_path / "missing.env",
        environ={
            "STATARB_PROFILE": "local-dev",
            "ALOR_CONTOUR": "test",
            "ALOR_TEST_REFRESH_TOKEN": "test-refresh",
        }
    )
    auth_client = StubAuthClient(
        {
            "AccessToken": "access-token",
            "ExpiresIn": 1800,
        }
    )
    http_client = StubHttpClient(
        {
            "/Securities": [{"symbol": "SBER", "exchange": "MOEX"}],
        }
    )

    result = run_read_only_smoke(
        settings,
        auth_client=auth_client,
        http_client=http_client,
        now=datetime(2026, 5, 21, 12, 0, tzinfo=UTC),
    )
    summary = result.to_dict()

    assert auth_client.calls[0].url == "https://oauthdev.alor.ru/refresh"
    assert summary["portfolio"] is None
    assert summary["smoke_scope"] == "public_only"
    assert [item["surface"] for item in summary["http"]] == ["securities"]
    assert summary["ws"] == []
    assert http_client.calls[0][0].headers == {"Accept": "application/json"}
    assert http_client.calls[0][1] == 10.0


def test_main_plan_command_prints_redacted_read_only_smoke_plan(tmp_path: Path, capsys) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "STATARB_PROFILE=local-dev",
                "ALOR_CONTOUR=test",
                "ALOR_TEST_REFRESH_TOKEN=test-refresh",
                "ALOR_TEST_PORTFOLIO=D39004",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "plan",
            "--env-file",
            str(env_file),
            "--skip-trades",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["command_path"] is False
    assert summary["contour"] == "test"
    assert summary["smoke_scope"] == "portfolio_scoped"
    assert summary["auth"]["json_body"] == {"token": "<REFRESH_TOKEN>"}
    assert [item["surface"] for item in summary["http"]] == ["positions", "summary", "orders"]
    assert summary["ws"][0]["payload"]["token"] == "<ACCESS_TOKEN>"


def test_main_plan_command_without_portfolio_prints_public_only_partial_plan(
    tmp_path: Path, capsys
) -> None:
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

    exit_code = main(
        [
            "plan",
            "--env-file",
            str(env_file),
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["portfolio"] is None
    assert summary["smoke_scope"] == "public_only"
    assert summary["auth"]["json_body"] == {"token": "<REFRESH_TOKEN>"}
    assert [item["surface"] for item in summary["http"]] == ["securities"]
    assert summary["ws"] == []
