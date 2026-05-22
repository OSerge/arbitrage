from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from statarb import load_settings
from statarb.runtime.alor_smoke_preview import build_canonical_preview
from statarb.runtime.alor_test_read_only import main, run_read_only_smoke
from tests.adapters.alor.fixtures.portfolio_payloads import SUMMARY_D00014


class StubAuthClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def exchange(self, request) -> dict[str, Any]:
        return self.payload


class StubHttpClient:
    def __init__(self, payloads_by_suffix: dict[str, Any]) -> None:
        self.payloads_by_suffix = payloads_by_suffix

    def execute(self, request, *, timeout_seconds: float):
        from statarb.runtime.alor_test_read_only import JsonHttpResponse

        for suffix, payload in self.payloads_by_suffix.items():
            if request.url.endswith(suffix):
                return JsonHttpResponse(status_code=200, payload=payload)
        raise AssertionError(f"Unexpected request URL: {request.url}")


def test_build_canonical_preview_maps_portfolio_scoped_http_results() -> None:
    class Result:
        surface = "summary"
        status_code = 200
        payload = SUMMARY_D00014

    preview = build_canonical_preview(
        http_results=[Result()],
        portfolio="D00014",
        exchange="MOEX",
        smoke_scope="portfolio_scoped",
        captured_at=datetime(2026, 5, 23, 12, 0, tzinfo=UTC),
    )

    assert preview.total_events == 1
    assert preview.surfaces[0].surface == "summary"
    assert preview.surfaces[0].events[0]["event_type"] == "account_summary"
    assert preview.surfaces[0].events[0]["payload"]["account_id"] == "D00014"


def test_build_canonical_preview_returns_empty_for_public_only_scope() -> None:
    preview = build_canonical_preview(
        http_results=[],
        portfolio=None,
        exchange="MOEX",
        smoke_scope="public_only",
    )

    assert preview.total_events == 0
    assert preview.surfaces == ()


def test_run_read_only_smoke_with_canonical_preview(tmp_path: Path) -> None:
    settings = load_settings(
        env_file=tmp_path / "missing.env",
        environ={
            "STATARB_PROFILE": "local-dev",
            "ALOR_CONTOUR": "test",
            "ALOR_TEST_REFRESH_TOKEN": "test-refresh",
            "ALOR_TEST_PORTFOLIO": "D00014",
        },
    )
    result = run_read_only_smoke(
        settings,
        auth_client=StubAuthClient({"AccessToken": "access-token", "ExpiresIn": 1800}),
        http_client=StubHttpClient(
            {
                "/positions": [],
                "/summary": SUMMARY_D00014,
                "/orders": [],
                "/trades": [],
            }
        ),
        now=datetime(2026, 5, 23, 12, 0, tzinfo=UTC),
    )

    summary = result.to_dict_with_canonical_preview()

    assert summary["canonical_preview"]["total_events"] == 1
    summary_surfaces = {
        item["surface"]: item for item in summary["canonical_preview"]["surfaces"]
    }
    assert summary_surfaces["summary"]["event_count"] == 1


def test_main_map_preview_command_prints_canonical_events(tmp_path: Path, capsys) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "STATARB_PROFILE=local-dev",
                "ALOR_CONTOUR=test",
                "ALOR_TEST_REFRESH_TOKEN=test-refresh",
                "ALOR_TEST_PORTFOLIO=D00014",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        ["map-preview", "--env-file", str(env_file)],
        auth_client=StubAuthClient({"AccessToken": "access-token", "ExpiresIn": 1800}),
        http_client=StubHttpClient(
            {
                "/positions": [],
                "/summary": SUMMARY_D00014,
                "/orders": [],
                "/trades": [],
            }
        ),
    )

    assert exit_code == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["portfolio"] == "D00014"
    assert preview["total_events"] == 1
