from pathlib import Path

import pytest

from statarb import DeploymentProfile, load_settings
from statarb.config import AlorContour, get_profile_spec


def test_statarb_shell_is_importable() -> None:
    assert DeploymentProfile.LOCAL_DEV.value == "local-dev"
    assert callable(load_settings)


def test_load_settings_reads_env_file(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "STATARB_PROFILE=local-dev",
                "STATARB_LOG_LEVEL=debug",
                f"STATARB_ARTIFACTS_DIR={artifacts_dir}",
                "ALOR_CONTOUR=test",
                "ALOR_TEST_REFRESH_TOKEN=test-refresh",
                "ALOR_TEST_PORTFOLIO=D39004",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env_file=env_file, environ={})

    assert settings.profile is DeploymentProfile.LOCAL_DEV
    assert settings.log_level == "DEBUG"
    assert settings.artifacts_dir == artifacts_dir
    assert settings.event_log_dir == artifacts_dir / "events"
    assert settings.alor.contour is AlorContour.TEST
    assert settings.alor.refresh_token == "test-refresh"
    assert settings.alor.portfolio == "D39004"


def test_environment_variables_override_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "ALOR_CONTOUR=test",
                "ALOR_TEST_REFRESH_TOKEN=file-refresh",
                "ALOR_TEST_PORTFOLIO=file-portfolio",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(
        env_file=env_file,
        environ={
            "STATARB_PROFILE": "local-dev",
            "ALOR_TEST_PORTFOLIO": "env-portfolio",
        },
    )

    assert settings.alor.refresh_token == "file-refresh"
    assert settings.alor.portfolio == "env-portfolio"


def test_single_host_prod_like_requires_database_url(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="DATABASE_URL"):
        load_settings(
            profile="single-host-prod-like",
            env_file=tmp_path / "missing.env",
            environ={},
        )


def test_live_contour_requires_explicit_enablement(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="STATARB_ENABLE_LIVE_TRADING"):
        load_settings(
            env_file=tmp_path / "missing.env",
            environ={
                "ALOR_CONTOUR": "live",
                "ALOR_LIVE_REFRESH_TOKEN": "refresh-token",
                "ALOR_LIVE_PORTFOLIO": "portfolio",
            }
        )


def test_profile_specs_capture_operational_defaults() -> None:
    local_dev = get_profile_spec("local-dev")
    prod_like = get_profile_spec("single-host-prod-like")

    assert local_dev.postgres_required is False
    assert prod_like.postgres_required is True
    assert prod_like.structured_logs_default is True
