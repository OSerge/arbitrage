from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from os import PathLike
from pathlib import Path
from typing import Mapping

from statarb.config.profiles import DeploymentProfile, ProfileSpec, get_profile_spec


class AlorContour(StrEnum):
    TEST = "test"
    LIVE = "live"

    @classmethod
    def coerce(cls, value: "AlorContour | str | None") -> "AlorContour":
        if value is None:
            return cls.TEST
        if isinstance(value, cls):
            return value

        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member

        allowed = ", ".join(contour.value for contour in cls)
        raise ValueError(f"Unsupported Alor contour: {value!r}. Expected one of: {allowed}.")


@dataclass(frozen=True, slots=True)
class PostgresSettings:
    dsn: str | None
    required: bool

    @property
    def is_configured(self) -> bool:
        return bool(self.dsn)


@dataclass(frozen=True, slots=True)
class AlorSettings:
    contour: AlorContour
    refresh_token: str | None
    portfolio: str | None

    @property
    def is_configured(self) -> bool:
        return bool(self.refresh_token and self.portfolio)


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    profile: DeploymentProfile
    profile_spec: ProfileSpec
    env_file: Path | None
    artifacts_dir: Path
    event_log_dir: Path
    log_level: str
    structured_logs: bool
    live_trading_enabled: bool
    postgres: PostgresSettings
    alor: AlorSettings


def load_settings(
    *,
    profile: DeploymentProfile | str | None = None,
    env_file: str | PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> RuntimeSettings:
    raw_environ = dict(os.environ if environ is None else environ)
    resolved_env_file = _resolve_env_file(env_file or raw_environ.get("STATARB_ENV_FILE"))
    file_values = _read_env_file(resolved_env_file) if resolved_env_file else {}
    effective_env = {**file_values, **raw_environ}

    resolved_profile = DeploymentProfile.coerce(profile or effective_env.get("STATARB_PROFILE"))
    profile_spec = get_profile_spec(resolved_profile)

    artifacts_dir = _resolve_path(
        effective_env.get("STATARB_ARTIFACTS_DIR"),
        default=profile_spec.default_artifacts_dir,
    )
    event_log_dir = _resolve_path(
        effective_env.get("STATARB_EVENT_LOG_DIR"),
        default=str(artifacts_dir / "events"),
    )
    log_level = effective_env.get("STATARB_LOG_LEVEL", "INFO").strip().upper()
    structured_logs = _bool_from_env(
        effective_env.get("STATARB_STRUCTURED_LOGS"),
        default=profile_spec.structured_logs_default,
    )
    live_trading_enabled = _bool_from_env(
        effective_env.get("STATARB_ENABLE_LIVE_TRADING"),
        default=False,
    )

    postgres_dsn = _optional_string(effective_env.get("DATABASE_URL"))
    if profile_spec.postgres_required and not postgres_dsn:
        raise ValueError(
            f"Profile {resolved_profile.value!r} requires DATABASE_URL to be configured."
        )

    contour = AlorContour.coerce(effective_env.get("ALOR_CONTOUR"))
    if contour is AlorContour.LIVE and not live_trading_enabled:
        raise ValueError(
            "ALOR_CONTOUR=live requires STATARB_ENABLE_LIVE_TRADING=true."
        )

    contour_prefix = "ALOR_TEST" if contour is AlorContour.TEST else "ALOR_LIVE"
    alor = AlorSettings(
        contour=contour,
        refresh_token=_optional_string(effective_env.get(f"{contour_prefix}_REFRESH_TOKEN")),
        portfolio=_optional_string(effective_env.get(f"{contour_prefix}_PORTFOLIO")),
    )

    return RuntimeSettings(
        profile=resolved_profile,
        profile_spec=profile_spec,
        env_file=resolved_env_file,
        artifacts_dir=artifacts_dir,
        event_log_dir=event_log_dir,
        log_level=log_level,
        structured_logs=structured_logs,
        live_trading_enabled=live_trading_enabled,
        postgres=PostgresSettings(dsn=postgres_dsn, required=profile_spec.postgres_required),
        alor=alor,
    )


def _resolve_env_file(env_file: str | PathLike[str] | None) -> Path | None:
    if env_file is None:
        default_env_file = _project_root() / ".env"
        return default_env_file if default_env_file.exists() else None

    path = Path(env_file)
    if not path.is_absolute():
        path = (_project_root() / path).resolve()
    return path


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        values[key] = value
    return values


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[7:].lstrip()

    key, separator, value = stripped.partition("=")
    if separator != "=":
        raise ValueError(f"Invalid .env line: {line!r}")

    key = key.strip()
    if not key:
        raise ValueError(f"Invalid .env line: {line!r}")

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]

    return key, value


def _resolve_path(value: str | None, *, default: str) -> Path:
    candidate = Path(value or default)
    if not candidate.is_absolute():
        candidate = (_project_root() / candidate).resolve()
    return candidate


def _bool_from_env(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"Unsupported boolean value: {value!r}")


def _optional_string(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


__all__ = [
    "AlorContour",
    "AlorSettings",
    "PostgresSettings",
    "RuntimeSettings",
    "load_settings",
]
