from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DeploymentProfile(StrEnum):
    LOCAL_DEV = "local-dev"
    SINGLE_HOST_PROD_LIKE = "single-host-prod-like"

    @classmethod
    def coerce(cls, value: "DeploymentProfile | str | None") -> "DeploymentProfile":
        if value is None:
            return cls.LOCAL_DEV
        if isinstance(value, cls):
            return value

        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member

        allowed = ", ".join(profile.value for profile in cls)
        raise ValueError(f"Unsupported deployment profile: {value!r}. Expected one of: {allowed}.")


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    profile: DeploymentProfile
    description: str
    postgres_required: bool
    structured_logs_default: bool
    default_artifacts_dir: str
    example_env_file: str


PROFILE_SPECS: dict[DeploymentProfile, ProfileSpec] = {
    DeploymentProfile.LOCAL_DEV: ProfileSpec(
        profile=DeploymentProfile.LOCAL_DEV,
        description="Single-process local development baseline with optional Postgres wiring.",
        postgres_required=False,
        structured_logs_default=False,
        default_artifacts_dir="var/local-dev",
        example_env_file="ops/env/local-dev.env.example",
    ),
    DeploymentProfile.SINGLE_HOST_PROD_LIKE: ProfileSpec(
        profile=DeploymentProfile.SINGLE_HOST_PROD_LIKE,
        description="Single-host multi-process baseline with explicit Postgres dependency.",
        postgres_required=True,
        structured_logs_default=True,
        default_artifacts_dir="var/single-host-prod-like",
        example_env_file="ops/env/single-host-prod-like.env.example",
    ),
}


def get_profile_spec(profile: DeploymentProfile | str) -> ProfileSpec:
    return PROFILE_SPECS[DeploymentProfile.coerce(profile)]


__all__ = [
    "DeploymentProfile",
    "ProfileSpec",
    "PROFILE_SPECS",
    "get_profile_spec",
]
