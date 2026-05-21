"""Minimal platform shell for the agent-operated MVP."""

from statarb.config import (
    AlorContour,
    AlorSettings,
    DeploymentProfile,
    PostgresSettings,
    RuntimeSettings,
    load_settings,
)

__all__ = [
    "AlorContour",
    "AlorSettings",
    "DeploymentProfile",
    "PostgresSettings",
    "RuntimeSettings",
    "load_settings",
]
