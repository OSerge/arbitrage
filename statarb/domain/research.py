from __future__ import annotations

"""Canonical research contracts."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping, TypeAlias

from statarb.domain.events import (
    SerializableContract,
    _coerce_decimal,
    _require_aware_datetime,
    _require_non_empty,
)
from statarb.domain.instruments import InstrumentRef

ResearchParameterValue: TypeAlias = str | int | float | bool


class ResearchRunStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @classmethod
    def coerce(cls, value: "ResearchRunStatus | str") -> "ResearchRunStatus":
        if isinstance(value, cls):
            return value

        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member

        allowed = ", ".join(member.value for member in cls)
        raise ValueError(f"Unsupported research run status: {value!r}. Expected one of: {allowed}.")


class SignalDirection(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

    @classmethod
    def coerce(cls, value: "SignalDirection | str") -> "SignalDirection":
        if isinstance(value, cls):
            return value

        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member

        allowed = ", ".join(member.value for member in cls)
        raise ValueError(f"Unsupported signal direction: {value!r}. Expected one of: {allowed}.")


@dataclass(frozen=True, slots=True)
class ResearchRun(SerializableContract):
    run_id: str
    strategy_id: str
    dataset_id: str
    status: ResearchRunStatus
    started_at: datetime
    completed_at: datetime | None = None
    parameters: Mapping[str, ResearchParameterValue] = field(default_factory=dict)
    artifact_uris: tuple[str, ...] = ()
    signal_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_non_empty(self.run_id, "run_id"))
        object.__setattr__(self, "strategy_id", _require_non_empty(self.strategy_id, "strategy_id"))
        object.__setattr__(self, "dataset_id", _require_non_empty(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "status", ResearchRunStatus.coerce(self.status))
        object.__setattr__(self, "started_at", _require_aware_datetime(self.started_at, "started_at"))

        if self.completed_at is not None:
            object.__setattr__(
                self,
                "completed_at",
                _require_aware_datetime(self.completed_at, "completed_at"),
            )
            if self.completed_at < self.started_at:
                raise ValueError("completed_at must be greater than or equal to started_at.")

        normalized_parameters = dict(self.parameters)
        object.__setattr__(self, "parameters", MappingProxyType(normalized_parameters))

        normalized_artifact_uris = tuple(
            _require_non_empty(uri, "artifact_uris")
            for uri in self.artifact_uris
        )
        object.__setattr__(self, "artifact_uris", normalized_artifact_uris)

        if self.signal_count < 0:
            raise ValueError("signal_count must be greater than or equal to 0.")


@dataclass(frozen=True, slots=True)
class SignalIntent(SerializableContract):
    intent_id: str
    research_run_id: str
    instrument: InstrumentRef
    direction: SignalDirection
    conviction: Decimal
    generated_at: datetime
    horizon: str | None = None
    thesis: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent_id", _require_non_empty(self.intent_id, "intent_id"))
        object.__setattr__(
            self,
            "research_run_id",
            _require_non_empty(self.research_run_id, "research_run_id"),
        )
        if not isinstance(self.instrument, InstrumentRef):
            raise ValueError("instrument must be an InstrumentRef instance.")

        object.__setattr__(self, "direction", SignalDirection.coerce(self.direction))
        conviction = _coerce_decimal(self.conviction, "conviction")
        if conviction < Decimal("0") or conviction > Decimal("1"):
            raise ValueError("conviction must be between 0 and 1.")
        object.__setattr__(self, "conviction", conviction)
        object.__setattr__(
            self,
            "generated_at",
            _require_aware_datetime(self.generated_at, "generated_at"),
        )

        if self.horizon is not None:
            horizon = self.horizon.strip()
            object.__setattr__(self, "horizon", horizon or None)

        if self.thesis is not None:
            thesis = self.thesis.strip()
            object.__setattr__(self, "thesis", thesis or None)


__all__ = [
    "ResearchParameterValue",
    "ResearchRun",
    "ResearchRunStatus",
    "SignalDirection",
    "SignalIntent",
]
