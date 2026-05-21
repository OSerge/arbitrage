from __future__ import annotations

"""Canonical event contracts and shared serialization helpers."""

from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Mapping


def _require_non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _require_aware_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value


def _coerce_decimal(value: Decimal | int | float | str, field_name: str) -> Decimal:
    if isinstance(value, Decimal):
        return value

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field_name} must be a decimal-compatible value.") from exc


def _serialize_value(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _serialize_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_serialize_value(item) for item in value]
    return value


class SerializableContract:
    def to_dict(self) -> dict[str, Any]:
        if not is_dataclass(self):
            raise TypeError("SerializableContract.to_dict() requires a dataclass instance.")
        return {
            field.name: _serialize_value(getattr(self, field.name))
            for field in fields(self)
        }


class ExecutionContour(StrEnum):
    PAPER = "paper"
    BROKER_TEST = "broker-test"
    BROKER_LIVE = "broker-live"
    REPLAY = "replay"

    @classmethod
    def coerce(cls, value: "ExecutionContour | str") -> "ExecutionContour":
        if isinstance(value, cls):
            return value

        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member

        allowed = ", ".join(member.value for member in cls)
        raise ValueError(f"Unsupported execution contour: {value!r}. Expected one of: {allowed}.")


class EventSource(StrEnum):
    RESEARCH = "research"
    PORTFOLIO = "portfolio"
    EXECUTION = "execution"
    ADAPTER = "adapter"
    REPLAY = "replay"
    OPS = "ops"


class ReplaySessionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @classmethod
    def coerce(cls, value: "ReplaySessionStatus | str") -> "ReplaySessionStatus":
        if isinstance(value, cls):
            return value

        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member

        allowed = ", ".join(member.value for member in cls)
        raise ValueError(f"Unsupported replay session status: {value!r}. Expected one of: {allowed}.")


@dataclass(frozen=True, slots=True)
class ReplaySession(SerializableContract):
    session_id: str
    started_at: datetime
    status: ReplaySessionStatus
    source_contour: ExecutionContour
    source_event_count: int
    ended_at: datetime | None = None
    replayed_event_count: int = 0
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _require_non_empty(self.session_id, "session_id"))
        object.__setattr__(self, "started_at", _require_aware_datetime(self.started_at, "started_at"))
        object.__setattr__(self, "status", ReplaySessionStatus.coerce(self.status))
        object.__setattr__(self, "source_contour", ExecutionContour.coerce(self.source_contour))

        if self.source_event_count < 0:
            raise ValueError("source_event_count must be greater than or equal to 0.")
        if self.replayed_event_count < 0:
            raise ValueError("replayed_event_count must be greater than or equal to 0.")

        if self.ended_at is not None:
            object.__setattr__(self, "ended_at", _require_aware_datetime(self.ended_at, "ended_at"))
            if self.ended_at < self.started_at:
                raise ValueError("ended_at must be greater than or equal to started_at.")

        if self.notes is not None:
            notes = self.notes.strip()
            object.__setattr__(self, "notes", notes or None)


__all__ = [
    "EventSource",
    "ExecutionContour",
    "ReplaySession",
    "ReplaySessionStatus",
    "SerializableContract",
    "_coerce_decimal",
    "_require_aware_datetime",
    "_require_non_empty",
]
