from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from statarb.adapters.alor.mapper import (
    AlorMappedEvent,
    AlorMappingContext,
    CanonicalAlorMapper,
    MappingStatus,
)
from statarb.domain.events import ExecutionContour
from statarb.domain.orders import FillEvent, OrderEvent
from statarb.domain.positions import AccountSummary, PositionSnapshot

_PORTFOLIO_SCOPED_SURFACES = frozenset({"positions", "summary", "orders", "trades"})


class HttpSmokeSurface(Protocol):
    surface: str
    status_code: int
    payload: Any


@dataclass(frozen=True, slots=True)
class MappedSurfacePreview:
    surface: str
    status: str
    event_count: int
    events: tuple[dict[str, object], ...]
    unmapped_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "status": self.status,
            "event_count": self.event_count,
            "events": list(self.events),
            "unmapped_reason": self.unmapped_reason,
        }


@dataclass(frozen=True, slots=True)
class CanonicalSmokePreview:
    smoke_scope: str
    portfolio: str | None
    exchange: str
    captured_at: str
    surfaces: tuple[MappedSurfacePreview, ...]
    total_events: int

    def to_dict(self) -> dict[str, object]:
        return {
            "smoke_scope": self.smoke_scope,
            "portfolio": self.portfolio,
            "exchange": self.exchange,
            "captured_at": self.captured_at,
            "surfaces": [surface.to_dict() for surface in self.surfaces],
            "total_events": self.total_events,
        }


def build_canonical_preview(
    *,
    http_results: Sequence[HttpSmokeSurface],
    portfolio: str | None,
    exchange: str,
    smoke_scope: str,
    captured_at: datetime | None = None,
) -> CanonicalSmokePreview:
    resolved_at = _coerce_utc_datetime(captured_at or datetime.now(UTC))
    if smoke_scope != "portfolio_scoped" or portfolio is None:
        return CanonicalSmokePreview(
            smoke_scope=smoke_scope,
            portfolio=portfolio,
            exchange=exchange,
            captured_at=resolved_at.isoformat(),
            surfaces=(),
            total_events=0,
        )

    mapper = CanonicalAlorMapper()
    context = AlorMappingContext(
        portfolio=portfolio,
        exchange=exchange,
        contour=ExecutionContour.BROKER_TEST,
        captured_at=resolved_at,
    )
    surface_previews: list[MappedSurfacePreview] = []
    total_events = 0
    for result in http_results:
        if result.surface not in _PORTFOLIO_SCOPED_SURFACES:
            continue
        if result.status_code != 200:
            surface_previews.append(
                MappedSurfacePreview(
                    surface=result.surface,
                    status=MappingStatus.DEFERRED.value,
                    event_count=0,
                    events=(),
                    unmapped_reason=f"HTTP status {result.status_code} is not mappable.",
                )
            )
            continue
        mapping = mapper.map_http_surface(
            surface=result.surface,
            payload=result.payload,
            context=context,
        )
        serialized = tuple(_serialize_mapped_event(event) for event in mapping.events)
        surface_previews.append(
            MappedSurfacePreview(
                surface=result.surface,
                status=mapping.status.value,
                event_count=len(serialized),
                events=serialized,
                unmapped_reason=mapping.unmapped_reason,
            )
        )
        total_events += len(serialized)

    return CanonicalSmokePreview(
        smoke_scope=smoke_scope,
        portfolio=portfolio,
        exchange=exchange,
        captured_at=resolved_at.isoformat(),
        surfaces=tuple(surface_previews),
        total_events=total_events,
    )


def _serialize_mapped_event(event: AlorMappedEvent) -> dict[str, object]:
    if isinstance(event, OrderEvent):
        return {"event_type": "order_event", "payload": event.to_dict()}
    if isinstance(event, FillEvent):
        return {"event_type": "fill_event", "payload": event.to_dict()}
    if isinstance(event, PositionSnapshot):
        return {"event_type": "position_snapshot", "payload": event.to_dict()}
    if isinstance(event, AccountSummary):
        return {"event_type": "account_summary", "payload": event.to_dict()}
    raise TypeError(f"Unsupported mapped event type: {type(event)!r}.")


def _coerce_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "CanonicalSmokePreview",
    "HttpSmokeSurface",
    "MappedSurfacePreview",
    "build_canonical_preview",
]
