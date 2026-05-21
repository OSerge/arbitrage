from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Protocol, runtime_checkable


class CanonicalEventKind(StrEnum):
    ORDER_EVENT = "order-event"
    FILL_EVENT = "fill-event"
    POSITION_SNAPSHOT = "position-snapshot"
    ACCOUNT_SUMMARY = "account-summary"
    RAW_BROKER_EVENT = "raw-broker-event"


class MappingStatus(StrEnum):
    PLACEHOLDER = "placeholder"
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class CanonicalEventPlaceholder:
    kind: CanonicalEventKind
    source_stream: str
    opcode: str
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class MappingResult:
    status: MappingStatus
    payload: Mapping[str, object]
    events: tuple[CanonicalEventPlaceholder, ...] = ()
    unmapped_reason: str | None = None


@runtime_checkable
class AlorPayloadMapper(Protocol):
    def map_payload(
        self,
        *,
        stream: str,
        opcode: str,
        payload: Mapping[str, object],
    ) -> MappingResult: ...


class PlaceholderAlorMapper:
    _CANONICAL_EVENT_KINDS: dict[str, CanonicalEventKind] = {
        "OrdersGetAndSubscribeV2": CanonicalEventKind.ORDER_EVENT,
        "TradesGetAndSubscribe": CanonicalEventKind.FILL_EVENT,
        "PositionsGetAndSubscribeV2": CanonicalEventKind.POSITION_SNAPSHOT,
        "SummariesGetAndSubscribeV2": CanonicalEventKind.ACCOUNT_SUMMARY,
    }

    def map_payload(
        self,
        *,
        stream: str,
        opcode: str,
        payload: Mapping[str, object],
    ) -> MappingResult:
        event_kind = self._CANONICAL_EVENT_KINDS.get(opcode)
        if event_kind is None:
            return MappingResult(
                status=MappingStatus.DEFERRED,
                payload=payload,
                unmapped_reason="No canonical event contract is frozen for this payload yet.",
            )
        placeholder = CanonicalEventPlaceholder(
            kind=event_kind,
            source_stream=stream,
            opcode=opcode,
            payload=payload,
        )
        return MappingResult(
            status=MappingStatus.PLACEHOLDER,
            payload=payload,
            events=(placeholder,),
        )


__all__ = [
    "AlorPayloadMapper",
    "CanonicalEventKind",
    "CanonicalEventPlaceholder",
    "MappingResult",
    "MappingStatus",
    "PlaceholderAlorMapper",
]
