from __future__ import annotations

"""Simple JSONL event persistence for paper and replay flows."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from statarb.domain.orders import FillEvent, OrderEvent
from statarb.domain.positions import AccountSummary, PositionSnapshot

RuntimeDomainEvent = OrderEvent | FillEvent | PositionSnapshot | AccountSummary


@dataclass(frozen=True, slots=True)
class EventJournalEntry:
    event_type: str
    payload: dict[str, Any]
    lineage: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "payload": self.payload,
            "lineage": self.lineage,
        }


def build_event_journal_entry(
    event: RuntimeDomainEvent,
    *,
    lineage: Mapping[str, str | None] | None = None,
) -> EventJournalEntry:
    entry_lineage = {
        key: value
        for key, value in _default_lineage(event).items()
        if value is not None
    }
    if lineage is not None:
        entry_lineage.update(
            {
                key: value
                for key, value in lineage.items()
                if value is not None
            }
        )

    return EventJournalEntry(
        event_type=_event_type_name(event),
        payload=event.to_dict(),
        lineage=entry_lineage,
    )


class JsonlEventSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        event: RuntimeDomainEvent,
        *,
        lineage: Mapping[str, str | None] | None = None,
    ) -> EventJournalEntry:
        entry = build_event_journal_entry(event, lineage=lineage)
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(entry.to_dict(), handle, sort_keys=True)
            handle.write("\n")
        return entry


def _event_type_name(event: RuntimeDomainEvent) -> str:
    if isinstance(event, OrderEvent):
        return "order_event"
    if isinstance(event, FillEvent):
        return "fill_event"
    if isinstance(event, PositionSnapshot):
        return "position_snapshot"
    if isinstance(event, AccountSummary):
        return "account_summary"
    raise TypeError(f"Unsupported runtime event type: {type(event)!r}.")


def _default_lineage(event: RuntimeDomainEvent) -> dict[str, str | None]:
    if isinstance(event, OrderEvent):
        return {
            "order_id": event.order_id,
            "target_position_id": event.target_position_id,
            "signal_intent_id": event.signal_intent_id,
        }
    if isinstance(event, FillEvent):
        return {
            "order_id": event.order_id,
            "fill_id": event.fill_id,
        }
    return {}


__all__ = ["EventJournalEntry", "JsonlEventSink", "RuntimeDomainEvent", "build_event_journal_entry"]
