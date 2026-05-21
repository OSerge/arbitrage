from __future__ import annotations

"""Load runtime event journals from JSONL."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from statarb.domain.instruments import InstrumentRef
from statarb.domain.orders import FillEvent, OrderEvent
from statarb.domain.positions import AccountSummary, PositionSnapshot

RuntimeDomainEvent = OrderEvent | FillEvent | PositionSnapshot | AccountSummary


@dataclass(frozen=True, slots=True)
class EventJournalRecord:
    event_type: str
    payload: dict[str, Any]
    lineage: dict[str, str]


def load_event_journal(path: str | Path) -> tuple[EventJournalRecord, ...]:
    journal_path = Path(path)
    if not journal_path.exists():
        return ()

    records: list[EventJournalRecord] = []
    with journal_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            records.append(
                EventJournalRecord(
                    event_type=item["event_type"],
                    payload=dict(item["payload"]),
                    lineage=dict(item.get("lineage", {})),
                )
            )
    return tuple(records)


def decode_event_record(record: EventJournalRecord) -> RuntimeDomainEvent:
    payload = dict(record.payload)

    if record.event_type == "order_event":
        payload["instrument"] = _instrument_from_payload(payload["instrument"])
        payload["occurred_at"] = _datetime_from_payload(payload["occurred_at"])
        return OrderEvent(**payload)
    if record.event_type == "fill_event":
        payload["instrument"] = _instrument_from_payload(payload["instrument"])
        payload["occurred_at"] = _datetime_from_payload(payload["occurred_at"])
        return FillEvent(**payload)
    if record.event_type == "position_snapshot":
        payload["instrument"] = _instrument_from_payload(payload["instrument"])
        payload["captured_at"] = _datetime_from_payload(payload["captured_at"])
        return PositionSnapshot(**payload)
    if record.event_type == "account_summary":
        payload["captured_at"] = _datetime_from_payload(payload["captured_at"])
        return AccountSummary(**payload)
    raise ValueError(f"Unsupported journal event type: {record.event_type!r}.")


def _instrument_from_payload(payload: dict[str, Any]) -> InstrumentRef:
    return InstrumentRef(
        symbol=payload["symbol"],
        exchange=payload["exchange"],
        instrument_type=payload["instrument_type"],
    )


def _datetime_from_payload(value: str) -> datetime:
    return datetime.fromisoformat(value)


__all__ = ["EventJournalRecord", "RuntimeDomainEvent", "decode_event_record", "load_event_journal"]
