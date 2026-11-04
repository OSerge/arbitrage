"""Read legacy events and committed v2 transaction records."""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any
import warnings

from statarb.domain.instruments import InstrumentRef
from statarb.domain.orders import FillEvent, OrderEvent
from statarb.domain.positions import AccountSummary, PositionSnapshot, PriceMark
from statarb.runtime.paper.event_sink import RuntimeDomainEvent, transaction_checksum


class UncommittedJournalTailWarning(UserWarning):
    pass


@dataclass(frozen=True, slots=True)
class EventJournalRecord:
    event_type: str
    payload: dict[str, Any]
    lineage: dict[str, str]
    transaction_id: str | None = None


def load_event_journal(path: str | Path) -> tuple[EventJournalRecord, ...]:
    path = Path(path)
    if not path.exists():
        return ()
    lines = path.read_bytes().splitlines(keepends=True)
    records = []
    transactions = {}
    for number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        if not raw_line.endswith(b"\n"):
            warnings.warn(
                f"Ignoring uncommitted final journal tail at line {number}.",
                UncommittedJournalTailWarning,
                stacklevel=2,
            )
            break
        try:
            item = json.loads(raw_line)
            transaction_id = None
            if "schema_version" in item:
                if item["schema_version"] != 2:
                    raise ValueError("Unsupported journal schema_version.")
                transaction_id = item["transaction_id"]
                entries = item["events"]
                if not transaction_id or not isinstance(entries, list) or not entries:
                    raise ValueError("Invalid transaction shape.")
                checksum = transaction_checksum(transaction_id, entries)
                if checksum != item["checksum"]:
                    raise ValueError("Journal transaction checksum mismatch.")
                if transaction_id in transactions:
                    if transactions[transaction_id] != checksum:
                        raise ValueError("Conflicting duplicate transaction id.")
                    continue
                transactions[transaction_id] = checksum
            else:
                entries = [item]
            for entry in entries:
                if "event_id" in entry:
                    payload = entry["payload"]
                    expected_id = payload.get(
                        "event_id",
                        payload.get("snapshot_id", payload.get("summary_id")),
                    )
                    expected_time = payload.get(
                        "occurred_at", payload.get("captured_at")
                    )
                    if (
                        entry["event_id"] != expected_id
                        or entry.get("occurred_at") != expected_time
                        or entry.get("contour") != payload["contour"]
                    ):
                        raise ValueError("Event envelope disagrees with its payload.")
                    if entry.get("source") not in {
                        "execution",
                        "adapter",
                        "replay",
                        "research",
                        "portfolio",
                        "ops",
                    }:
                        raise ValueError("Invalid event source.")
                record = EventJournalRecord(
                    entry["event_type"],
                    dict(entry["payload"]),
                    dict(entry.get("lineage", {})),
                    transaction_id,
                )
                decode_event_record(record)
                records.append(record)
        except (ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"Invalid committed journal record at line {number}: {exc}"
            ) from exc
    return tuple(records)


def decode_event_record(record: EventJournalRecord) -> RuntimeDomainEvent:
    payload = dict(record.payload)
    classes = {
        "order_event": OrderEvent,
        "fill_event": FillEvent,
        "position_snapshot": PositionSnapshot,
        "account_summary": AccountSummary,
        "price_mark": PriceMark,
    }
    cls = classes.get(record.event_type)
    if cls is None:
        raise ValueError(f"Unsupported journal event type: {record.event_type!r}")
    if "instrument" in payload:
        payload["instrument"] = InstrumentRef(**payload["instrument"])
    field = (
        "captured_at" if cls in (PositionSnapshot, AccountSummary) else "occurred_at"
    )
    payload[field] = datetime.fromisoformat(payload[field])
    return cls(**payload)


__all__ = [
    "EventJournalRecord",
    "RuntimeDomainEvent",
    "decode_event_record",
    "load_event_journal",
]
