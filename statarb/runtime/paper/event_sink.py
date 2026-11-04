"""Checksummed, fsynced transaction journal with legacy event compatibility."""

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from statarb.domain.orders import FillEvent, OrderEvent
from statarb.domain.positions import AccountSummary, PositionSnapshot, PriceMark

RuntimeDomainEvent = (
    OrderEvent | FillEvent | PositionSnapshot | AccountSummary | PriceMark
)


@dataclass(frozen=True, slots=True)
class EventJournalEntry:
    event_type: str
    payload: dict[str, Any]
    lineage: dict[str, str]

    def to_dict(self):
        contour = self.payload["contour"]
        source = (
            "adapter"
            if contour.startswith("broker-")
            else "replay"
            if contour == "replay"
            else "execution"
        )
        return {
            "event_type": self.event_type,
            "payload": self.payload,
            "lineage": self.lineage,
            "event_id": self.payload.get(
                "event_id",
                self.payload.get("snapshot_id", self.payload.get("summary_id")),
            ),
            "occurred_at": self.payload.get(
                "occurred_at", self.payload.get("captured_at")
            ),
            "source": source,
            "contour": contour,
        }


def build_event_journal_entry(
    event: RuntimeDomainEvent, *, lineage: Mapping[str, str | None] | None = None
):
    keys = ("order_id", "fill_id", "target_position_id", "signal_intent_id")
    combined = {
        key: getattr(event, key)
        for key in keys
        if getattr(event, key, None) is not None
    }
    combined.update(
        {key: value for key, value in (lineage or {}).items() if value is not None}
    )
    return EventJournalEntry(_event_type_name(event), event.to_dict(), combined)


def transaction_checksum(transaction_id: str, entries: list[dict]) -> str:
    body = json.dumps(
        {"transaction_id": transaction_id, "events": entries},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(body.encode()).hexdigest()


class JsonlEventSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._expected_size = self.path.stat().st_size if self.path.exists() else 0

    def append(self, event: RuntimeDomainEvent, *, lineage=None):
        entry = build_event_journal_entry(event, lineage=lineage)
        self.append_batch((entry,), transaction_id=f"event-{uuid4().hex}")
        return entry

    def append_batch(
        self, entries: Sequence[EventJournalEntry], *, transaction_id: str
    ):
        if not transaction_id or not entries:
            raise ValueError(
                "A journal transaction requires an id and at least one event."
            )
        payloads = [entry.to_dict() for entry in entries]
        checksum = transaction_checksum(transaction_id, payloads)
        line = (
            json.dumps(
                {
                    "schema_version": 2,
                    "transaction_id": transaction_id,
                    "events": payloads,
                    "checksum": checksum,
                },
                sort_keys=True,
            )
            + "\n"
        ).encode()
        # A single cooperating writer owns a journal. A process restart must
        # reconstruct its ledger before appending (PaperExecutionEngine.recover).
        with self.path.open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0, os.SEEK_END)
            offset = handle.tell()
            if offset != self._expected_size:
                raise OSError(
                    "Journal changed under this writer; recover before appending."
                )
            if offset:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    raise OSError(
                        "Uncommitted journal tail; recover to a new journal before appending."
                    )
            try:
                handle.seek(0, os.SEEK_END)
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
                self._expected_size = offset + len(line)
            except OSError:
                handle.truncate(offset)
                handle.flush()
                os.fsync(handle.fileno())
                raise
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _event_type_name(event):
    for cls, name in (
        (OrderEvent, "order_event"),
        (FillEvent, "fill_event"),
        (PositionSnapshot, "position_snapshot"),
        (AccountSummary, "account_summary"),
        (PriceMark, "price_mark"),
    ):
        if isinstance(event, cls):
            return name
    raise TypeError(f"Unsupported runtime event type: {type(event)!r}")


__all__ = [
    "EventJournalEntry",
    "JsonlEventSink",
    "RuntimeDomainEvent",
    "build_event_journal_entry",
]
