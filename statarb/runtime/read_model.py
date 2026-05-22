from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from statarb.domain.orders import FillEvent, OrderEvent, OrderStatus
from statarb.domain.positions import AccountSummary, PositionSnapshot
from statarb.runtime.replay.loader import decode_event_record, load_event_journal

_TERMINAL_ORDER_STATUSES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELED,
    OrderStatus.REJECTED,
}


@dataclass(frozen=True, slots=True)
class EventTypeCounts:
    order_event: int = 0
    fill_event: int = 0
    position_snapshot: int = 0
    account_summary: int = 0

    @property
    def total(self) -> int:
        return self.order_event + self.fill_event + self.position_snapshot + self.account_summary

    def to_dict(self) -> dict[str, int]:
        return {
            "order_event": self.order_event,
            "fill_event": self.fill_event,
            "position_snapshot": self.position_snapshot,
            "account_summary": self.account_summary,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class OpsReadModel:
    journal_path: str
    total_events: int
    event_counts: EventTypeCounts
    source_contours: tuple[str, ...]
    position_symbols: tuple[str, ...]
    open_order_ids: tuple[str, ...]
    latest_account_summary: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "journal_path": self.journal_path,
            "total_events": self.total_events,
            "event_counts": self.event_counts.to_dict(),
            "source_contours": list(self.source_contours),
            "position_symbols": list(self.position_symbols),
            "open_order_ids": list(self.open_order_ids),
            "latest_account_summary": self.latest_account_summary,
        }


def build_ops_read_model(journal_path: str | Path) -> OpsReadModel:
    resolved_path = Path(journal_path)
    records = load_event_journal(resolved_path)
    counts = EventTypeCounts()
    contours: set[str] = set()
    latest_positions: dict[str, PositionSnapshot] = {}
    latest_orders: dict[str, OrderEvent] = {}
    latest_account_summary: AccountSummary | None = None

    for record in records:
        event = decode_event_record(record)
        if isinstance(event, OrderEvent):
            counts = EventTypeCounts(
                order_event=counts.order_event + 1,
                fill_event=counts.fill_event,
                position_snapshot=counts.position_snapshot,
                account_summary=counts.account_summary,
            )
            contours.add(event.contour.value)
            latest_orders[event.order_id] = event
            continue
        if isinstance(event, FillEvent):
            counts = EventTypeCounts(
                order_event=counts.order_event,
                fill_event=counts.fill_event + 1,
                position_snapshot=counts.position_snapshot,
                account_summary=counts.account_summary,
            )
            contours.add(event.contour.value)
            continue
        if isinstance(event, PositionSnapshot):
            counts = EventTypeCounts(
                order_event=counts.order_event,
                fill_event=counts.fill_event,
                position_snapshot=counts.position_snapshot + 1,
                account_summary=counts.account_summary,
            )
            contours.add(event.contour.value)
            latest_positions[event.instrument.symbol] = event
            continue
        if isinstance(event, AccountSummary):
            counts = EventTypeCounts(
                order_event=counts.order_event,
                fill_event=counts.fill_event,
                position_snapshot=counts.position_snapshot,
                account_summary=counts.account_summary + 1,
            )
            contours.add(event.contour.value)
            latest_account_summary = event

    open_order_ids = tuple(
        sorted(
            order_id
            for order_id, order_event in latest_orders.items()
            if order_event.status not in _TERMINAL_ORDER_STATUSES
        )
    )
    return OpsReadModel(
        journal_path=str(resolved_path),
        total_events=counts.total,
        event_counts=counts,
        source_contours=tuple(sorted(contours)),
        position_symbols=tuple(sorted(latest_positions)),
        open_order_ids=open_order_ids,
        latest_account_summary=(
            latest_account_summary.to_dict() if latest_account_summary is not None else None
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "inspect":
        summary = build_ops_read_model(args.journal_path).to_dict()
        _print_summary(summary)
        return 0
    parser.error(f"Unsupported command: {args.command!r}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m statarb.runtime.read_model",
        description="Build file-backed ops read-models from canonical JSONL event journals.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser(
        "inspect",
        help="Print an aggregated ops read-model for a canonical event journal.",
    )
    inspect.add_argument("--journal-path", required=True)
    return parser


def _print_summary(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
