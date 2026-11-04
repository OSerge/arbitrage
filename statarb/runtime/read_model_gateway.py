from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from statarb.domain.orders import OrderEvent
from statarb.domain.positions import AccountSummary, PositionSnapshot
from statarb.runtime.read_model import OpsReadModel, build_ops_read_model
from statarb.runtime.replay.loader import decode_event_record, load_event_journal

READ_MODEL_SCHEMA_VERSION = "1"


@dataclass(frozen=True, slots=True)
class ResearchReadModel:
    journal_path: str
    source_contours: tuple[str, ...]
    signal_intent_ids: tuple[str, ...]
    target_position_ids: tuple[str, ...]
    research_run_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "journal_path": self.journal_path,
            "source_contours": list(self.source_contours),
            "signal_intent_ids": list(self.signal_intent_ids),
            "target_position_ids": list(self.target_position_ids),
            "research_run_ids": list(self.research_run_ids),
        }


@dataclass(frozen=True, slots=True)
class RiskReadModel:
    journal_path: str
    open_order_ids: tuple[str, ...]
    position_symbols: tuple[str, ...]
    net_exposure_by_symbol: dict[str, str]
    latest_equity: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "journal_path": self.journal_path,
            "open_order_ids": list(self.open_order_ids),
            "position_symbols": list(self.position_symbols),
            "net_exposure_by_symbol": dict(self.net_exposure_by_symbol),
            "latest_equity": self.latest_equity,
        }


@dataclass(frozen=True, slots=True)
class MvpReadModelBundle:
    schema_version: str
    generated_at: datetime
    ops: OpsReadModel
    research: ResearchReadModel
    risk: RiskReadModel

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at.isoformat(),
            "ops": self.ops.to_dict(),
            "research": self.research.to_dict(),
            "risk": self.risk.to_dict(),
        }


def build_mvp_read_model_bundle(
    journal_path: str | Path,
    *,
    generated_at: datetime | None = None,
) -> MvpReadModelBundle:
    resolved_path = Path(journal_path)
    ops = build_ops_read_model(resolved_path)
    research = _build_research_read_model(resolved_path)
    risk = _build_risk_read_model(resolved_path, ops=ops)
    return MvpReadModelBundle(
        schema_version=READ_MODEL_SCHEMA_VERSION,
        generated_at=_coerce_utc_datetime(generated_at or datetime.now(UTC)),
        ops=ops,
        research=research,
        risk=risk,
    )


def export_mvp_read_model_bundle(
    journal_path: str | Path,
    output_dir: str | Path,
    *,
    generated_at: datetime | None = None,
) -> Path:
    bundle = build_mvp_read_model_bundle(journal_path, generated_at=generated_at)
    resolved_dir = Path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolved_dir / "manifest.json"
    _write_json(manifest_path, bundle.to_dict())
    _write_json(resolved_dir / "ops.json", bundle.ops.to_dict())
    _write_json(resolved_dir / "research.json", bundle.research.to_dict())
    _write_json(resolved_dir / "risk.json", bundle.risk.to_dict())
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "export":
        manifest_path = export_mvp_read_model_bundle(
            args.journal_path,
            args.output_dir,
        )
        _print_summary(
            {
                "schema_version": READ_MODEL_SCHEMA_VERSION,
                "manifest_path": str(manifest_path),
                "output_dir": str(Path(args.output_dir)),
            }
        )
        return 0
    parser.error(f"Unsupported command: {args.command!r}")
    return 0


def _build_research_read_model(journal_path: Path) -> ResearchReadModel:
    records = load_event_journal(journal_path)
    contours: set[str] = set()
    signal_intent_ids: set[str] = set()
    target_position_ids: set[str] = set()
    research_run_ids: set[str] = set()
    for record in records:
        event = decode_event_record(record)
        contours.add(event.contour.value)
        lineage = record.lineage
        if lineage.get("signal_intent_id"):
            signal_intent_ids.add(str(lineage["signal_intent_id"]))
        if lineage.get("target_position_id"):
            target_position_ids.add(str(lineage["target_position_id"]))
        if lineage.get("research_run_id"):
            research_run_ids.add(str(lineage["research_run_id"]))
        if isinstance(event, OrderEvent):
            if event.signal_intent_id:
                signal_intent_ids.add(event.signal_intent_id)
            if event.target_position_id:
                target_position_ids.add(event.target_position_id)
    return ResearchReadModel(
        journal_path=str(journal_path),
        source_contours=tuple(sorted(contours)),
        signal_intent_ids=tuple(sorted(signal_intent_ids)),
        target_position_ids=tuple(sorted(target_position_ids)),
        research_run_ids=tuple(sorted(research_run_ids)),
    )


def _build_risk_read_model(journal_path: Path, *, ops: OpsReadModel) -> RiskReadModel:
    records = load_event_journal(journal_path)
    latest_positions: dict[str, PositionSnapshot] = {}
    latest_equity: str | None = (
        ops.latest_account_summary["equity"] if ops.latest_account_summary else None
    )
    for record in records:
        event = decode_event_record(record)
        if isinstance(event, PositionSnapshot):
            latest_positions[event.instrument.symbol] = event
        if isinstance(event, AccountSummary):
            latest_equity = str(event.equity)
    net_exposure = {
        symbol: str(snapshot.net_quantity)
        for symbol, snapshot in sorted(latest_positions.items())
    }
    return RiskReadModel(
        journal_path=str(journal_path),
        open_order_ids=ops.open_order_ids,
        position_symbols=ops.position_symbols,
        net_exposure_by_symbol=net_exposure,
        latest_equity=latest_equity,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m statarb.runtime.read_model_gateway",
        description="Export frozen MVP read-model JSON files for future UI consumption.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser(
        "export",
        help="Write ops/research/risk read-model JSON files plus manifest.json.",
    )
    export.add_argument("--journal-path", required=True)
    export.add_argument("--output-dir", required=True)
    return parser


def _coerce_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _print_summary(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


__all__ = [
    "MvpReadModelBundle",
    "READ_MODEL_SCHEMA_VERSION",
    "ResearchReadModel",
    "RiskReadModel",
    "build_mvp_read_model_bundle",
    "export_mvp_read_model_bundle",
]
