from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from statarb.bridges.historical import (
    HistoricalSmokeBundle,
    build_default_historical_smoke,
)
from statarb.runtime.historical import (
    HistoricalPaperReplaySmokeResult,
    run_historical_paper_replay_smoke,
)


@dataclass(frozen=True, slots=True)
class PaperTradingDaySummary:
    research_run_id: str
    source_pair: tuple[str, str]
    journal_path: Path
    paper_result_count: int
    journal_event_count: int
    replay_status: str
    source_contour: str
    source_event_count: int
    replayed_event_count: int
    position_symbols: tuple[str, ...]
    replay_parity: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "research_run_id": self.research_run_id,
            "source_pair": list(self.source_pair),
            "journal_path": str(self.journal_path),
            "paper_result_count": self.paper_result_count,
            "journal_event_count": self.journal_event_count,
            "replay_status": self.replay_status,
            "source_contour": self.source_contour,
            "source_event_count": self.source_event_count,
            "replayed_event_count": self.replayed_event_count,
            "position_symbols": list(self.position_symbols),
            "replay_parity": self.replay_parity,
        }


def build_paper_trading_day_summary(
    result: HistoricalPaperReplaySmokeResult,
) -> PaperTradingDaySummary:
    replay_parity = (
        result.replay_position_snapshots_by_symbol
        == result.position_snapshots_by_symbol
        and result.replay_result.account_summary == result.account_summary
    )
    return PaperTradingDaySummary(
        research_run_id=result.bundle.research_run.run_id,
        source_pair=result.bundle.source_pair,
        journal_path=result.journal_path,
        paper_result_count=len(result.paper_results),
        journal_event_count=len(result.journal_records),
        replay_status=result.replay_result.session.status.name,
        source_contour=result.replay_result.session.source_contour.value,
        source_event_count=result.replay_result.session.source_event_count,
        replayed_event_count=result.replay_result.session.replayed_event_count,
        position_symbols=tuple(sorted(result.position_snapshots_by_symbol)),
        replay_parity=replay_parity,
    )


def run_default_paper_trading_day(
    *,
    journal_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    bundle: HistoricalSmokeBundle | None = None,
) -> HistoricalPaperReplaySmokeResult:
    resolved_bundle = bundle or build_default_historical_smoke()
    resolved_journal_path = (
        Path(journal_path)
        if journal_path is not None
        else _default_journal_path(
            artifacts_dir=artifacts_dir,
            bundle=resolved_bundle,
        )
    )
    return run_historical_paper_replay_smoke(
        bundle=resolved_bundle,
        journal_path=resolved_journal_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = run_default_paper_trading_day(
        journal_path=args.journal_path,
        artifacts_dir=args.artifacts_dir,
    )
    summary = build_paper_trading_day_summary(result)
    _print_summary(summary.to_dict())
    return 0 if summary.replay_parity and summary.replay_status == "SUCCEEDED" else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m statarb.runtime.paper_trading_day",
        description=(
            "Run the default historical paper/replay trading-day smoke and print a redacted summary."
        ),
    )
    parser.add_argument(
        "--journal-path",
        help="Optional explicit JSONL journal path. Defaults to artifacts/historical-smoke/<pair>-<timestamp>.jsonl",
    )
    parser.add_argument(
        "--artifacts-dir",
        help="Base artifacts directory used when --journal-path is omitted.",
    )
    return parser


def _default_journal_path(
    *,
    artifacts_dir: str | Path | None,
    bundle: HistoricalSmokeBundle,
) -> Path:
    base_dir = (
        Path(artifacts_dir)
        if artifacts_dir is not None
        else _project_root() / "artifacts"
    )
    timestamp_slug = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    pair_slug = "-".join(bundle.source_pair).lower()
    return base_dir / "historical-smoke" / f"{pair_slug}-{timestamp_slug}.jsonl"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _print_summary(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
