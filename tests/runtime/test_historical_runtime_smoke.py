from __future__ import annotations

from decimal import Decimal

from statarb.bridges.historical import build_default_historical_smoke
from statarb.domain.events import ExecutionContour, ReplaySessionStatus
from statarb.runtime.historical import run_historical_paper_replay_smoke


def test_historical_bridge_runs_end_to_end_through_paper_and_replay(tmp_path) -> None:
    bundle = build_default_historical_smoke()

    result = run_historical_paper_replay_smoke(
        bundle=bundle,
        journal_path=tmp_path / "historical-smoke.jsonl",
    )

    assert result.bundle.research_run.run_id == bundle.research_run.run_id
    assert result.bundle.source_pair == ("SRM5", "SPM5")
    assert len(result.paper_results) == 2
    assert len(result.journal_records) == 8

    assert result.replay_result.session.status is ReplaySessionStatus.SUCCEEDED
    assert result.replay_result.session.source_contour is ExecutionContour.PAPER
    assert result.replay_result.session.source_event_count == 8
    assert result.replay_result.session.replayed_event_count == 8

    assert result.fill_prices_by_symbol == {
        "SRM5": Decimal("31379.0"),
        "SPM5": Decimal("31265.0"),
    }

    assert result.position_snapshots_by_symbol["SRM5"].net_quantity == bundle.target_positions[0].target_quantity
    assert result.position_snapshots_by_symbol["SPM5"].net_quantity == bundle.target_positions[1].target_quantity
    assert result.replay_position_snapshots_by_symbol["SRM5"] == result.position_snapshots_by_symbol["SRM5"]
    assert result.replay_position_snapshots_by_symbol["SPM5"] == result.position_snapshots_by_symbol["SPM5"]

    assert result.account_summary is not None
    assert result.replay_result.account_summary == result.account_summary
