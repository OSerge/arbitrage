from __future__ import annotations

import json
from pathlib import Path

from statarb.runtime.paper_trading_day import build_paper_trading_day_summary, main, run_default_paper_trading_day


def test_run_default_paper_trading_day_produces_replay_parity(tmp_path: Path) -> None:
    journal_path = tmp_path / "paper-day.jsonl"

    result = run_default_paper_trading_day(journal_path=journal_path)
    summary = build_paper_trading_day_summary(result)

    assert summary.source_pair == ("SRM5", "SPM5")
    assert summary.paper_result_count == 2
    assert summary.journal_event_count == 8
    assert summary.replay_status == "SUCCEEDED"
    assert summary.replay_parity is True
    assert journal_path.exists()


def test_main_prints_summary_and_returns_zero_on_success(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "--journal-path",
            str(tmp_path / "paper-day-cli.jsonl"),
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["replay_status"] == "SUCCEEDED"
    assert summary["replay_parity"] is True
    assert summary["source_pair"] == ["SRM5", "SPM5"]
