from __future__ import annotations

import json
from pathlib import Path

from statarb.runtime.paper_trading_day import run_default_paper_trading_day
from statarb.runtime.read_model import build_ops_read_model, main


def test_build_ops_read_model_from_default_paper_trading_day_journal(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "paper-day-read-model.jsonl"
    run_default_paper_trading_day(journal_path=journal_path)

    model = build_ops_read_model(journal_path)

    assert model.total_events == 5
    assert model.event_counts.order_event == 0
    assert model.event_counts.fill_event == 0
    assert model.event_counts.position_snapshot == 2
    assert model.event_counts.account_summary == 1
    assert model.event_counts.price_mark == 2
    assert model.source_contours == ("paper",)
    assert model.position_symbols == ("SPM5", "SRM5")
    assert model.open_order_ids == ()
    assert model.latest_account_summary is not None
    assert model.latest_account_summary["account_id"] == "paper-historical-smoke"


def test_main_inspect_prints_ops_read_model(tmp_path: Path, capsys) -> None:
    journal_path = tmp_path / "paper-day-cli-read-model.jsonl"
    run_default_paper_trading_day(journal_path=journal_path)

    exit_code = main(["inspect", "--journal-path", str(journal_path)])

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["total_events"] == 5
    assert summary["position_symbols"] == ["SPM5", "SRM5"]
    assert summary["open_order_ids"] == []
