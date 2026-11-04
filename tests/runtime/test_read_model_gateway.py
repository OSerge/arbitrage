from __future__ import annotations

import json
from pathlib import Path

from statarb.runtime.paper_trading_day import run_default_paper_trading_day
from statarb.runtime.read_model_gateway import (
    READ_MODEL_SCHEMA_VERSION,
    build_mvp_read_model_bundle,
    export_mvp_read_model_bundle,
    main,
)


def test_build_mvp_read_model_bundle_from_default_paper_day(tmp_path: Path) -> None:
    journal_path = tmp_path / "paper-day-bundle.jsonl"
    run_default_paper_trading_day(journal_path=journal_path)

    bundle = build_mvp_read_model_bundle(journal_path)

    assert bundle.schema_version == READ_MODEL_SCHEMA_VERSION
    assert bundle.ops.total_events == 5
    assert bundle.research.source_contours == ("paper",)
    assert bundle.risk.position_symbols == ("SPM5", "SRM5")
    assert bundle.risk.net_exposure_by_symbol["SRM5"]
    assert bundle.risk.latest_equity is not None


def test_export_mvp_read_model_bundle_writes_json_files(tmp_path: Path) -> None:
    journal_path = tmp_path / "paper-day-export.jsonl"
    output_dir = tmp_path / "read-models"
    run_default_paper_trading_day(journal_path=journal_path)

    manifest_path = export_mvp_read_model_bundle(journal_path, output_dir)

    assert manifest_path == output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == READ_MODEL_SCHEMA_VERSION
    assert (output_dir / "ops.json").exists()
    assert (output_dir / "research.json").exists()
    assert (output_dir / "risk.json").exists()


def test_main_export_command_prints_manifest_location(tmp_path: Path, capsys) -> None:
    journal_path = tmp_path / "paper-day-cli-export.jsonl"
    output_dir = tmp_path / "read-models-cli"
    run_default_paper_trading_day(journal_path=journal_path)

    exit_code = main(
        ["export", "--journal-path", str(journal_path), "--output-dir", str(output_dir)]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema_version"] == READ_MODEL_SCHEMA_VERSION
    assert summary["manifest_path"] == str(output_dir / "manifest.json")
