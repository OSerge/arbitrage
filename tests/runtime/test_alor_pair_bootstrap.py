from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from statarb.adapters.alor.http_market_data import HistoricalBarsRequest, normalize_securities_response
from statarb.data.alor_fetch import main
from statarb.data.alor_storage import ingest_history_payload, persist_raw_alor_payload, write_security_snapshot_dataset
from statarb.domain.events import ReplaySessionStatus
from statarb.runtime.alor_pair_bootstrap import run_alor_pair_bootstrap

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"


def test_run_alor_pair_bootstrap_resolves_instrument_group_from_local_snapshots(tmp_path) -> None:
    left_batch, left_request = _ingest_legacy_fixture_as_alor_history(
        tmp_path=tmp_path,
        symbol="SRM5",
        request_id="srm5-enriched",
    )
    right_batch, _ = _ingest_legacy_fixture_as_alor_history(
        tmp_path=tmp_path,
        symbol="SPM5",
        request_id="spm5-enriched",
    )
    _write_rfud_enrichment_snapshot(
        tmp_path=tmp_path,
        symbol="SRM5",
        request_id="srm5-enriched-ref",
    )
    _write_rfud_enrichment_snapshot(
        tmp_path=tmp_path,
        symbol="SPM5",
        request_id="spm5-enriched-ref",
    )

    result = run_alor_pair_bootstrap(
        storage_root=tmp_path,
        left_symbol="SRM5",
        right_symbol="SPM5",
        exchange="MOEX",
        timeframe="60",
        from_time=left_request.from_time,
        to_time=left_request.to_time,
        journal_path=tmp_path / "pair-enriched-smoke.jsonl",
    )

    assert result.left_manifest_path == left_batch.manifest_path
    assert result.right_manifest_path == right_batch.manifest_path
    assert result.left_instrument_group == "RFUD"
    assert result.right_instrument_group == "RFUD"
    assert result.smoke_result is not None
    assert result.smoke_result.replay_result.session.status is ReplaySessionStatus.SUCCEEDED


def test_run_alor_pair_bootstrap_discovers_datasets_and_runs_smoke(tmp_path) -> None:
    left_batch, left_request = _ingest_legacy_fixture_as_alor_history(
        tmp_path=tmp_path,
        symbol="SRM5",
        request_id="srm5-bootstrap",
    )
    right_batch, _ = _ingest_legacy_fixture_as_alor_history(
        tmp_path=tmp_path,
        symbol="SPM5",
        request_id="spm5-bootstrap",
    )

    result = run_alor_pair_bootstrap(
        storage_root=tmp_path,
        left_symbol="SRM5",
        right_symbol="SPM5",
        exchange="MOEX",
        timeframe="60",
        instrument_group=left_request.instrument_group,
        from_time=left_request.from_time,
        to_time=left_request.to_time,
        journal_path=tmp_path / "pair-smoke.jsonl",
    )

    assert result.left_manifest_path == left_batch.manifest_path
    assert result.right_manifest_path == right_batch.manifest_path
    assert result.bundle.research_run.strategy_id == "alor-normalized-pairs-walk-forward"
    assert result.bundle.source_pair == ("SRM5", "SPM5")
    assert result.smoke_result is not None
    assert result.smoke_result.replay_result.session.status is ReplaySessionStatus.SUCCEEDED
    assert result.smoke_result.journal_path == tmp_path / "pair-smoke.jsonl"


def test_main_pair_bootstrap_command_prints_summary(tmp_path, capsys) -> None:
    left_batch, left_request = _ingest_legacy_fixture_as_alor_history(
        tmp_path=tmp_path,
        symbol="SRM5",
        request_id="srm5-cli",
    )
    right_batch, _ = _ingest_legacy_fixture_as_alor_history(
        tmp_path=tmp_path,
        symbol="SPM5",
        request_id="spm5-cli",
    )

    exit_code = main(
        [
            "pair-bootstrap",
            "--storage-root",
            str(tmp_path),
            "--left-symbol",
            "SRM5",
            "--right-symbol",
            "SPM5",
            "--exchange",
            "MOEX",
            "--timeframe",
            "60",
            "--instrument-group",
            left_request.instrument_group or "",
            "--from",
            str(left_request.from_time),
            "--to",
            str(left_request.to_time),
            "--journal-path",
            str(tmp_path / "pair-cli-smoke.jsonl"),
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["left_dataset_id"] == left_batch.manifest.dataset_id
    assert summary["right_dataset_id"] == right_batch.manifest.dataset_id
    assert summary["pair"] == ["SRM5", "SPM5"]
    assert summary["smoke"]["replay_status"] == "SUCCEEDED"
    assert summary["smoke"]["journal_path"] == str(tmp_path / "pair-cli-smoke.jsonl")


def _ingest_legacy_fixture_as_alor_history(
    *,
    tmp_path: Path,
    symbol: str,
    request_id: str,
):
    frame = pd.read_csv(_DATA_DIR / f"{symbol}.csv").tail(700).reset_index(drop=True)
    request = HistoricalBarsRequest(
        symbol=symbol,
        exchange="MOEX",
        instrument_group="RFUD",
        tf="60",
        from_time=int(frame.iloc[0]["time"]),
        to_time=int(frame.iloc[-1]["time"]),
        split_adjust=True,
    )
    payload = {
        "history": [
            {
                "time": int(row.time),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": int(row.volume),
            }
            for row in frame.itertuples(index=False)
        ]
    }
    batch = ingest_history_payload(
        storage_root=tmp_path,
        request=request,
        payload=payload,
        fetched_at_utc=datetime(2026, 5, 21, 12, 0, tzinfo=UTC),
        request_id=request_id,
    )
    return batch, request


def _write_rfud_enrichment_snapshot(
    *,
    tmp_path: Path,
    symbol: str,
    request_id: str,
) -> None:
    observed_at = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    securities_payload = [
        {
            "symbol": symbol,
            "exchange": "MOEX",
            "market": "FORTS",
            "board": "RFUD",
            "primaryBoard": "RFUD",
            "type": "Future",
        }
    ]
    securities_raw = persist_raw_alor_payload(
        storage_root=tmp_path,
        relative_uri=(
            f"data/raw/vendor=alor/dataset=instruments/load_date=2026-05-21/request_id={request_id}/response.json.gz"
        ),
        payload=securities_payload,
    )
    write_security_snapshot_dataset(
        storage_root=tmp_path,
        records=normalize_securities_response(securities_payload),
        observed_at_utc=observed_at,
        request_id=request_id,
        raw_storage_uri=securities_raw.storage_uri,
        request_params={"query": symbol, "exchange": "MOEX"},
    )
