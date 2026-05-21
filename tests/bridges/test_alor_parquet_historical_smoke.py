from __future__ import annotations

import importlib
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import statarb.data as data_module

from statarb.adapters.alor.http_market_data import HistoricalBarsRequest
from statarb.data.alor_storage import ingest_history_payload

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"


def test_bridge_builds_contracts_from_normalized_alor_parquet_pair(tmp_path) -> None:
    loader = getattr(data_module, "load_history_bars_dataset", None)
    assert loader is not None, "statarb.data.load_history_bars_dataset export is missing"

    bridge_module = importlib.import_module("statarb.bridges.historical")
    build_from_alor = getattr(bridge_module, "build_historical_pair_bridge_from_alor_datasets", None)
    assert (
        build_from_alor is not None
    ), "statarb.bridges.historical.build_historical_pair_bridge_from_alor_datasets is missing"

    left_batch = _ingest_legacy_fixture_as_alor_history(
        tmp_path=tmp_path,
        symbol="SRM5",
        request_id="srm5-bridge",
    )
    right_batch = _ingest_legacy_fixture_as_alor_history(
        tmp_path=tmp_path,
        symbol="SPM5",
        request_id="spm5-bridge",
    )

    left_dataset = loader(storage_root=tmp_path, manifest_path=left_batch.manifest_path)
    right_dataset = loader(storage_root=tmp_path, manifest_path=right_batch.manifest_path)

    bundle = build_from_alor(
        left_dataset=left_dataset,
        right_dataset=right_dataset,
    )

    assert bundle.research_run.strategy_id == "alor-normalized-pairs-walk-forward"
    assert bundle.research_run.signal_count == 2
    assert bundle.research_run.artifact_uris == (
        f"file://{left_batch.normalized_artifact.storage_uri}",
        f"file://{right_batch.normalized_artifact.storage_uri}",
    )
    assert bundle.source_pair == ("SRM5", "SPM5")
    assert bundle.source_signal == 1
    assert bundle.source_row_count >= 600
    assert bundle.signal_intents[0].instrument.symbol == "SRM5"
    assert bundle.signal_intents[1].instrument.symbol == "SPM5"
    assert bundle.target_positions[0].target_quantity > 0
    assert bundle.target_positions[1].target_quantity < 0
    assert bundle.fill_prices_by_symbol == {
        "SRM5": Decimal("31379.0"),
        "SPM5": Decimal("31265.0"),
    }


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
    return ingest_history_payload(
        storage_root=tmp_path,
        request=request,
        payload=payload,
        fetched_at_utc=datetime(2026, 5, 21, 12, 0, tzinfo=UTC),
        request_id=request_id,
    )
