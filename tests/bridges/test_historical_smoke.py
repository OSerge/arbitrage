from __future__ import annotations

import importlib
import importlib.util


def test_bridge_builds_contracts_from_real_legacy_pair_slice() -> None:
    module_spec = importlib.util.find_spec("statarb.bridges.historical")
    assert module_spec is not None, "statarb.bridges.historical module is missing"

    historical_bridge = importlib.import_module("statarb.bridges.historical")
    result = historical_bridge.build_default_historical_smoke()

    assert result.research_run.strategy_id == "legacy-pairs-walk-forward"
    assert result.research_run.dataset_id == "legacy-csv:SRM5-SPM5:tail-700"
    assert result.research_run.signal_count == 2
    assert result.research_run.artifact_uris == (
        "file://data/SRM5.csv",
        "file://data/SPM5.csv",
    )

    assert result.source_pair == ("SRM5", "SPM5")
    assert result.source_row_count >= 600
    assert result.source_signal == 1
    assert len(result.signal_intents) == 2
    assert len(result.target_positions) == 2

    first_intent, second_intent = result.signal_intents
    first_target, second_target = result.target_positions

    assert first_intent.instrument.symbol == "SRM5"
    assert second_intent.instrument.symbol == "SPM5"
    assert first_intent.direction.value == "long"
    assert second_intent.direction.value == "short"
    assert first_intent.generated_at == result.as_of
    assert second_intent.generated_at == result.as_of

    assert first_target.instrument.symbol == "SRM5"
    assert second_target.instrument.symbol == "SPM5"
    assert first_target.target_quantity > 0
    assert second_target.target_quantity < 0
    assert first_target.research_run_id == result.research_run.run_id
    assert second_target.research_run_id == result.research_run.run_id
