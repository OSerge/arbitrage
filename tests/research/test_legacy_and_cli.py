from dataclasses import asdict
import json

import numpy as np
import pytest

from core.backtest import Backtester
from core.backtest_improved import ImprovedBacktester, SignalGenerator
from core.pipeline import ArbitragePipeline
from statarb.research.cli import main


def test_legacy_financial_calls_require_explicit_economics():
    with pytest.raises(ValueError, match="BacktestRequest"):
        Backtester(np.arange(600), np.arange(600)).run_full_backtest()
    with pytest.raises(ValueError, match="Ambiguous legacy exchange commission"):
        Backtester(exchange_commission=1)
    with pytest.raises(ValueError, match="ambiguous units"):
        ArbitragePipeline(["LEFT", "RIGHT"], exchange_commission=1)


def test_old_generators_share_crossing_and_nan_policy():
    series = np.array([3.0, np.nan, -3.0, -3.0, 3.0])
    expected = [-1, -1, 0, 1, 0]
    np.testing.assert_array_equal(Backtester().generate_signals(series), expected)
    np.testing.assert_array_equal(SignalGenerator().generate_signals(series), expected)


def test_both_legacy_backtesters_use_one_kernel(request_factory):
    request = request_factory()
    old = Backtester(request=request).run_full_backtest()
    improved = ImprovedBacktester(request=request).run_backtest()
    np.testing.assert_array_equal(old["returns"], improved["returns"])
    assert old["performance"] == improved["performance"]


def test_pipeline_runs_without_legacy_commission_type_error(request_factory, tmp_path):
    request = request_factory()
    for inst, frame in zip(
        request.instruments, (request.left_bars, request.right_bars), strict=True
    ):
        frame.to_csv(tmp_path / f"{inst.symbol}.csv", index=False)
    pipeline = ArbitragePipeline(
        [inst.symbol for inst in request.instruments],
        data_directory=tmp_path,
        instrument_catalog=request.instrument_catalog,
        initial_cash=request.initial_cash,
        pair_budget=request.pair_budget,
        max_hedge_error=request.max_hedge_error,
        costs=request.costs,
        risk_policy=request.risk_policy,
        lookback=4,
        min_estimation_window=12,
        recompute_frequency=4,
        min_avg_volume=0,
    )
    results = pipeline.run_pipeline(download=False)
    assert "TESTH6-HEDGEH6" in results["results"]
    assert "independent pair accounts" in results["accounting"]


def test_cli_uses_config_relative_inputs_and_writes_json(
    request_factory, tmp_path, capsys
):
    request = request_factory()
    request.left_bars.to_csv(tmp_path / "left.csv", index=False)
    request.right_bars.to_csv(tmp_path / "right.csv", index=False)
    (tmp_path / "specs.json").write_text(
        json.dumps(
            {
                "instruments": [
                    spec.to_dict() for spec in request.instrument_catalog.specs
                ]
            }
        )
    )
    config = {
        "left": {"instrument": request.left_instrument.to_dict(), "bars": "left.csv"},
        "right": {
            "instrument": request.right_instrument.to_dict(),
            "bars": "right.csv",
        },
        "specifications": "specs.json",
        "initial_cash": "1000",
        "pair_budget": "200",
        "max_hedge_error": "0.1",
        "costs": {name: str(value) for name, value in asdict(request.costs).items()},
        "strategy": asdict(request.strategy),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "result.json"
    assert (
        main(
            [
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--journal",
                str(tmp_path / "events.jsonl"),
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert (
        result["status"] == "no_trades"
    )  # Constant second leg is correctly ineligible under the real estimator.
    assert result["schema_version"] == 2
    assert json.loads(capsys.readouterr().out)["output"] == str(output)


def test_report_cannot_replace_the_execution_journal(tmp_path, capsys):
    path = tmp_path / "same.jsonl"
    with pytest.raises(SystemExit) as exit_:
        main(
            [
                "--config",
                str(tmp_path / "unused.json"),
                "--output",
                str(path),
                "--journal",
                str(path),
            ]
        )
    assert exit_.value.code == 2
    assert "Output and journal paths must differ" in capsys.readouterr().err
    assert not path.exists()


@pytest.mark.parametrize(
    "input_name",
    ["config.json", "left.csv", "right.csv", "specs.json", "calendar.json"],
)
@pytest.mark.parametrize("destination", ["output", "journal"])
def test_cli_rejects_output_overwriting_any_input_before_execution(
    tmp_path, capsys, input_name, destination
):
    config_path = tmp_path / "config.json"
    payload = {
        "left": {"bars": "left.csv"},
        "right": {"bars": "right.csv"},
        "specifications": "specs.json",
        "trading_calendar": "calendar.json",
    }
    config_path.write_text(json.dumps(payload))
    for name in ("left.csv", "right.csv", "specs.json", "calendar.json"):
        (tmp_path / name).write_text("original input")
    protected = tmp_path / input_name
    original = protected.read_bytes()
    output = protected if destination == "output" else tmp_path / "result.json"
    journal = protected if destination == "journal" else tmp_path / "events.jsonl"
    with pytest.raises(SystemExit) as exit_:
        main(
            [
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--journal",
                str(journal),
            ]
        )
    assert exit_.value.code == 2
    assert "overwrite" in capsys.readouterr().err
    assert protected.read_bytes() == original
    assert not (journal if destination == "output" else output).exists()
