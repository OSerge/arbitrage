from dataclasses import replace
from decimal import Decimal
import json

import numpy as np
import pandas as pd
import pytest

from statarb.research.backtest import run_backtest, run_universe
from statarb.research.models import signal_for_zscore
from statarb.runtime.replay.loader import decode_event_record, load_event_journal
from statarb.runtime.replay.replayer import RuntimeReplayer

from .conftest import fixed_model, synthetic_bars


def test_round_trip_charges_each_leg_on_entry_and_exit(request_factory):
    result = run_backtest(request_factory(), estimator=fixed_model)
    fills = [
        decode_event_record(record)
        for record in result.journal_records
        if record.event_type == "fill_event"
    ]
    assert len(fills) == 4
    assert [fill.occurred_at.hour for fill in fills] == [13, 13, 14, 14]
    assert result.costs["commission"] == 8
    assert result.costs["gross_pnl"] == 0
    assert result.account_summary.equity == 992
    assert result.costs["net_pnl"] == -8
    assert result.performance["closed_pairs"] == 1
    assert not any(
        snapshot.net_quantity for snapshot in result.position_snapshots.values()
    )
    assert result.research_run.signal_count == 4
    assert len(result.signal_intents) == 4
    assert all(
        record.payload["signal_intent_id"]
        in {intent.intent_id for intent in result.signal_intents}
        for record in result.journal_records
        if record.event_type == "order_event"
    )
    json.dumps(result.to_dict(), allow_nan=False)


def test_future_suffix_does_not_rewrite_models_fills_or_equity(request_factory):
    def changing_beta(left, right, **kwargs):
        return replace(fixed_model(left, right, **kwargs), beta=1 + len(left) / 100)

    short_request = request_factory(24, max_hedge_error=Decimal("0.3"))
    longer = request_factory(36, max_hedge_error=Decimal("0.3"))
    longer.left_bars.loc[24:, ["open", "high", "low", "close"]] *= 2
    short = run_backtest(short_request, estimator=changing_beta)
    long = run_backtest(longer, estimator=changing_beta)
    assert short.models == long.models[: len(short.models)]
    assert short.decisions == long.decisions[:24]
    pd.testing.assert_series_equal(short.equity, long.equity.iloc[: len(short.equity)])
    assert short.journal_records == long.journal_records[: len(short.journal_records)]
    assert short.performance["fill_count"] > 0


def test_universe_refits_after_a_gap_use_the_actual_last_formation_bar(
    request_factory, monkeypatch
):
    monkeypatch.setattr("statarb.research.models.estimate_model", fixed_model)
    request = request_factory()
    for frame in (request.left_bars, request.right_bars):
        frame.loc[12:, "time"] += 48 * 3600
    pair = tuple(inst.symbol for inst in request.instruments)
    result = run_universe({pair: request})[pair]
    expected = run_backtest(request, estimator=fixed_model)
    assert result.models == expected.models
    assert result.journal_records == expected.journal_records
    pd.testing.assert_series_equal(result.equity, expected.equity)
    assert result.performance["fill_count"] == 4


def test_open_position_freezes_entry_model_at_later_refit(request_factory):
    prices = np.r_[
        np.tile([99.0, 101.0], 6),
        [110.0, 120.0, 140.0, 160.0, 200.0, 260.0, 400.0, 600.0],
    ]
    request = request_factory(20, left_bars=synthetic_bars(prices))

    def estimator(left, right, **kwargs):
        return replace(
            fixed_model(left, right, **kwargs), beta=1 if len(left) < 16 else 2
        )

    result = run_backtest(request, estimator=estimator)
    assert result.models[-1].beta == 2
    assert result.decisions[16].position == -1
    assert result.decisions[16].model.beta == 1
    assert result.performance["fill_count"] == 2
    assert result.estimated_liquidation_cost == 4


def test_lost_cointegration_preserves_earlier_pnl(request_factory):
    def estimator(left, right, **kwargs):
        return replace(fixed_model(left, right, **kwargs), eligible=len(left) < 16)

    result = run_backtest(request_factory(), estimator=estimator)
    assert not result.models[-1].eligible
    assert result.costs["net_pnl"] == -8
    assert result.status == "completed"


def test_loss_of_eligibility_closes_an_existing_position_next_open(request_factory):
    prices = np.r_[
        np.tile([99.0, 101.0], 6),
        [110.0, 120.0, 140.0, 160.0, 200.0, 260.0, 400.0, 600.0],
    ]
    request = request_factory(20, left_bars=synthetic_bars(prices))

    def estimator(left, right, **kwargs):
        return replace(fixed_model(left, right, **kwargs), eligible=len(left) < 16)

    result = run_backtest(request, estimator=estimator)
    assert result.decisions[16].position == -1
    assert result.decisions[16].signal == 0
    fills = [
        decode_event_record(record)
        for record in result.journal_records
        if record.event_type == "fill_event"
    ]
    assert [fill.occurred_at.hour for fill in fills] == [13, 13, 17, 17]
    assert result.costs["net_pnl"] == -8


def test_spread_and_adverse_tick_rounding_charge_both_sides(request_factory):
    request = request_factory()
    result = run_backtest(
        replace(request, costs=replace(request.costs, spread_bps=Decimal("100"))),
        estimator=fixed_model,
    )
    assert result.costs["execution_drag"] == 4
    assert result.costs["commission"] == 8
    assert result.costs["net_pnl"] == -12
    assert result.costs["gross_pnl"] == 0


def test_future_intrabar_range_does_not_change_open_execution(request_factory):
    original = request_factory()
    wider = request_factory()
    wider.left_bars.loc[13, "high"] = 10000
    expected = run_backtest(original, estimator=fixed_model)
    actual = run_backtest(wider, estimator=fixed_model)
    assert actual.journal_records == expected.journal_records
    assert actual.costs == expected.costs


def test_no_trades_is_a_valid_result(request_factory):
    def ineligible(left, right, **kwargs):
        return replace(fixed_model(left, right, **kwargs), eligible=False)

    result = run_backtest(request_factory(), estimator=ineligible)
    assert result.status == "no_trades"
    assert (result.equity == 1000).all()


def test_exit_crossing_never_reverses_and_nan_does_not_flatten(request_factory):
    config = request_factory().strategy
    assert signal_for_zscore(-4, -1, config) == 0
    assert signal_for_zscore(4, 1, config) == 0
    assert signal_for_zscore(float("nan"), -1, config) == -1
    assert signal_for_zscore(float("inf"), 1, config) == 1


def test_full_journal_reconstructs_the_futures_account(request_factory, tmp_path):
    request = request_factory()
    journal = tmp_path / "backtest.jsonl"
    result = run_backtest(request, estimator=fixed_model, journal_path=journal)
    records = load_event_journal(journal)
    replay = RuntimeReplayer(
        account_id="backtest-pair",
        starting_cash=request.initial_cash,
        instrument_catalog=request.instrument_catalog,
    ).replay(records)
    assert replay.account_summary.equity == result.account_summary.equity
    assert (
        replay.account_summary.available_cash == result.account_summary.available_cash
    )
    assert all(
        replay.position_snapshots[inst].net_quantity == snapshot.net_quantity
        for inst, snapshot in result.position_snapshots.items()
    )
    assert replay.position_snapshots == result.position_snapshots
    assert replay.account_summary == result.account_summary


def test_costs_cannot_improve_a_fixed_round_trip(request_factory):
    request = request_factory()
    baseline = run_backtest(request, estimator=fixed_model)
    expensive = run_backtest(
        replace(
            request, costs=replace(request.costs, broker_per_contract=Decimal("5"))
        ),
        estimator=fixed_model,
    )
    assert expensive.performance["fill_count"] == baseline.performance["fill_count"]
    assert baseline.account_summary.equity - expensive.account_summary.equity == 16


def test_missing_historical_metadata_fails_before_journal_write(
    request_factory, tmp_path
):
    from statarb.domain import InstrumentCatalog

    journal = tmp_path / "never-created.jsonl"
    with pytest.raises(ValueError, match="Missing historical InstrumentSpec"):
        run_backtest(
            replace(request_factory(), instrument_catalog=InstrumentCatalog()),
            journal_path=journal,
        )
    assert not journal.exists()


def test_final_entry_cannot_fill_without_a_next_bar(request_factory):
    result = run_backtest(request_factory(13), estimator=fixed_model)
    assert result.decisions[-1].signal == -1
    assert result.performance["fill_count"] == 0
    assert any("no next common bar" in diagnostic for diagnostic in result.diagnostics)


def test_selection_cutoff_blocks_trading_before_selection_is_known(request_factory):
    request = request_factory()
    cutoff = pd.Timestamp("2026-01-05T15:00Z").to_pydatetime()
    result = run_backtest(replace(request, trading_start=cutoff), estimator=fixed_model)
    assert result.equity.index[0] >= cutoff
    assert all(
        decision.signal == 0
        for decision in result.decisions
        if decision.generated_at <= cutoff
    )
    assert result.performance["fill_count"] == 0


def test_sizing_finds_a_smaller_funded_pair_after_fees(request_factory):
    from statarb.domain import InstrumentCatalog

    request = request_factory(pair_budget=Decimal("2000"))
    catalog = InstrumentCatalog(
        replace(
            spec, initial_margin_long=Decimal("60"), initial_margin_short=Decimal("60")
        )
        for spec in request.instrument_catalog.specs
    )
    result = run_backtest(
        replace(request, instrument_catalog=catalog), estimator=fixed_model
    )
    fills = [
        decode_event_record(record)
        for record in result.journal_records
        if record.event_type == "fill_event"
    ]
    assert len(fills) == 4
    assert all(fill.fill_quantity == 8 for fill in fills)
    assert result.account_summary.equity == 936


def test_explicit_calendar_must_cover_the_whole_evaluation_before_writes(
    request_factory, tmp_path
):
    journal = tmp_path / "missing-calendar.jsonl"
    with pytest.raises(ValueError, match="Trading calendar does not cover"):
        run_backtest(
            replace(request_factory(), trading_calendar={}),
            estimator=fixed_model,
            journal_path=journal,
        )
    assert not journal.exists()


def test_real_estimator_runs_an_entire_quantized_cointegrated_history(
    request_factory, tmp_path
):
    from statarb.domain import InstrumentCatalog
    from statarb.research.config import ExecutionCosts, StrategyConfig

    rng = np.random.default_rng(17)
    right = 100 + np.cumsum(rng.normal(size=600))
    left = 10 + 1.7 * right + rng.normal(scale=0.3, size=600)
    request = request_factory(600)

    def traded_bars(values):
        frame = synthetic_bars(np.round(values, 2))
        frame["open"] = np.r_[frame.close.iloc[0], frame.close.to_numpy()[:-1]]
        frame["high"] = np.round(frame[["open", "close"]].max(axis=1) + 0.01, 2)
        frame["low"] = np.round(frame[["open", "close"]].min(axis=1) - 0.01, 2)
        return frame

    catalog = InstrumentCatalog(
        replace(spec, tick_size=Decimal("0.01"), tick_value=Decimal("0.01"))
        for spec in request.instrument_catalog.specs
    )
    request = replace(
        request,
        left_bars=traded_bars(left),
        right_bars=traded_bars(right),
        instrument_catalog=catalog,
        initial_cash=Decimal("10000"),
        pair_budget=Decimal("5000"),
        max_hedge_error=Decimal("0.05"),
        costs=ExecutionCosts("0.01", "0.01", "0", "0"),
        strategy=StrategyConfig(
            lookback=30,
            min_estimation_window=256,
            recompute_frequency=32,
            entry_threshold=1.5,
            min_avg_volume=0,
        ),
    )
    path = tmp_path / "real-estimator.jsonl"
    result = run_backtest(request, journal_path=path)
    assert result.performance["fill_count"] >= 4
    replay = RuntimeReplayer(
        account_id="backtest-pair",
        starting_cash=request.initial_cash,
        instrument_catalog=catalog,
    ).replay(load_event_journal(path))
    assert replay.account_summary == result.account_summary
    assert replay.position_snapshots == result.position_snapshots
    assert result.costs["commission"] > 0
    assert result.costs["net_pnl"] != -result.costs["commission"]
