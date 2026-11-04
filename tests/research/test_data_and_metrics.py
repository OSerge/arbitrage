from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from statarb.data.bars import prepare_pair_bars
from statarb.domain import InstrumentCatalog
from statarb.domain.events import _coerce_decimal
from statarb.research.metrics import BootstrapConfig, block_bootstrap, equity_metrics
from statarb.research.models import estimate_model
from statarb.research.sizing import size_pair


def test_alignment_uses_timestamps_before_tail_and_preserves_ohlcv(request_factory):
    request = request_factory()
    left = request.left_bars.drop(index=[2, 5])
    right = request.right_bars.drop(index=[1, 3])
    bars = prepare_pair_bars(left, right, timeframe_seconds=3600, tail_rows=10)
    assert len(bars) == 10
    assert bars.left.index.equals(bars.right.index)
    assert bars.left.close.iloc[-1] == request.left_bars.close.iloc[-1]
    assert bars.diagnostics


def test_range_excludes_bars_that_are_not_completed(request_factory):
    request = request_factory()
    cutoff = datetime(2026, 1, 5, 16, 30, tzinfo=UTC)
    bars = prepare_pair_bars(
        request.left_bars, request.right_bars, timeframe_seconds=3600, to_time=cutoff
    )
    assert len(bars) == 16
    assert bars.end_at(len(bars) - 1) <= cutoff


def test_capture_time_excludes_a_still_forming_bar_even_with_a_later_requested_end(
    request_factory,
):
    request = request_factory()
    captured = datetime(2026, 1, 5, 16, 30, tzinfo=UTC)
    bars = prepare_pair_bars(
        request.left_bars,
        request.right_bars,
        timeframe_seconds=3600,
        to_time=datetime(2026, 1, 6, tzinfo=UTC),
        available_at=captured,
    )
    assert len(bars) == 16
    assert bars.end_at(len(bars) - 1) <= captured


@pytest.mark.parametrize(
    "column,value", [("close", float("nan")), ("volume", -1), ("high", 1)]
)
def test_invalid_ohlcv_is_rejected(request_factory, column, value):
    request = request_factory()
    frame = request.left_bars.copy()
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        prepare_pair_bars(frame, request.right_bars, timeframe_seconds=3600)


def test_conflicting_duplicates_are_rejected_but_identical_duplicates_collapse(
    request_factory,
):
    request = request_factory()
    duplicated = pd.concat([request.left_bars, request.left_bars.iloc[[0]]])
    assert (
        len(prepare_pair_bars(duplicated, request.right_bars, timeframe_seconds=3600))
        == 24
    )
    duplicated.iloc[-1, duplicated.columns.get_loc("volume")] += 1
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        prepare_pair_bars(duplicated, request.right_bars, timeframe_seconds=3600)


def test_negative_beta_and_different_point_values_produce_signed_integer_hedge(
    request_factory,
):
    request = request_factory()
    left, right = request.instrument_catalog.specs
    left = replace(left, tick_value=Decimal("2"))
    size = size_pair(
        beta="-1.5",
        direction=1,
        left_price="100",
        right_price="100",
        left_spec=left,
        right_spec=right,
        budget="500",
        max_hedge_error="0",
        max_abs_position="100",
    )
    assert (size.left, size.right) == (1, 3)
    assert size.gross_notional == 500
    assert size.hedge_error == 0


def test_catalog_rejects_overlap_and_future_spec_is_not_backfilled(request_factory):
    request = request_factory()
    first = request.instrument_catalog.specs[0]
    with pytest.raises(ValueError, match="Overlapping"):
        InstrumentCatalog((first, replace(first, version="another")))
    with pytest.raises(ValueError, match="Missing historical"):
        InstrumentCatalog(
            (replace(first, valid_from=datetime(2026, 6, 1, tzinfo=UTC)),)
        ).resolve(first.instrument, datetime(2026, 1, 1, tzinfo=UTC))


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", Decimal("NaN")])
def test_nonfinite_domain_values_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        _coerce_decimal(value, "price")


def test_initial_loss_is_in_drawdown():
    index = pd.date_range("2026-01-05", periods=2, tz="UTC", freq="D")
    metrics = equity_metrics(
        pd.Series([90.0, 90.0], index=index),
        initial_equity=100,
        started_at=datetime(2026, 1, 4, tzinfo=UTC),
    )
    assert metrics["max_drawdown"] == pytest.approx(-0.1)
    assert metrics["var_95"] is None


def test_block_bootstrap_is_seeded_and_uses_compound_equity():
    config = BootstrapConfig(horizon_days=2, block_days=2, seed=7, simulations=20)
    first = block_bootstrap(np.array([0.1, -0.1]), config)
    second = block_bootstrap(np.array([0.1, -0.1]), config)
    assert first["var_95"] == pytest.approx(0.01)
    np.testing.assert_array_equal(
        first["max_drawdown_dist"], second["max_drawdown_dist"]
    )
    np.testing.assert_allclose(first["max_drawdown_dist"], -0.1)


def test_estimation_handles_constant_and_cointegrated_prices():
    at = datetime(2026, 1, 5, tzinfo=UTC)
    assert not estimate_model(np.ones(100), np.ones(100), fitted_until=at).eligible
    rng = np.random.default_rng(17)
    right = 100 + np.cumsum(rng.normal(size=600))
    left = 10 + 1.7 * right + rng.normal(scale=0.1, size=600)
    model = estimate_model(left, right, fitted_until=at)
    assert model.eligible
    assert model.beta == pytest.approx(1.7, abs=0.01)


def test_holm_correction_accounts_for_the_candidate_family(monkeypatch):
    import json
    from statarb.research import models

    at = datetime(2026, 1, 5, tzinfo=UTC)

    def estimate(left, right, **kwargs):
        return models.ModelSnapshot(
            at,
            len(left),
            beta=1,
            p_value=float(left[0]),
            adjusted_p_value=float(left[0]),
            eligible=True,
        )

    monkeypatch.setattr(models, "estimate_model", estimate)
    selected = models.select_models(
        {
            ("A", "B"): (np.array([0.03]), np.array([1.0])),
            ("A", "C"): (np.array([0.04]), np.array([1.0])),
        },
        fitted_until=at,
    )
    assert all(not model.eligible for model in selected.values())
    assert all(
        model.adjusted_p_value == pytest.approx(0.06) for model in selected.values()
    )
    json.dumps([model.to_dict() for model in selected.values()], allow_nan=False)


def test_midnight_bar_end_belongs_to_the_day_that_just_closed():
    from statarb.research.metrics import daily_equity

    index = pd.DatetimeIndex(["2026-01-05T20:00:00Z", "2026-01-05T21:00:00Z"])
    daily = daily_equity(pd.Series([100.0, 105.0], index=index))
    assert len(daily) == 1
    assert str(daily.index[0]) == "2026-01-05"
    assert daily.iloc[0] == 105


def test_supplied_calendar_groups_overnight_sessions_explicitly():
    from statarb.research.metrics import daily_equity

    index = pd.DatetimeIndex(["2026-01-05T20:00:00Z", "2026-01-06T02:00:00Z"])
    daily = daily_equity(
        pd.Series([100.0, 105.0], index=index),
        trading_days=["2026-01-06", "2026-01-06"],
    )
    assert len(daily) == 1
    assert str(daily.index[0]) == "2026-01-06"
