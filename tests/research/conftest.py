"""Synthetic prices and economics, never historical MOEX specifications."""

from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from statarb.domain import (
    InstrumentCatalog,
    InstrumentRef,
    InstrumentSpec,
    InstrumentType,
)
from statarb.research.backtest import BacktestRequest
from statarb.research.config import ExecutionCosts, StrategyConfig
from statarb.research.models import ModelSnapshot
from statarb.runtime.paper.risk_checks import PaperRiskPolicy


def synthetic_bars(prices):
    close = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {
            "time": pd.date_range(
                "2026-01-05", periods=len(close), freq="h", tz="UTC"
            ).asi8
            // 10**9,
            "open": np.full(len(close), 100.0),
            "high": np.round(np.maximum(close, 100.0) + 1, 2),
            "low": np.round(np.minimum(close, 100.0) - 1, 2),
            "close": close,
            "volume": 1000,
        }
    )


def fixed_model(left, right, *, fitted_until, significance):
    return ModelSnapshot(
        fitted_until,
        len(left),
        alpha=0,
        beta=1,
        p_value=0.001,
        adjusted_p_value=0.001,
        eligible=True,
        reason="synthetic model",
    )


@pytest.fixture
def request_factory():
    def make(n=24, **kwargs):
        left = InstrumentRef("TESTH6", "MOEX", InstrumentType.FUTURE)
        right = InstrumentRef("HEDGEH6", "MOEX", InstrumentType.FUTURE)
        specs = [
            InstrumentSpec(
                inst,
                "synthetic-v1",
                "test fixture",
                datetime(2025, 1, 1, tzinfo=UTC),
                Decimal("1"),
                Decimal("1"),
                Decimal("1"),
                Decimal("10"),
                Decimal("10"),
                datetime(2026, 12, 1, tzinfo=UTC),
            )
            for inst in (left, right)
        ]
        prices = np.full(n, 100.0)
        prices[:12] += np.tile([-1.0, 1.0], 6)
        prices[12] = 110
        params = dict(
            left_instrument=left,
            right_instrument=right,
            left_bars=synthetic_bars(prices),
            right_bars=synthetic_bars(np.full(n, 100.0)),
            instrument_catalog=InstrumentCatalog(specs),
            initial_cash=Decimal("1000"),
            pair_budget=Decimal("200"),
            max_hedge_error=Decimal("0"),
            costs=ExecutionCosts("1", "1", "0", "0"),
            risk_policy=PaperRiskPolicy(),
            session_id="synthetic-test-run",
            strategy=StrategyConfig(
                lookback=4,
                min_estimation_window=12,
                recompute_frequency=4,
                min_avg_volume=0,
            ),
        )
        params.update(kwargs)
        return BacktestRequest(**params)

    return make
