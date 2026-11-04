"""Deprecated names backed by the same causal strategy/accounting implementation."""

import numpy as np

from core.backtest import Backtester, MIGRATION
from statarb.research.config import StrategyConfig
from statarb.research.metrics import block_bootstrap, equity_metrics
from statarb.research.models import signal_for_zscore


class SignalGenerator:
    def __init__(self, entry_threshold=2.0, exit_threshold=0.5):
        self.config = StrategyConfig(
            entry_threshold=entry_threshold, exit_threshold=exit_threshold
        )

    def generate_signals(self, z_score):
        result, position = [], 0
        for value in np.asarray(z_score, dtype=float):
            position = signal_for_zscore(value, position, self.config)
            result.append(position)
        return np.asarray(result)


class ReturnsCalculator:
    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise ValueError(
                "Specify typed ExecutionCosts on BacktestRequest. " + MIGRATION
            )

    def calculate_returns(self, *args, **kwargs):
        raise ValueError(
            "Independent close-array returns have been removed. " + MIGRATION
        )


class PerformanceAnalyzer:
    def calculate_metrics(self, equity, *, initial_equity, started_at, **kwargs):
        return equity_metrics(
            equity, initial_equity=initial_equity, started_at=started_at, **kwargs
        )


class RiskAnalyzer:
    def monte_carlo_analysis(self, daily_returns, *, config):
        return block_bootstrap(daily_returns, config)


class ImprovedBacktester(Backtester):
    """Compatibility alias; there is no separate full-sample model/return path."""
