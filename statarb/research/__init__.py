"""Causal pair research and financial backtests, independent of broker connections."""

from importlib import import_module

__all__ = [
    "BacktestRequest",
    "BacktestResult",
    "run_backtest",
    "run_universe",
    "StrategyConfig",
    "ExecutionCosts",
]


def __getattr__(name):
    modules = {
        "BacktestRequest": "backtest",
        "BacktestResult": "backtest",
        "run_backtest": "backtest",
        "run_universe": "backtest",
        "StrategyConfig": "config",
        "ExecutionCosts": "config",
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{modules[name]}"), name)
