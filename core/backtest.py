"""Compatibility entry points for the shared statarb financial backtester.

Raw close arrays alone cannot describe a funded futures backtest. Use an explicit
BacktestRequest, including OHLCV, historical specifications and cost assumptions.
"""

from dataclasses import replace

import numpy as np

from core.config import DEFAULT_FUTURES
from core.utils import generate_symbol_pairs, check_data_files_exist
from core.interfaces import IBacktester
from statarb.domain import InstrumentRef, InstrumentType
from statarb.research.backtest import BacktestRequest, run_backtest
from statarb.research.config import StrategyConfig
from statarb.research.metrics import block_bootstrap, daily_equity
from statarb.research.models import signal_for_zscore

futures = DEFAULT_FUTURES
check_files = check_data_files_exist
MIGRATION = (
    "Financial futures backtests require BacktestRequest with timestamped OHLCV, "
    "historical InstrumentCatalog, initial_cash, pair_budget, max_hedge_error, "
    "explicit ExecutionCosts and PaperRiskPolicy. See docs/superpowers/runbooks/causal-backtest.md."
)


def legacy_result(result, request):
    model = result.models[-1] if result.models else None
    equity = result.equity.astype(float)
    previous = equity.shift(1)
    previous.iloc[0] = float(request.initial_cash)
    returns = (equity / previous - 1).to_numpy()
    cointegration = model.to_dict() if model else {}
    cointegration["is_cointegrated"] = model.eligible if model else False
    return {
        "cointegration": cointegration,
        "performance": result.performance,
        "risk_analysis": result.risk_analysis,
        "returns": returns,
        "signals": np.array([decision.signal for decision in result.decisions]),
        "equity_curve": result.equity,
        "financial_result": result,
    }


class Backtester(IBacktester):
    def __init__(
        self,
        series_1=None,
        series_2=None,
        lookback=60,
        entry_threshold=2.0,
        exit_threshold=0.5,
        *,
        request: BacktestRequest | None = None,
        **legacy_parameters,
    ):
        if (
            isinstance(series_1, BacktestRequest)
            and request is None
            and series_2 is None
        ):
            request = series_1
            series_1 = None
        if (
            "exchange_commission" in legacy_parameters
            or "exchange_commission_rate" in legacy_parameters
        ):
            raise ValueError(
                "Ambiguous legacy exchange commission. Specify exchange_per_contract in RUB "
                "or notional_fee_bps explicitly in ExecutionCosts."
            )
        if request is not None and (
            series_1 is not None
            or series_2 is not None
            or legacy_parameters
            or (lookback, entry_threshold, exit_threshold) != (60, 2.0, 0.5)
        ):
            raise ValueError(
                "Do not combine BacktestRequest with legacy arrays or trading assumptions."
            )
        self.request = request
        self.series_1 = np.asarray(series_1) if series_1 is not None else None
        self.series_2 = np.asarray(series_2) if series_2 is not None else None
        self.config = (
            request.strategy
            if request
            else StrategyConfig(
                lookback=lookback,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
            )
        )
        self.lookback = self.config.lookback
        self.entry_threshold, self.exit_threshold = (
            self.config.entry_threshold,
            self.config.exit_threshold,
        )
        self.signals = self.returns = self.performance = self.risk_analysis = (
            self.result
        ) = None

    def generate_signals(self, z_score):
        # Diagnostic directions only. The financial engine separately confirms fills.
        position = 0
        signals = []
        for value in np.asarray(z_score, dtype=float):
            position = signal_for_zscore(value, position, self.config)
            signals.append(position)
        self.signals = np.asarray(signals)
        return self.signals

    def run_full_backtest(self, analyzer=None):
        if self.request is None:
            raise ValueError(MIGRATION)
        self.result = run_backtest(self.request)
        result = legacy_result(self.result, self.request)
        self.signals, self.returns = result["signals"], result["returns"]
        self.performance, self.risk_analysis = (
            result["performance"],
            result["risk_analysis"],
        )
        return result

    def run_backtest(self, analyzer=None):
        return self.run_full_backtest(analyzer)

    def calculate_returns(self, beta=None):
        if beta is not None:
            raise ValueError(
                "A scalar beta cannot reconstruct historical holdings. " + MIGRATION
            )
        if self.result is None:
            raise ValueError("Run the funded BacktestRequest first.")
        return self.returns

    def performance_metrics(self):
        if self.result is None:
            raise ValueError("Run the funded BacktestRequest first.")
        return self.result.performance

    def monte_carlo_analysis(self, *, config):
        if self.result is None:
            raise ValueError("Run the funded BacktestRequest first.")
        equity = self.result.equity.astype(float)
        labels = None
        if self.request.trading_calendar is not None:
            from datetime import timedelta

            labels = [
                self.request.trading_calendar[
                    at.to_pydatetime()
                    - timedelta(seconds=self.request.timeframe_seconds)
                ]
                for at in equity.index
            ]
        daily = daily_equity(equity, trading_days=labels)
        previous = daily.shift(1)
        previous.iloc[0] = float(self.request.initial_cash)
        return block_bootstrap((daily / previous - 1).to_numpy(), config)


def print_backtest_results(results, layout="single", pair_name=None, backtester=None):
    from core.formatters import HorizontalResultsFormatter, RichResultsFormatter

    formatter = (
        HorizontalResultsFormatter()
        if layout == "horizontal"
        else RichResultsFormatter()
    )
    formatter.format_single_result(results, pair_name)


def print_summary_results(results_dict):
    from core.formatters import RichResultsFormatter

    RichResultsFormatter().format_summary_results(results_dict)


def create_enhanced_backtester(
    symbol1,
    symbol2,
    data_manager,
    *,
    instrument_catalog,
    initial_cash,
    pair_budget,
    max_hedge_error,
    costs,
    risk_policy,
    strategy=None,
    timeframe_seconds=3600,
    **kwargs,
):
    request = BacktestRequest(
        InstrumentRef(symbol1, "MOEX", InstrumentType.FUTURE),
        InstrumentRef(symbol2, "MOEX", InstrumentType.FUTURE),
        data_manager.storage.load_data(symbol1),
        data_manager.storage.load_data(symbol2),
        instrument_catalog,
        initial_cash,
        pair_budget,
        max_hedge_error,
        costs,
        risk_policy,
        timeframe_seconds=timeframe_seconds,
        strategy=strategy or StrategyConfig(),
        **kwargs,
    )
    return Backtester(request=request)


def run_comparison_backtest(symbol1, symbol2, data_manager, **kwargs):
    realistic = create_enhanced_backtester(symbol1, symbol2, data_manager, **kwargs)
    costs = realistic.request.costs
    if isinstance(costs, dict):
        baseline_costs = {
            inst: replace(cost, spread_bps=0, slippage_bps=0)
            for inst, cost in costs.items()
        }
    else:
        baseline_costs = replace(costs, spread_bps=0, slippage_bps=0)
    baseline = Backtester(request=replace(realistic.request, costs=baseline_costs))
    return {
        "naive": {"results": baseline.run_full_backtest(), "backtester": baseline},
        "realistic": {
            "results": realistic.run_full_backtest(),
            "backtester": realistic,
        },
    }


__all__ = [
    "Backtester",
    "create_enhanced_backtester",
    "run_comparison_backtest",
    "print_backtest_results",
    "print_summary_results",
    "generate_symbol_pairs",
    "check_files",
]


if __name__ == "__main__":
    from statarb.research.cli import main

    raise SystemExit(main())
