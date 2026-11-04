"""Legacy orchestration facade; pair research and accounting live in statarb."""

from datetime import UTC, datetime, timedelta
from itertools import combinations
import pandas as pd

from core.analysis import DataAnalyzer
from core.backtest import MIGRATION, legacy_result
from core.data import DataManager
from core.downloader import SecurityDownloader
from statarb.data.bars import prepare_pair_bars
from statarb.domain import InstrumentRef, InstrumentType
from statarb.research.backtest import BacktestRequest, run_universe
from statarb.research.config import StrategyConfig
from statarb.research.models import select_models


class ArbitragePipeline:
    def __init__(
        self,
        symbols,
        lookback=60,
        entry_threshold=2.0,
        exit_threshold=0.5,
        broker_commission=None,
        exchange_commission=None,
        max_workers=5,
        min_avg_volume=200,
        *,
        instrument_catalog=None,
        initial_cash=None,
        pair_budget=None,
        max_hedge_error=None,
        costs=None,
        risk_policy=None,
        data_directory=None,
        timeframe_seconds=3600,
        min_estimation_window=512,
        recompute_frequency=16,
    ):
        if broker_commission is not None or exchange_commission is not None:
            raise ValueError(
                "Legacy commission arguments have ambiguous units; supply explicit ExecutionCosts."
            )
        self.symbols = tuple(symbols)
        self.config = StrategyConfig(
            lookback=lookback,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            min_avg_volume=min_avg_volume,
            min_estimation_window=min_estimation_window,
            recompute_frequency=recompute_frequency,
        )
        self.min_avg_volume = min_avg_volume
        self.timeframe_seconds = timeframe_seconds
        self.catalog, self.initial_cash, self.pair_budget = (
            instrument_catalog,
            initial_cash,
            pair_budget,
        )
        self.max_hedge_error, self.costs, self.risk_policy = (
            max_hedge_error,
            costs,
            risk_policy,
        )
        self.downloader = SecurityDownloader(max_workers=max_workers)
        self.analyzer = DataAnalyzer(alpha=self.config.alpha)
        self.data_manager = DataManager(data_directory=data_directory)
        self.data_storage = self.data_manager.storage
        self.downloader.data_storage = self.data_storage
        self.selection_as_of = None

    def _require_financial_inputs(self):
        if any(
            value is None
            for value in (
                self.catalog,
                self.initial_cash,
                self.pair_budget,
                self.max_hedge_error,
                self.costs,
                self.risk_policy,
            )
        ):
            raise ValueError(MIGRATION)

    def download_data(self, days=90):
        end = datetime.now(UTC)
        result = self.downloader.download_multiple_securities(
            symbols=list(self.symbols),
            start_time=end - timedelta(days=days),
            end_time=end,
            timeframe=self.timeframe_seconds,
            use_parallel=True,
        )
        missing = set(self.symbols) - set(result)
        if missing:
            raise ValueError(
                f"Download failed for {sorted(missing)}; refusing to reuse stale local files."
            )
        self.data_manager.cache.clear()
        return result

    def find_cointegrated_pairs(self, min_pairs=1):
        """Diagnostic selection on initial formation only; not an execution filter."""
        frames = {
            symbol: self.data_storage.load_data(symbol) for symbol in self.symbols
        }
        aligned = {
            pair: prepare_pair_bars(
                frames[pair[0]],
                frames[pair[1]],
                timeframe_seconds=self.timeframe_seconds,
            )
            for pair in combinations(self.symbols, 2)
        }
        if not aligned or any(
            len(frame) < self.config.min_estimation_window for frame in aligned.values()
        ):
            raise ValueError("Insufficient common formation bars for pair selection.")
        cutoff = max(
            frame.end_at(self.config.min_estimation_window - 1)
            for frame in aligned.values()
        )
        histories = {}
        for pair, frame in aligned.items():
            mask = frame.index + pd.Timedelta(seconds=self.timeframe_seconds) <= cutoff
            histories[pair] = (
                frame.left.close.loc[mask].to_numpy(),
                frame.right.close.loc[mask].to_numpy(),
            )
        models = select_models(
            histories, fitted_until=cutoff, significance=self.config.alpha
        )
        self.selection_as_of = cutoff
        pairs = [pair for pair, model in models.items() if model.eligible]
        return pairs if len(pairs) >= min_pairs else []

    def _is_liquid(self, frame):
        return "volume" in frame and frame.volume.mean() >= self.min_avg_volume

    def run_backtest(self, pairs):
        self._require_financial_inputs()
        requests = {}
        for left, right in pairs:
            requests[(left, right)] = BacktestRequest(
                InstrumentRef(left, "MOEX", InstrumentType.FUTURE),
                InstrumentRef(right, "MOEX", InstrumentType.FUTURE),
                self.data_manager.load_data_with_cache(left),
                self.data_manager.load_data_with_cache(right),
                self.catalog,
                self.initial_cash,
                self.pair_budget,
                self.max_hedge_error,
                self.costs,
                self.risk_policy,
                timeframe_seconds=self.timeframe_seconds,
                strategy=self.config,
                session_id=f"backtest-{left}-{right}",
                trading_start=self.selection_as_of,
            )
        results = run_universe(requests)
        return {
            f"{pair[0]}-{pair[1]}": legacy_result(result, requests[pair])
            for pair, result in results.items()
        }

    def run_pipeline(self, days=120, min_pairs=1, *, download=True):
        self._require_financial_inputs()
        if download:
            self.download_data(days)
        pairs = list(combinations(self.symbols, 2))
        self.selection_as_of = None
        if len(pairs) < min_pairs:
            raise ValueError("The configured universe has fewer pairs than min_pairs.")
        # Preserve every outcome, including unprofitable/no-trade pairs. Selection
        # occurs at each historical fit, never once on the complete future sample.
        return {
            "pairs": pairs,
            "results": self.run_backtest(pairs),
            "accounting": "independent pair accounts; results are not a pooled portfolio",
        }


if __name__ == "__main__":
    from statarb.research.cli import main

    raise SystemExit(main())
