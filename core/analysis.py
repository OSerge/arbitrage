"""Legacy analysis names delegated to shared, validated research functions."""

from datetime import UTC, datetime
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from core.interfaces import IAnalyzer
from statarb.data.bars import prepare_pair_bars
from statarb.research.models import estimate_model, rolling_zscore, select_models


class DataAnalyzer(IAnalyzer):
    def __init__(self, alpha=0.05):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be within (0, 1).")
        self.alpha = alpha

    def check_stationarity(self, series):
        values = np.asarray(series, dtype=float)
        if values.ndim != 1 or len(values) < 12 or not np.isfinite(values).all():
            raise ValueError("Stationarity requires at least 12 finite observations.")
        p = float(adfuller(values, autolag="AIC")[1]) if np.ptp(values) else 0.0
        return p, p < self.alpha

    def engle_granger_test(self, series1, series2):
        model = estimate_model(
            series1,
            series2,
            fitted_until=datetime(1970, 1, 1, tzinfo=UTC),
            significance=self.alpha,
        )
        return {
            "alpha": model.alpha,
            "beta": model.beta,
            "adf_statistic": model.statistic,
            "p_value": model.p_value,
            "is_cointegrated": model.eligible,
            "reason": model.reason,
        }

    def check_cointegration(self, series1, series2):
        return self.engle_granger_test(series1, series2)

    def calculate_zscore(self, series1, series2, beta, lookback, alpha=0.0):
        return rolling_zscore(
            pd.Series(series1), pd.Series(series2), beta, lookback, alpha
        )

    def find_cointegrated_pairs(self, df):
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
            raise ValueError(
                "Pair selection requires timestamp-aligned formation prices in UTC."
            )
        histories = {}
        for left, right in combinations(df.columns, 2):
            paired = df[[left, right]].dropna()
            histories[(left, right)] = tuple(
                paired[col].to_numpy() for col in (left, right)
            )
        models = select_models(
            histories,
            fitted_until=df.index.max().to_pydatetime(),
            significance=self.alpha,
        )
        return [
            {
                "pair": pair,
                "results": {**model.to_dict(), "is_cointegrated": model.eligible},
            }
            for pair, model in models.items()
            if model.eligible
        ]

    @staticmethod
    def join_pair(df1, df2, *, timeframe_seconds=3600):
        bars = prepare_pair_bars(df1, df2, timeframe_seconds=timeframe_seconds)
        return pd.DataFrame({"close_1": bars.left.close, "close_2": bars.right.close})
