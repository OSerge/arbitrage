"""Past-only Engle–Granger estimation and auditable model snapshots."""

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Mapping
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, coint

from statarb.domain.events import SerializableContract, _require_aware_datetime
from statarb.research.config import StrategyConfig


@dataclass(frozen=True, slots=True)
class ModelSnapshot(SerializableContract):
    fitted_until: datetime
    observation_count: int
    alpha: float = 0.0
    beta: float = 0.0
    p_value: float = 1.0
    adjusted_p_value: float = 1.0
    eligible: bool = False
    reason: str = "not estimated"
    statistic: float | None = None

    def __post_init__(self):
        if not isinstance(self.eligible, (bool, np.bool_)):
            raise ValueError("Model eligibility must be a boolean.")
        object.__setattr__(self, "eligible", bool(self.eligible))
        object.__setattr__(
            self,
            "fitted_until",
            _require_aware_datetime(self.fitted_until, "fitted_until"),
        )
        if not isinstance(self.observation_count, int) or self.observation_count < 0:
            raise ValueError("Model observation count must be a non-negative integer.")
        if not np.isfinite(
            [self.alpha, self.beta, self.p_value, self.adjusted_p_value]
        ).all():
            raise ValueError("Model coefficients and p-values must be finite.")
        if not 0 <= self.p_value <= 1 or not 0 <= self.adjusted_p_value <= 1:
            raise ValueError("Model p-values must be within [0, 1].")
        if self.eligible and abs(self.beta) < 1e-12:
            raise ValueError("An eligible hedge requires nonzero beta.")


def estimate_model(
    left, right, *, fitted_until: datetime, significance: float = 0.05
) -> ModelSnapshot:
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    base = ModelSnapshot(fitted_until, len(x))
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 12:
        return replace(base, reason="insufficient aligned formation history")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return replace(base, reason="non-finite formation prices")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return replace(base, reason="constant formation prices")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # AEG requires I(1) inputs. Pretests are diagnostics/eligibility checks,
            # not a proof of the integration order.
            level_p = (adfuller(x, autolag="AIC")[1], adfuller(y, autolag="AIC")[1])
            diff_p = (
                adfuller(np.diff(x), autolag="AIC")[1],
                adfuller(np.diff(y), autolag="AIC")[1],
            )
            if any(p < significance for p in level_p) or any(
                p >= significance for p in diff_p
            ):
                return replace(base, reason="I(1) pretests did not support eligibility")
            statistic, p_value, _ = coint(
                x, y, trend="c", autolag="aic", return_results=False
            )
            alpha, beta = np.linalg.lstsq(
                np.column_stack((np.ones(len(y)), y)), x, rcond=None
            )[0]
        if (
            not np.isfinite([alpha, beta, p_value, statistic]).all()
            or abs(beta) < 1e-12
        ):
            return replace(base, reason="degenerate cointegration estimate")
        return replace(
            base,
            alpha=float(alpha),
            beta=float(beta),
            p_value=float(p_value),
            adjusted_p_value=float(p_value),
            eligible=bool(p_value < significance),
            reason="eligible" if p_value < significance else "no cointegration",
            statistic=float(statistic),
        )
    except (ValueError, np.linalg.LinAlgError, RuntimeWarning) as exc:
        return replace(base, reason=f"estimation failed: {type(exc).__name__}")


def select_models(
    histories: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    *,
    fitted_until: datetime,
    significance: float = 0.05,
) -> dict[tuple[str, str], ModelSnapshot]:
    """Apply Holm to the whole candidate family, including ineligible candidates."""
    results = {
        pair: estimate_model(
            *series, fitted_until=fitted_until, significance=significance
        )
        for pair, series in sorted(histories.items())
    }
    if not results:
        return {}
    adjusted = multipletests(
        [result.p_value for result in results.values()],
        alpha=significance,
        method="holm",
    )[1]
    return {
        pair: replace(
            result,
            adjusted_p_value=float(p),
            eligible=result.eligible and p < significance,
            reason=result.reason
            if not result.eligible or p < significance
            else "Holm rejected eligibility",
        )
        for (pair, result), p in zip(results.items(), adjusted, strict=True)
    }


def current_zscore(
    left, right, index: int, model: ModelSnapshot, lookback: int
) -> float:
    if index < lookback:
        return float("nan")
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    residuals = (
        x[index - lookback : index]
        - model.beta * y[index - lookback : index]
        - model.alpha
    )
    std = np.std(residuals, ddof=1)
    if not np.isfinite(std) or std <= np.finfo(float).eps * max(
        1.0, np.max(np.abs(residuals))
    ):
        return float("nan")
    return float(
        (x[index] - model.beta * y[index] - model.alpha - np.mean(residuals)) / std
    )


def rolling_zscore(
    left: pd.Series, right: pd.Series, beta: float, lookback: int, alpha: float = 0.0
) -> pd.Series:
    residuals = left - beta * right - alpha
    history = residuals.shift(1)
    std = history.rolling(lookback).std(ddof=1).replace(0, np.nan)
    return (residuals - history.rolling(lookback).mean()) / std


def signal_for_zscore(zscore: float, position: int, config: StrategyConfig) -> int:
    """Close on directional crossing; a close can never reverse in the same decision."""
    if not np.isfinite(zscore):
        return position
    if position > 0:
        return 0 if zscore >= -config.exit_threshold else position
    if position < 0:
        return 0 if zscore <= config.exit_threshold else position
    if zscore >= config.entry_threshold:
        return -1
    if zscore <= -config.entry_threshold:
        return 1
    return 0
