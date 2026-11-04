"""Account-equity performance, daily tail risk and seeded block bootstrap."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import math


def equity_metrics(
    equity: pd.Series,
    *,
    initial_equity: float,
    started_at: datetime,
    risk_free_annual: float = 0.0,
    periods_per_year: int = 252,
    trading_days=None,
) -> dict:
    if not isinstance(equity.index, pd.DatetimeIndex) or equity.index.tz is None:
        raise ValueError("Equity requires timezone-aware timestamps.")
    if equity.empty or initial_equity <= 0 or not np.isfinite(initial_equity):
        raise ValueError(
            "Positive initial equity and nonempty observations are required."
        )
    if (
        not equity.index.is_monotonic_increasing
        or equity.index.has_duplicates
        or not np.isfinite(equity.to_numpy(dtype=float)).all()
    ):
        raise ValueError("Equity observations must be finite, ordered and unique.")
    if (
        periods_per_year <= 0
        or not np.isfinite(risk_free_annual)
        or risk_free_annual <= -1
    ):
        raise ValueError("Invalid annualization assumptions.")
    values = np.r_[initial_equity, equity.to_numpy(dtype=float)]
    running_peak = np.maximum.accumulate(values)
    drawdowns = values / running_peak - 1
    total = values[-1] / initial_equity - 1
    elapsed = (equity.index[-1].to_pydatetime() - started_at).total_seconds()
    if elapsed < 0:
        raise ValueError("started_at must not follow the equity observations.")
    years = elapsed / (365.25 * 86400)
    exponent = math.log1p(total) / years if years > 0 and total > -1 else None
    annual = math.expm1(exponent) if exponent is not None and exponent < 709 else None
    # Retain only observed MOEX local trading dates. Flat observed days remain in
    # the sample; weekends/holidays are not invented by resampling.
    daily_values = daily_equity(equity, trading_days=trading_days)
    previous = daily_values.shift(1)
    previous.iloc[0] = initial_equity
    daily_returns = daily_values / previous - 1
    if (previous <= 0).any():
        daily_returns = daily_returns.loc[previous > 0]
    mean = float(daily_returns.mean()) if len(daily_returns) else 0.0
    std = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else 0.0
    daily_rf = (1 + risk_free_annual) ** (1 / periods_per_year) - 1
    sharpe = (mean - daily_rf) / std * np.sqrt(periods_per_year) if std > 0 else None
    losses = -daily_returns.to_numpy(dtype=float)
    # 20 observations are a minimal empirical 5% tail, not a reliability claim.
    var = float(np.quantile(losses, 0.95)) if len(losses) >= 20 else None
    es = float(losses[losses >= var].mean()) if var is not None else None
    return {
        "total_return": float(total),
        "annual_return": annual,
        "volatility": std * np.sqrt(periods_per_year),
        "sharpe_ratio": float(sharpe) if sharpe is not None else None,
        "max_drawdown": float(drawdowns.min()),
        "daily_observations": len(daily_returns),
        "insolvent": bool(np.min(values) <= 0),
        "var_95": var,
        "cvar_95": es,
        "risk_horizon": "one observed trading day",
        "calendar": "supplied trading-day labels"
        if trading_days is not None
        else "observed Europe/Moscow dates; bar end exclusive",
        "time_info": {
            "total_hours": elapsed / 3600,
            "trading_days": len(daily_values),
            "annual_factor": 1 / years if years > 0 else 0,
            "hours_in_year": None,
        },
    }


def daily_equity(equity: pd.Series, *, trading_days=None):
    if trading_days is None:
        labels = (
            (equity.index - pd.Timedelta(nanoseconds=1))
            .tz_convert("Europe/Moscow")
            .date
        )
    else:
        from datetime import date

        labels = [date.fromisoformat(str(value)) for value in trading_days]
        if len(labels) != len(equity) or labels != sorted(labels):
            raise ValueError("Trading-day labels must cover equity chronologically.")
    return equity.groupby(labels).last()


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    horizon_days: int
    block_days: int
    seed: int
    simulations: int = 1000

    def __post_init__(self):
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (self.horizon_days, self.block_days, self.simulations)
        ):
            raise ValueError(
                "Bootstrap horizon, block size and simulations must be positive integers."
            )
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Bootstrap seed must be an explicit non-negative integer.")


def block_bootstrap(daily_returns, config: BootstrapConfig) -> dict:
    returns = np.asarray(daily_returns, dtype=float)
    if (
        returns.ndim != 1
        or len(returns) < config.block_days
        or not np.isfinite(returns).all()
        or (returns < -1).any()
    ):
        raise ValueError(
            "Insufficient valid daily returns for the requested bootstrap blocks."
        )
    rng = np.random.default_rng(config.seed)
    blocks = (config.horizon_days + config.block_days - 1) // config.block_days
    starts = rng.integers(
        0, len(returns) - config.block_days + 1, (config.simulations, blocks)
    )
    samples = returns[starts[..., None] + np.arange(config.block_days)].reshape(
        config.simulations, -1
    )[:, : config.horizon_days]
    paths = np.concatenate(
        (np.ones((config.simulations, 1)), np.cumprod(1 + samples, axis=1)), axis=1
    )
    losses = 1 - paths[:, -1]
    var = float(np.quantile(losses, 0.95))
    return {
        "var_95": var,
        "cvar_95": float(losses[losses >= var].mean()),
        "max_drawdown_dist": np.min(
            paths / np.maximum.accumulate(paths, axis=1) - 1, axis=1
        ),
        "horizon_days": config.horizon_days,
        "seed": config.seed,
    }
