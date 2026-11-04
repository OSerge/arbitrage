"""Research state is confirmed by executions, never by an unfilled signal."""

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Mapping

import numpy as np

from statarb.data.bars import AlignedBars
from statarb.domain.events import SerializableContract
from statarb.research.config import StrategyConfig
from statarb.research.models import (
    ModelSnapshot,
    current_zscore,
    estimate_model,
    signal_for_zscore,
)


@dataclass(frozen=True, slots=True)
class PairDecision(SerializableContract):
    index: int
    generated_at: datetime
    position: int
    signal: int
    zscore: float | None
    model: ModelSnapshot | None
    reason: str


class PairResearchState:
    def __init__(
        self,
        config: StrategyConfig,
        *,
        estimator: Callable = estimate_model,
        model_schedule: Mapping[datetime, ModelSnapshot] | None = None,
    ):
        self.config, self.estimator, self.model_schedule = (
            config,
            estimator,
            model_schedule,
        )
        self.position = 0
        self.entry_model = None
        self.latest_model = None
        self.models = []

    def confirm(self, signal: int, model: ModelSnapshot | None):
        if signal not in (-1, 0, 1):
            raise ValueError("Pair direction must be -1, 0 or 1.")
        if self.position and signal and self.position != signal:
            raise ValueError("Close must execute before an opposite entry.")
        if signal == 0:
            self.entry_model = None
        elif self.position == 0:
            if model is None or not model.eligible:
                raise ValueError("Entry requires an eligible historical model.")
            self.entry_model = model
        self.position = signal

    def observe(self, bars: AlignedBars, index: int) -> PairDecision:
        config = self.config
        timestamp = bars.end_at(index)
        if index < config.min_estimation_window:
            return PairDecision(
                index, timestamp, self.position, self.position, None, None, "formation"
            )
        if (index - config.min_estimation_window) % config.recompute_frequency == 0:
            fitted_until = bars.end_at(index - 1)
            if fitted_until > bars.index[index].to_pydatetime():
                raise ValueError(
                    "Formation observations must be completed before the decision bar."
                )
            if self.model_schedule is not None:
                model = self.model_schedule[bars.index[index].to_pydatetime()]
            else:
                model = self.estimator(
                    bars.left.close.to_numpy()[:index],
                    bars.right.close.to_numpy()[:index],
                    fitted_until=fitted_until,
                    significance=config.alpha,
                )
            if model.fitted_until > fitted_until or model.observation_count > index:
                raise ValueError(
                    "A model snapshot contains future formation observations."
                )
            self.latest_model = model
            self.models.append(model)
        model = self.entry_model if self.position else self.latest_model
        if self.latest_model is None or not self.latest_model.eligible:
            return PairDecision(
                index, timestamp, self.position, 0, None, model, "model ineligible"
            )
        zscore = current_zscore(
            bars.left.close.to_numpy(),
            bars.right.close.to_numpy(),
            index,
            model,
            config.lookback,
        )
        signal = signal_for_zscore(zscore, self.position, config)
        if signal and not self.position:
            # Volume of the execution bar is unknown at its open. Only previously
            # observed formation volumes can qualify an entry.
            volumes = (
                frame.volume.iloc[index - config.lookback : index].mean()
                for frame in (bars.left, bars.right)
            )
            if any(volume < config.min_avg_volume for volume in volumes):
                return PairDecision(
                    index,
                    timestamp,
                    self.position,
                    0,
                    float(zscore) if np.isfinite(zscore) else None,
                    model,
                    "insufficient historical liquidity",
                )
        reason = (
            "hold"
            if signal == self.position
            else ("exit crossing" if not signal else "entry threshold")
        )
        if not np.isfinite(zscore):
            reason = "undefined zscore; retain observed position"
        return PairDecision(
            index,
            timestamp,
            self.position,
            signal,
            float(zscore) if np.isfinite(zscore) else None,
            model,
            reason,
        )


def run_signal_research(
    bars: AlignedBars, config: StrategyConfig, *, estimator=estimate_model
):
    """Non-financial diagnostic trace; assumes next-bar signal confirmations."""
    state = PairResearchState(config, estimator=estimator)
    decisions = []
    for index in range(len(bars)):
        if decisions and decisions[-1].signal != state.position:
            state.confirm(decisions[-1].signal, decisions[-1].model)
        decisions.append(state.observe(bars, index))
    return tuple(decisions), tuple(state.models)
