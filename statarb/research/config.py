"""Explicit strategy, sizing and cost assumptions."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import math

from statarb.domain.events import _coerce_decimal
from statarb.domain.specifications import InstrumentSpec


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    lookback: int = 60
    min_estimation_window: int = 512
    recompute_frequency: int = 16
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    alpha: float = 0.05
    min_avg_volume: float = 200.0

    def __post_init__(self):
        for name in ("lookback", "min_estimation_window", "recompute_frequency"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.lookback < 2 or self.min_estimation_window < max(self.lookback, 12):
            raise ValueError(
                "Formation history must cover lookback and at least 12 observations."
            )
        for name in ("entry_threshold", "exit_threshold", "alpha", "min_avg_volume"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        if not 0 <= self.exit_threshold < self.entry_threshold:
            raise ValueError("Require 0 <= exit_threshold < entry_threshold.")
        if not 0 < self.alpha < 1 or self.min_avg_volume < 0:
            raise ValueError("Invalid significance or minimum volume.")


@dataclass(frozen=True, slots=True)
class ExecutionCosts:
    broker_per_contract: Decimal
    exchange_per_contract: Decimal
    spread_bps: Decimal
    slippage_bps: Decimal
    broker_vat_rate: Decimal = Decimal("0")
    notional_fee_bps: Decimal = Decimal("0")

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = _coerce_decimal(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, value)
        if self.spread_bps / 2 + self.slippage_bps >= 10000:
            raise ValueError(
                "Spread/slippage would produce non-positive execution prices."
            )

    def execution_price(
        self, reference: Decimal, side: int, spec: InstrumentSpec
    ) -> Decimal:
        reference = _coerce_decimal(reference, "reference_price")
        if reference <= 0 or side not in (-1, 1):
            raise ValueError("Execution requires a positive price and side +/-1.")
        adverse = (self.spread_bps / 2 + self.slippage_bps) / 10000
        price = reference * (1 + side * adverse)
        ticks = (price / spec.tick_size).to_integral_value(
            rounding=ROUND_CEILING if side > 0 else ROUND_FLOOR
        )
        result = ticks * spec.tick_size
        if result <= 0:
            raise ValueError(
                "Execution price must remain positive after tick rounding."
            )
        return result

    def commission(
        self, quantity: Decimal, price: Decimal, spec: InstrumentSpec
    ) -> Decimal:
        return abs(quantity) * (
            self.broker_per_contract * (1 + self.broker_vat_rate)
            + self.exchange_per_contract
            + abs(price) * spec.point_value * self.notional_fee_bps / 10000
        )
