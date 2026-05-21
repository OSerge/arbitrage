from __future__ import annotations

"""Minimal pre-trade checks for the paper runtime."""

from dataclasses import dataclass
from decimal import Decimal

from statarb.domain.events import _coerce_decimal
from statarb.domain.positions import TargetPosition


@dataclass(frozen=True, slots=True)
class PaperRiskPolicy:
    max_order_quantity: Decimal = Decimal("100")
    max_abs_position: Decimal = Decimal("100")

    def __post_init__(self) -> None:
        max_order_quantity = _coerce_decimal(self.max_order_quantity, "max_order_quantity")
        max_abs_position = _coerce_decimal(self.max_abs_position, "max_abs_position")
        if max_order_quantity <= Decimal("0"):
            raise ValueError("max_order_quantity must be greater than 0.")
        if max_abs_position <= Decimal("0"):
            raise ValueError("max_abs_position must be greater than 0.")

        object.__setattr__(self, "max_order_quantity", max_order_quantity)
        object.__setattr__(self, "max_abs_position", max_abs_position)


@dataclass(frozen=True, slots=True)
class RiskCheckResult:
    name: str
    passed: bool
    reason: str | None = None


def evaluate_paper_risk_checks(
    *,
    current_quantity: Decimal,
    target_position: TargetPosition,
    order_quantity: Decimal,
    policy: PaperRiskPolicy,
) -> tuple[RiskCheckResult, ...]:
    results = [
        RiskCheckResult(
            name="max_order_quantity",
            passed=order_quantity <= policy.max_order_quantity,
            reason=(
                None
                if order_quantity <= policy.max_order_quantity
                else (
                    "max_order_quantity exceeded: "
                    f"{order_quantity} > {policy.max_order_quantity}"
                )
            ),
        ),
        RiskCheckResult(
            name="max_abs_position",
            passed=abs(target_position.target_quantity) <= policy.max_abs_position,
            reason=(
                None
                if abs(target_position.target_quantity) <= policy.max_abs_position
                else (
                    "max_abs_position exceeded: "
                    f"{abs(target_position.target_quantity)} > {policy.max_abs_position}"
                )
            ),
        ),
    ]

    if order_quantity == Decimal("0"):
        results.append(
            RiskCheckResult(
                name="delta_required",
                passed=True,
                reason=None,
            )
        )

    return tuple(results)


__all__ = ["PaperRiskPolicy", "RiskCheckResult", "evaluate_paper_risk_checks"]
