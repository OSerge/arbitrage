"""Integer signed futures hedges constrained by a gross-notional budget."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

from statarb.domain.events import _coerce_decimal
from statarb.domain.specifications import InstrumentSpec


@dataclass(frozen=True, slots=True)
class PairSize:
    left: Decimal
    right: Decimal
    gross_notional: Decimal
    hedge_error: Decimal
    margin: Decimal


def size_pair(
    *,
    beta,
    direction: int,
    left_price,
    right_price,
    left_spec: InstrumentSpec,
    right_spec: InstrumentSpec,
    budget,
    max_hedge_error,
    max_abs_position,
    available_collateral=None,
    left_costs=None,
    right_costs=None,
) -> PairSize | None:
    beta = _coerce_decimal(beta, "beta")
    left_price, right_price = (
        _coerce_decimal(left_price, "left_price"),
        _coerce_decimal(right_price, "right_price"),
    )
    budget = _coerce_decimal(budget, "pair_budget")
    tolerance = _coerce_decimal(max_hedge_error, "max_hedge_error")
    max_position = _coerce_decimal(max_abs_position, "max_abs_position")
    if (
        direction not in (-1, 1)
        or beta == 0
        or min(left_price, right_price, budget, max_position) <= 0
    ):
        raise ValueError(
            "Sizing requires direction +/-1, nonzero beta and positive prices/budget/limit."
        )
    if not 0 <= tolerance < 1:
        raise ValueError("max_hedge_error must be within [0, 1).")
    collateral = (
        _coerce_decimal(available_collateral, "available_collateral")
        if available_collateral is not None
        else None
    )
    if collateral is not None and (left_costs is None or right_costs is None):
        raise ValueError(
            "Collateral-constrained sizing requires explicit costs for both legs."
        )
    left_unit = left_price * left_spec.point_value
    right_unit = right_price * right_spec.point_value
    ratio = beta * left_spec.point_value / right_spec.point_value
    candidates = []
    for quantity in range(1, min(int(budget / left_unit), int(max_position)) + 1):
        q1 = Decimal(quantity * direction)
        ideal = -ratio * q1
        q2 = ideal.to_integral_value(rounding=ROUND_HALF_UP)
        if q2 == 0 or abs(q2) > max_position:
            continue
        gross = abs(q1) * left_unit + abs(q2) * right_unit
        error = abs(q2 - ideal) / abs(ideal)
        if gross <= budget and error <= tolerance:
            margin = left_spec.margin(q1) + right_spec.margin(q2)
            if collateral is not None:
                entry_cost = Decimal("0")
                for quantity_, reference, spec, costs in (
                    (q1, left_price, left_spec, left_costs),
                    (q2, right_price, right_spec, right_costs),
                ):
                    execution = costs.execution_price(
                        reference, 1 if quantity_ > 0 else -1, spec
                    )
                    entry_cost += (
                        costs.commission(quantity_, execution, spec)
                        + abs(execution - reference) * abs(quantity_) * spec.point_value
                    )
                if margin + entry_cost > collateral:
                    continue
            candidates.append(PairSize(q1, q2, gross, error, margin))
    return (
        min(
            candidates,
            key=lambda size: (size.hedge_error, -size.gross_notional, abs(size.left)),
        )
        if candidates
        else None
    )
