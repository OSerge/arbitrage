# Domain Execution Contracts

Статус: foundation

## Contract: TargetPosition

- layer: portfolio
- purpose: desired net position after research and portfolio sizing.
- required-fields:
  - `target_position_id`
  - `instrument`
  - `target_quantity`
  - `as_of`
- optional-fields:
  - `signal_intent_id`
  - `research_run_id`
  - `reason`
- invariants:
  - quantity is expressed in canonical instrument units.
  - zero quantity is valid and means explicit flattening.
  - contract does not carry broker execution lifecycle fields.

## Contract: OrderEvent

- layer: execution
- purpose: canonical state transition for an order lifecycle.
- required-fields:
  - `event_id`
  - `order_id`
  - `instrument`
  - `order_type`
  - `side`
  - `quantity`
  - `status`
  - `occurred_at`
  - `contour`
- optional-fields:
  - `limit_price`
  - `broker_order_id`
  - `reason`
  - `target_position_id`
  - `signal_intent_id`
- invariants:
  - `quantity > 0`
  - `limit_price` is required for `limit` orders.
  - `status` captures lifecycle transition, not fill aggregation.

## Contract: FillEvent

- layer: execution
- purpose: canonical execution fill after order acceptance.
- required-fields:
  - `event_id`
  - `fill_id`
  - `order_id`
  - `instrument`
  - `side`
  - `fill_quantity`
  - `fill_price`
  - `occurred_at`
  - `contour`
- optional-fields:
  - `broker_trade_id`
  - `commission`
- invariants:
  - `fill_quantity > 0`
  - `fill_price > 0`
  - fill payload is append-only; corrections require a new event.

## Contract: PositionSnapshot

- layer: execution-state
- purpose: observed net position state after fills, paper simulation, or replay.
- required-fields:
  - `snapshot_id`
  - `instrument`
  - `net_quantity`
  - `captured_at`
  - `contour`
- optional-fields:
  - `average_price`
  - `mark_price`
  - `realized_pnl`
  - `unrealized_pnl`
- invariants:
  - snapshot describes observed state, not desired state.
  - pnl fields are additive accounting outputs and may be zero.

## Contract: AccountSummary

- layer: execution-state
- purpose: minimal account-level snapshot shared by paper, replay, and future broker mapping.
- required-fields:
  - `summary_id`
  - `account_id`
  - `captured_at`
  - `contour`
  - `equity`
  - `available_cash`
- optional-fields:
  - `buying_power`
  - `margin_used`
- invariants:
  - monetary fields are decimal-compatible.
  - summary is an observed state snapshot, not an approval or risk policy.
