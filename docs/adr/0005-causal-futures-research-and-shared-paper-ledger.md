# ADR 0005: causal futures research and a shared paper ledger

- Status: accepted by the user, 2026-11-04.
- Supersedes the restriction on moving research responsibilities out of `core/` for this migration.

## Decision

Implement the approved audit/remediation plan in `statarb/`. Keep `core/` public entry points as compatibility adapters, without a second financial engine. Research, sizing and execution remain separate responsibilities. No broker command or live enablement is included.

The baseline is raw-price Engle–Granger pairs, integer RUB futures, a gross-notional budget per pair, entry-frozen coefficients and quantities, and closing without immediate reversal. Signals use completed bars and execute at the next common bar's open. Formation observations are excluded from performance. End-of-data valuation is mark-to-market, with estimated liquidation costs reported separately.

Financial runs require explicit capital, budget, hedge-error tolerance, fees, spread/slippage and effective-dated instrument specifications. There is no implicit unit multiplier, historical margin or tax assumption. Existing historical smoke fixtures may use **explicitly labelled synthetic specifications** solely to exercise contracts; their output is not a financial backtest.

## Accounting and persistence

Futures cash changes by realized PnL and fees; equity also includes unrealized PnL. Collateral is the conservative sum of per-leg initial margins. Cash settlement at exchange clearing and automatic contract rolls are outside this slice; free collateral is based on marked equity. Specification point values must stay constant within a run; historical margin may change.

Paper and backtest share fills, risk checks, marks and ledger arithmetic. Paper pair groups are all-or-none simulator transactions, not a claim of atomic broker execution. Validate on a private ledger copy, durably append a checksummed transaction, then publish state. A journal error leaves the published ledger unchanged. Persist price marks so replay never trusts snapshots as pricing inputs.

Journal v2 stores one complete transaction per newline-terminated JSON record. The reader continues to accept v1 events, rejects corrupt committed records and warns about an uncommitted final tail. Replay validates every financial checkpoint and returns reconstructed state. Fill identity is idempotent and conflicting duplicates fail.

## Compatibility

Existing event payload fields and read-model schema v1 remain available. Price marks and transaction envelopes are additive. Financial legacy calls lacking explicit metadata fail with a migration diagnostic. Snapshot smoke helpers remain labelled fixtures, and consume the final causal signal, including flat.

## Verification

Require prefix invariance, timestamp alignment/range checks, hand-calculated round trips and futures marks, integer signed hedges, risk/journal failure atomicity, duplicate/conflicting fill tests, intermediate replay tampering and full historical backtest/paper/replay parity. Run the complete pytest and Ruff checks.
