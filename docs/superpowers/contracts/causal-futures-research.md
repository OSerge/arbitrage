# Causal futures research, accounting and journals

Status: implemented under accepted ADR 0005, user approval 2026-11-04.

## Inputs and responsibilities

- `statarb.data.bars`: completed UTC OHLCV, deduplicate identical timestamps, reject conflicting timestamps/nonfinite values/invalid OHLC, align both legs before range/tail selection. Financial execution uses only the intersection; excluded observations are reported. Gaps do not create prices or fills.
- `InstrumentRef`: unchanged identity. `InstrumentSpec` adds effective-dated, sourced economics: version, valid_from/valid_until, tick_size, tick_value, contract_size, long/short initial margin, last_trade_at and RUB settlement currency. Canonical futures quantities are integer contracts. Tick value is already per contract; contract_size is not multiplied a second time.
- `InstrumentCatalog`: rejects overlapping versions and missing historical coverage. A financial run requires constant point value throughout its input slice. Margin may change with time. A current reference snapshot is valid from its observation time and is never silently backfilled into past research.
- `BacktestRequest`: two identified MOEX futures, timestamped bars, timeframe in seconds, instrument catalog, initial_cash, gross pair_budget, relative max_hedge_error, per-leg or common `ExecutionCosts`, `PaperRiskPolicy`, strategy and optional UTC range/trading-start/calendar. No implicit fees, historical ГО or economic multipliers.
- `ModelSnapshot`: coefficients, raw/corrected p-values, eligibility/reason, observation count and fitted_until. The fitted history ends before the current decision bar. Explicit supplied model schedules must obey that boundary. For family selection, each pair retains its actual last formation bar end as fitted_until, including across gaps; the shared selection cutoff is a separate scheduling time.
- `SignalIntent` remains direction/conviction only. Sizing and `TargetPosition` remain downstream of research.

## Strategy and sizing

Baseline: raw prices, expanding training history, 512 initial common observations, trailing 60-observation z-score using `ddof=1`, refit every 16 common observations, entry 2, exit 0.5. ADF pretests check I(1) eligibility; these are not a proof of integration order. Candidate-family selection uses Holm at 0.05 on training observations only, including failed candidates with p=1. Entry liquidity uses prior observed volumes, not the future execution bar's volume.

Orders generated after a closed bar execute at the next common open. Confirm state only after accepted execution. Freeze entry alpha/beta and quantities; calculate subsequent trailing residual statistics with those coefficients. New estimates govern eligibility and future entries. Long-spread exits use z >= -exit; short-spread exits use z <= exit. Closing never reverses immediately. Undefined z holds observed state; invalid price data fail validation. Loss of model eligibility or insufficient marked collateral queues closing. No last-estimate filter can remove earlier PnL.

Let m=tick_value/tick_size. Enumerate integer first-leg quantities, round the signed hedge q2=-beta*q1*m1/m2 using ROUND_HALF_UP, and enforce both-leg quantity/order limits, budget and maximum relative rounding error. Minimize relative hedge error, then maximize used notional. Financial sizing also requires that conservative summed margins plus commissions and adverse tick-rounded fills fit free collateral. An infeasible pair produces a diagnostic, not a fractional hedge.

## Money and valuation

Futures cash = initial cash + realized PnL - fees. Equity includes unrealized PnL at independent price marks. Free collateral = marked equity - summed leg margins; buying power is max(0, free collateral). Negative free collateral remains observable and risk-reducing exits remain possible. Reductions exceeding max order size are split into legal integer orders without raising limits.

Fill costs apply to every changed contract on both legs, on entry and exit. `ExecutionCosts` distinguishes RUB per contract, notional basis points, full spread basis points, slippage basis points and an explicit broker VAT factor (default zero). Buy execution rounds adversely up to tick; sell execution rounds down. Future candle high/low never influence an opening fill.

End-of-data equity marks open positions; outstanding decisions remain unexecuted without another common open. Estimated liquidation costs are separate and are not fabricated fills. No contract roll, delivery, clearing-time cash settlement, cross-margin credit or microstructure/queue model is claimed. Complete bar fills and atomic two-leg groups are simulator assumptions.

## Results and evaluation

`available_at` bounds observable completed bars independently of a later requested `to_time`. Financial requests default to their creation time; historical raw CSVs are assumed to contain finalized closes. Snapshot callers must supply the actual capture cutoff. The Alor manifest bridge subtracts advertised data delay from the manifest creation time. No capture cutoff can repair an already incomplete historical CSV close without its original snapshot metadata.

`BacktestResult` exposes model/decision traces, fills/orders/marks through its journal, equity, positions, actual costs, estimated liquidation costs and diagnostics. JSON schema_version=2 includes all financial assumptions, sourced specification versions and hashes of the normalized input bars. Independent pair accounts are not a pooled portfolio.

Formation is excluded from performance. Initial equity participates in drawdown. CAGR uses elapsed calendar time. Daily Sharpe uses mean excess daily return/std with 252 periods/year and zero annual risk-free rate by default. Group observed bar closes by Europe/Moscow date with an exclusive bar-end boundary. An optional bar-start -> exchange trading-date map handles explicit overnight/session calendars; incomplete maps fail before execution. No missing trading days are invented. The result names its calendar convention.

Historical VaR/ES use one-day losses at 95%, with sample count; fewer than 20 observed days return unavailable tail estimates. Optional block bootstrap requires horizon_days, block_days and seed; simulations default to 1000 and compounded paths include initial equity. Zero variance/insufficient samples yield unavailable Sharpe rather than artificial certainty.

## Paper transactions and replay

Prepare all fills/marks/checkpoints on a private ledger copy. Reject the whole pair if either leg or group collateral fails. Persist a v2 transaction, fsync, then publish the copy. Emit ACCEPTED -> fill -> FILLED; completed orders must not remain open in read models. Persist `PriceMark(event_id, instrument, price, occurred_at, contour)` independently of derived snapshots, including marks without fills.

V2 newline-terminated records contain schema_version, transaction_id, events and a SHA-256 checksum over the transaction id and canonical event list. The file has one cooperating writer; stale writers fail by expected file size under a file lock. A failed append truncates to the previous committed offset and poisons that engine until recovery. Fill ids are idempotent, conflicting duplicates fail. Session-qualified ids avoid restart collisions. A repeated in-process transaction id returns its original result; conflicting requests fail.

The loader accepts legacy v1 event lines, validates every committed line and warns about an incomplete final tail. Reader compatibility does not reinterpret old incorrect futures economics as correct; replay still requires matching explicit specifications and correct financial checkpoints.

Replay rebuilds every checkpoint from fills/marks and validates all position/account financial fields, identity, chronology and order/fill consistency. It returns reconstructed snapshots, not the journal's asserted snapshots. Recover a ledger before appending to an existing journal. Recover an incomplete tail into a new file; preserve the original evidence.

## Legacy and fixtures

`core.Backtester` and `ImprovedBacktester` delegate to the same financial kernel. Raw arrays cannot produce financial returns without a `BacktestRequest`. Old ambiguous commission arguments fail with migration guidance. Direction-only helpers share the crossing/NaN policy and are explicitly non-financial.

Historical snapshot smoke uses the final causal decision, including flat, and prices the final bar. It never searches backwards for an active signal or takes abs(beta). The default plumbing fixture labels its instrument economics SYNTHETIC; it is not a financial backtest or evidence of strategy profitability. External Alor smoke requires a catalog. Full lifecycle accounting is verified separately with hand-calculated and stochastic cointegrated fixtures.
