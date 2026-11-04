"""A causal futures backtest driven through the shared paper execution engine."""

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Callable, Mapping
import hashlib
from uuid import uuid4

import pandas as pd

from statarb.data.bars import prepare_pair_bars
from statarb.domain import (
    AccountSummary,
    InstrumentRef,
    InstrumentType,
    TargetPosition,
    ResearchRun,
    ResearchRunStatus,
    SignalIntent,
    SignalDirection,
)
from statarb.domain.events import (
    _coerce_decimal,
    _serialize_value,
    _require_non_empty,
    _require_aware_datetime,
)
from statarb.domain.specifications import InstrumentCatalog
from statarb.research.config import ExecutionCosts, StrategyConfig
from statarb.research.metrics import (
    BootstrapConfig,
    block_bootstrap,
    daily_equity,
    equity_metrics,
)
from statarb.research.models import ModelSnapshot, estimate_model, select_models
from statarb.research.sizing import size_pair
from statarb.research.strategy import PairDecision, PairResearchState
from statarb.runtime.paper.engine import PaperExecutionEngine
from statarb.runtime.paper.event_sink import JsonlEventSink
from statarb.runtime.paper.ledger import PaperLedger
from statarb.runtime.paper.risk_checks import PaperRiskPolicy
from statarb.runtime.replay.loader import EventJournalRecord


@dataclass(frozen=True)
class BacktestRequest:
    left_instrument: InstrumentRef
    right_instrument: InstrumentRef
    left_bars: pd.DataFrame = field(repr=False, compare=False)
    right_bars: pd.DataFrame = field(repr=False, compare=False)
    instrument_catalog: InstrumentCatalog
    initial_cash: Decimal
    pair_budget: Decimal
    max_hedge_error: Decimal
    costs: ExecutionCosts | Mapping[InstrumentRef, ExecutionCosts]
    risk_policy: PaperRiskPolicy
    timeframe_seconds: int = 3600
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    from_time: datetime | None = None
    to_time: datetime | None = None
    tail_rows: int | None = None
    model_schedule: Mapping[datetime, ModelSnapshot] | None = field(
        default=None, repr=False
    )
    session_id: str = field(default_factory=lambda: uuid4().hex)
    bootstrap: BootstrapConfig | None = None
    trading_start: datetime | None = None
    trading_calendar: Mapping[datetime, str] | None = None
    available_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        if self.left_instrument == self.right_instrument:
            raise ValueError("A pair requires distinct instruments.")
        if any(
            inst.instrument_type is not InstrumentType.FUTURE
            for inst in self.instruments
        ):
            raise ValueError("Financial pair backtests currently require RUB futures.")
        if any(inst.exchange != "MOEX" for inst in self.instruments):
            raise ValueError("This financial model supports MOEX futures only.")
        if not isinstance(self.instrument_catalog, InstrumentCatalog):
            raise ValueError(
                "Financial backtests require an explicit historical InstrumentCatalog."
            )
        if not isinstance(self.strategy, StrategyConfig) or not isinstance(
            self.risk_policy, PaperRiskPolicy
        ):
            raise ValueError(
                "Explicit StrategyConfig and PaperRiskPolicy instances are required."
            )
        object.__setattr__(
            self, "session_id", _require_non_empty(self.session_id, "session_id")
        )
        object.__setattr__(
            self,
            "available_at",
            _require_aware_datetime(self.available_at, "available_at"),
        )
        for name in ("initial_cash", "pair_budget", "max_hedge_error"):
            object.__setattr__(self, name, _coerce_decimal(getattr(self, name), name))
        if (
            min(self.initial_cash, self.pair_budget) <= 0
            or not 0 <= self.max_hedge_error < 1
        ):
            raise ValueError(
                "Positive capital/budget and max_hedge_error within [0, 1) are required."
            )
        for inst in self.instruments:
            if not isinstance(self.cost_for(inst), ExecutionCosts):
                raise ValueError("Explicit ExecutionCosts are required for each leg.")

    @property
    def instruments(self):
        return self.left_instrument, self.right_instrument

    def cost_for(self, instrument):
        return (
            self.costs
            if isinstance(self.costs, ExecutionCosts)
            else self.costs[instrument]
        )

    def prepare(self):
        return prepare_pair_bars(
            self.left_bars,
            self.right_bars,
            timeframe_seconds=self.timeframe_seconds,
            from_time=self.from_time,
            to_time=self.to_time,
            tail_rows=self.tail_rows,
            available_at=self.available_at,
        )


@dataclass(frozen=True)
class BacktestResult:
    pair: tuple[str, str]
    models: tuple[ModelSnapshot, ...]
    decisions: tuple[PairDecision, ...]
    equity: pd.Series
    performance: dict
    diagnostics: tuple[str, ...]
    journal_records: tuple[EventJournalRecord, ...]
    position_snapshots: Mapping
    account_summary: AccountSummary
    costs: dict
    estimated_liquidation_cost: Decimal
    status: str
    risk_analysis: dict
    assumptions: dict = field(default_factory=dict)
    signal_intents: tuple[SignalIntent, ...] = ()
    research_run: ResearchRun | None = None

    def to_dict(self):
        return _serialize_value(
            {
                "schema_version": 2,
                "pair": self.pair,
                "status": self.status,
                "models": self.models,
                "decisions": self.decisions,
                "assumptions": self.assumptions,
                "signal_intents": self.signal_intents,
                "research_run": self.research_run,
                "equity": [
                    {"at": at.to_pydatetime(), "equity": str(value)}
                    for at, value in self.equity.items()
                ],
                "performance": self.performance,
                "diagnostics": self.diagnostics,
                "position_snapshots": [
                    snapshot.to_dict() for snapshot in self.position_snapshots.values()
                ],
                "account_summary": self.account_summary,
                "costs": self.costs,
                "estimated_liquidation_cost": self.estimated_liquidation_cost,
                "risk_analysis": {
                    key: (
                        {
                            name: value.tolist() if hasattr(value, "tolist") else value
                            for name, value in item.items()
                        }
                        if isinstance(item, dict)
                        else item
                    )
                    for key, item in self.risk_analysis.items()
                },
                "journal": [
                    {
                        "event_type": record.event_type,
                        "payload": record.payload,
                        "lineage": record.lineage,
                    }
                    for record in self.journal_records
                ],
            }
        )


def run_backtest(
    request: BacktestRequest, *, estimator: Callable = estimate_model, journal_path=None
) -> BacktestResult:
    bars = request.prepare()
    config = request.strategy
    if len(bars) <= config.min_estimation_window:
        raise ValueError(
            "Insufficient completed common bars after range filtering for formation and evaluation."
        )
    # Validate economics for the entire slice before recording any execution.
    for inst in request.instruments:
        point_values = set()
        frame = bars.left if inst == request.left_instrument else bars.right
        for index in range(len(bars)):
            spec = request.instrument_catalog.resolve(
                inst, bars.index[index].to_pydatetime()
            )
            closing_spec = request.instrument_catalog.resolve(inst, bars.end_at(index))
            point_values.update((spec.point_value, closing_spec.point_value))
            if any(
                Decimal(str(frame.iloc[index][name])) % spec.tick_size
                for name in ("open", "high", "low", "close")
            ):
                raise ValueError(
                    f"OHLC prices for {inst.symbol} do not align with the historical tick size."
                )
            if bars.end_at(index) > spec.last_trade_at:
                raise ValueError(
                    "Backtest range extends beyond a contract's last trading time; split the contract slice."
                )
        if len(point_values) > 1:
            raise ValueError(
                "Variable point values within a backtest require a separate settlement model."
            )
    evaluation_index = config.min_estimation_window
    if request.trading_start is not None:
        from statarb.domain.events import _require_aware_datetime

        start = _require_aware_datetime(request.trading_start, "trading_start")
        while evaluation_index < len(bars) and bars.index[evaluation_index] < start:
            evaluation_index += 1
        if evaluation_index >= len(bars):
            raise ValueError("No evaluation bars after the formation selection cutoff.")
    trading_days = None
    if request.trading_calendar is not None:
        try:
            trading_days = [
                request.trading_calendar[at.to_pydatetime()]
                for at in bars.index[evaluation_index:]
            ]
        except KeyError as exc:
            raise ValueError(
                "Trading calendar does not cover all evaluation bars."
            ) from exc
        # Validate labels before journal creation.
        daily_equity(
            pd.Series([1.0] * len(trading_days), index=bars.index[evaluation_index:]),
            trading_days=trading_days,
        )
    ledger = PaperLedger(
        account_id="backtest-pair",
        starting_cash=request.initial_cash,
        instrument_catalog=request.instrument_catalog,
    )
    engine = PaperExecutionEngine(
        ledger=ledger,
        risk_policy=request.risk_policy,
        event_sink=JsonlEventSink(journal_path) if journal_path else None,
        session_id=request.session_id,
    )
    state = PairResearchState(
        config, estimator=estimator, model_schedule=request.model_schedule
    )
    decisions, values, dates = [], [], []
    intents = []
    diagnostics = list(bars.diagnostics)
    pending = None
    fees = drag = turnover = Decimal("0")
    fill_count = closed_pairs = 0
    instruments = request.instruments
    for index in range(len(bars)):
        opened_at = bars.index[index].to_pydatetime()
        if pending is not None and pending.signal != state.position:
            reference = {
                inst: Decimal(str(frame.open.iloc[index]))
                for inst, frame in zip(
                    instruments, (bars.left, bars.right), strict=True
                )
            }
            specs = {
                inst: request.instrument_catalog.resolve(inst, opened_at, trading=True)
                for inst in instruments
            }
            if pending.signal:
                size = size_pair(
                    beta=str(pending.model.beta),
                    direction=pending.signal,
                    left_price=reference[instruments[0]],
                    right_price=reference[instruments[1]],
                    left_spec=specs[instruments[0]],
                    right_spec=specs[instruments[1]],
                    budget=request.pair_budget,
                    max_hedge_error=request.max_hedge_error,
                    max_abs_position=min(
                        request.risk_policy.max_abs_position,
                        request.risk_policy.max_order_quantity,
                    ),
                    available_collateral=ledger.build_account_summary(
                        summary_id="sizing", captured_at=opened_at
                    ).available_cash,
                    left_costs=request.cost_for(instruments[0]),
                    right_costs=request.cost_for(instruments[1]),
                )
                quantities = (size.left, size.right) if size is not None else None
            else:
                quantities = (Decimal("0"), Decimal("0"))
            if quantities is None:
                diagnostics.append(
                    f"{opened_at.isoformat()}: no integer hedge within budget/error/quantity/collateral limits"
                )
            else:
                targets, prices, commissions = [], {}, {}
                for leg, (inst, quantity) in enumerate(
                    zip(instruments, quantities, strict=True), start=1
                ):
                    delta = quantity - ledger.current_quantity(inst)
                    cost = request.cost_for(inst)
                    prices[inst] = cost.execution_price(
                        reference[inst], 1 if delta >= 0 else -1, specs[inst]
                    )
                    commissions[inst] = cost.commission(
                        delta, prices[inst], specs[inst]
                    )
                    targets.append(
                        TargetPosition(
                            f"{request.session_id}-target-{index}-{leg}",
                            inst,
                            quantity,
                            opened_at,
                            signal_intent_id=f"{request.session_id}-intent-{pending.index}-{leg}",
                            research_run_id=request.session_id,
                            reason=pending.reason,
                        )
                    )
                batch = engine.execute_target_positions(
                    target_positions=targets,
                    fill_prices=prices,
                    commissions=commissions,
                    mark_prices=reference,
                    transaction_id=f"{request.session_id}-execution-{index}",
                )
                if batch.accepted:
                    if state.position and not pending.signal:
                        closed_pairs += 1
                    state.confirm(pending.signal, pending.model)
                    for result in batch.results:
                        for fill in result.fill_events:
                            inst = fill.instrument
                            fees += fill.commission
                            drag += (
                                abs(fill.fill_price - reference[inst])
                                * fill.fill_quantity
                                * specs[inst].point_value
                            )
                            turnover += (
                                fill.fill_quantity
                                * fill.fill_price
                                * specs[inst].point_value
                            )
                            fill_count += 1
                else:
                    reasons = sorted(
                        {
                            check.reason
                            for result in batch.results
                            for check in result.risk_results
                            if not check.passed and check.reason
                        }
                    )
                    diagnostics.append(f"{opened_at.isoformat()}: {'; '.join(reasons)}")
        if index >= evaluation_index:
            summary = engine.mark_to_market(
                prices={
                    inst: Decimal(str(frame.close.iloc[index]))
                    for inst, frame in zip(
                        instruments, (bars.left, bars.right), strict=True
                    )
                },
                occurred_at=bars.end_at(index),
            )
            values.append(summary.equity)
            dates.append(bars.end_at(index))
        decision = state.observe(bars, index)
        if index < evaluation_index:
            decision = replace(
                decision, signal=0, reason="formation/selection before evaluation"
            )
        else:
            if state.position and summary.available_cash < 0:
                decision = replace(
                    decision, signal=0, reason="collateral breach: reduce risk"
                )
            if any(
                decision.generated_at
                >= request.instrument_catalog.resolve(
                    inst, decision.generated_at
                ).last_trade_at
                for inst in instruments
            ):
                decision = replace(
                    decision, signal=0, reason="contract trading horizon ended"
                )
        decisions.append(decision)
        if decision.signal != decision.position:
            theoretical = (
                decision.signal,
                -decision.signal * decision.model.beta if decision.model else 0,
            )
            for leg, (inst, signed) in enumerate(
                zip(instruments, theoretical, strict=True), start=1
            ):
                direction = (
                    SignalDirection.LONG
                    if signed > 0
                    else SignalDirection.SHORT
                    if signed < 0
                    else SignalDirection.FLAT
                )
                intents.append(
                    SignalIntent(
                        f"{request.session_id}-intent-{decision.index}-{leg}",
                        request.session_id,
                        inst,
                        direction,
                        Decimal("1") if decision.signal else Decimal("0"),
                        decision.generated_at,
                        horizon="bar-based",
                        thesis=decision.reason,
                    )
                )
        pending = decision
    if pending.signal != state.position:
        diagnostics.append(
            "Final decision remains unexecuted: there is no next common bar open."
        )
    last_time = bars.end_at(len(bars) - 1)
    latest_snapshot_ids = {
        InstrumentRef(**entry.payload["instrument"]): entry.payload["snapshot_id"]
        for entry in engine.journal
        if entry.event_type == "position_snapshot"
    }
    snapshots = {
        inst: ledger.build_position_snapshot(
            instrument=inst,
            snapshot_id=latest_snapshot_ids[inst],
            captured_at=last_time,
        )
        for inst in instruments
    }
    account = summary
    liquidation_cost = Decimal("0")
    for inst, frame in zip(instruments, (bars.left, bars.right), strict=True):
        quantity = ledger.current_quantity(inst)
        if quantity:
            spec = request.instrument_catalog.resolve(inst, last_time)
            price = Decimal(str(frame.close.iloc[-1]))
            cost = request.cost_for(inst)
            execution = cost.execution_price(price, -1 if quantity > 0 else 1, spec)
            liquidation_cost += abs(execution - price) * abs(
                quantity
            ) * spec.point_value + cost.commission(quantity, execution, spec)
    equity = pd.Series(values, index=pd.DatetimeIndex(dates), name="equity")
    performance = equity_metrics(
        equity.astype(float),
        initial_equity=float(request.initial_cash),
        started_at=bars.index[evaluation_index].to_pydatetime(),
        trading_days=trading_days,
    )
    performance.update(fill_count=fill_count, closed_pairs=closed_pairs)
    risk = {
        key: performance[key]
        for key in ("var_95", "cvar_95", "risk_horizon", "daily_observations")
    }
    if request.bootstrap is not None:
        daily = daily_equity(equity.astype(float), trading_days=trading_days)
        previous = daily.shift(1)
        previous.iloc[0] = float(request.initial_cash)
        risk["bootstrap"] = block_bootstrap(
            (daily / previous - 1).to_numpy(), request.bootstrap
        )
    net = account.equity - request.initial_cash
    result = BacktestResult(
        pair=tuple(inst.symbol for inst in instruments),
        models=tuple(state.models),
        decisions=tuple(decisions),
        equity=equity,
        performance=performance,
        diagnostics=tuple(diagnostics),
        journal_records=tuple(
            EventJournalRecord(entry.event_type, entry.payload, entry.lineage)
            for entry in engine.journal
        ),
        position_snapshots=snapshots,
        account_summary=account,
        costs={
            "commission": fees,
            "execution_drag": drag,
            "turnover": turnover,
            "net_pnl": net,
            "gross_pnl": net + fees + drag,
        },
        estimated_liquidation_cost=liquidation_cost,
        status="completed" if fill_count else "no_trades",
        risk_analysis=risk,
        assumptions={
            "initial_cash": request.initial_cash,
            "pair_budget": request.pair_budget,
            "max_hedge_error": request.max_hedge_error,
            "strategy": config,
            "costs": {inst.symbol: request.cost_for(inst) for inst in instruments},
            "risk_policy": request.risk_policy,
            "instrument_specifications": request.instrument_catalog.specs,
            "calendar": performance["calendar"],
            "timeframe_seconds": bars.timeframe_seconds,
            "range_start": bars.index[0].to_pydatetime(),
            "range_end": last_time,
            "evaluation_start": bars.index[evaluation_index].to_pydatetime(),
            "data_sha256": {
                inst.symbol: hashlib.sha256(
                    frame.to_csv(float_format="%.17g").encode()
                ).hexdigest()
                for inst, frame in zip(
                    instruments, (bars.left, bars.right), strict=True
                )
            },
            "pandas_version": pd.__version__,
            "session_id": request.session_id,
            "available_at": request.available_at,
            "trading_calendar": {
                at.isoformat(): day for at, day in request.trading_calendar.items()
            }
            if request.trading_calendar is not None
            else None,
        },
        signal_intents=tuple(intents),
    )
    # Market-time simulation metadata is distinct from operator wall-clock time.
    run = ResearchRun(
        run_id=request.session_id,
        strategy_id="causal-futures-pairs",
        dataset_id="sha256:" + "+".join(result.assumptions["data_sha256"].values()),
        status=ResearchRunStatus.SUCCEEDED,
        started_at=bars.index[0].to_pydatetime(),
        completed_at=last_time,
        parameters=result.assumptions,
        signal_count=len(intents),
    )
    return replace(result, research_run=run)


def run_universe(
    requests: Mapping[tuple[str, str], BacktestRequest],
) -> dict[tuple[str, str], BacktestResult]:
    """Independent pair accounts; Holm selection is shared, capital is not pooled."""
    bars = {pair: request.prepare() for pair, request in requests.items()}
    schedules = {pair: {} for pair in requests}
    timestamps = sorted(
        {
            frame.index[index].to_pydatetime()
            for pair, frame in bars.items()
            for index in range(
                requests[pair].strategy.min_estimation_window,
                len(frame),
                requests[pair].strategy.recompute_frequency,
            )
        }
    )
    if len({request.strategy.alpha for request in requests.values()}) > 1:
        raise ValueError("A selection family must share one significance level.")
    for timestamp in timestamps:
        histories = {}
        for pair, frame in bars.items():
            mask = (
                frame.index + pd.Timedelta(seconds=frame.timeframe_seconds) <= timestamp
            )
            if mask.sum() >= requests[pair].strategy.min_estimation_window:
                histories[pair] = (
                    frame.left.close.loc[mask].to_numpy(),
                    frame.right.close.loc[mask].to_numpy(),
                )
        models = select_models(
            histories,
            fitted_until=timestamp,
            significance=next(iter(requests.values())).strategy.alpha,
        )
        for pair, frame in bars.items():
            if timestamp in frame.index:
                index = frame.index.get_loc(timestamp)
                config = requests[pair].strategy
                if (
                    index >= config.min_estimation_window
                    and (index - config.min_estimation_window)
                    % config.recompute_frequency
                    == 0
                ):
                    # Selection shares a wall-clock cutoff, but this pair's
                    # latest observation may precede it across session gaps.
                    schedules[pair][timestamp] = replace(
                        models[pair], fitted_until=frame.end_at(index - 1)
                    )
    return {
        pair: run_backtest(replace(request, model_schedule=schedules[pair]))
        for pair, request in requests.items()
    }
