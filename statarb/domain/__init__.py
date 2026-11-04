"""Canonical domain contracts for the agent-operated MVP."""

from statarb.domain import events, instruments, orders, positions, research
from statarb.domain.events import (
    EventSource,
    ExecutionContour,
    ReplaySession,
    ReplaySessionStatus,
)
from statarb.domain.instruments import InstrumentRef, InstrumentType
from statarb.domain.orders import (
    FillEvent,
    OrderEvent,
    OrderSide,
    OrderStatus,
    OrderType,
)
from statarb.domain.positions import (
    AccountSummary,
    PositionSnapshot,
    PriceMark,
    TargetPosition,
)
from statarb.domain.specifications import InstrumentCatalog, InstrumentSpec
from statarb.domain.research import (
    ResearchRun,
    ResearchRunStatus,
    SignalDirection,
    SignalIntent,
)

__all__ = [
    "AccountSummary",
    "EventSource",
    "ExecutionContour",
    "FillEvent",
    "InstrumentRef",
    "InstrumentSpec",
    "InstrumentCatalog",
    "InstrumentType",
    "OrderEvent",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PositionSnapshot",
    "PriceMark",
    "ReplaySession",
    "ReplaySessionStatus",
    "ResearchRun",
    "ResearchRunStatus",
    "SignalDirection",
    "SignalIntent",
    "TargetPosition",
    "events",
    "instruments",
    "orders",
    "positions",
    "research",
]
