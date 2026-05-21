"""Canonical domain contracts for the agent-operated MVP."""

from statarb.domain import events, instruments, orders, positions, research
from statarb.domain.events import EventSource, ExecutionContour, ReplaySession, ReplaySessionStatus
from statarb.domain.instruments import InstrumentRef, InstrumentType
from statarb.domain.orders import FillEvent, OrderEvent, OrderSide, OrderStatus, OrderType
from statarb.domain.positions import AccountSummary, PositionSnapshot, TargetPosition
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
    "InstrumentType",
    "OrderEvent",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PositionSnapshot",
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
