"""Paper runtime primitives for the agent-operated MVP."""

from statarb.runtime.paper.engine import PaperExecutionEngine, PaperExecutionResult
from statarb.runtime.paper.event_sink import EventJournalEntry, JsonlEventSink
from statarb.runtime.paper.ledger import PaperLedger
from statarb.runtime.paper.risk_checks import (
    PaperRiskPolicy,
    RiskCheckResult,
    evaluate_paper_risk_checks,
)

__all__ = [
    "EventJournalEntry",
    "JsonlEventSink",
    "PaperExecutionEngine",
    "PaperExecutionResult",
    "PaperLedger",
    "PaperRiskPolicy",
    "RiskCheckResult",
    "evaluate_paper_risk_checks",
]
