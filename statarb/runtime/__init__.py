"""Minimal paper/replay runtime shell."""

from statarb.runtime.alor_pair_bootstrap import PairBootstrapResult, run_alor_pair_bootstrap
from statarb.runtime.historical import (
    HistoricalPaperReplaySmokeResult,
    run_historical_paper_replay_smoke,
)
from statarb.runtime.paper import (
    EventJournalEntry,
    JsonlEventSink,
    PaperExecutionEngine,
    PaperExecutionResult,
    PaperLedger,
    PaperRiskPolicy,
    RiskCheckResult,
    evaluate_paper_risk_checks,
)
from statarb.runtime.replay import (
    EventJournalRecord,
    ReplayResult,
    RuntimeReplayer,
    decode_event_record,
    load_event_journal,
)

__all__ = [
    "EventJournalEntry",
    "EventJournalRecord",
    "HistoricalPaperReplaySmokeResult",
    "JsonlEventSink",
    "PairBootstrapResult",
    "PaperExecutionEngine",
    "PaperExecutionResult",
    "PaperLedger",
    "PaperRiskPolicy",
    "ReplayResult",
    "RiskCheckResult",
    "RuntimeReplayer",
    "decode_event_record",
    "evaluate_paper_risk_checks",
    "load_event_journal",
    "run_alor_pair_bootstrap",
    "run_historical_paper_replay_smoke",
]
