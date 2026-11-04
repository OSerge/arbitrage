"""Offline financial backtest entry point with explicit, file-backed assumptions."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

import pandas as pd

from statarb.domain import InstrumentCatalog, InstrumentRef
from statarb.research.backtest import BacktestRequest, run_backtest
from statarb.research.config import ExecutionCosts, StrategyConfig
from statarb.research.metrics import BootstrapConfig
from statarb.runtime.paper.risk_checks import PaperRiskPolicy


def load_request(path: str | Path) -> BacktestRequest:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _request_from_payload(path, payload)


def _request_from_payload(path: Path, payload: dict) -> BacktestRequest:
    def resolve(value):
        candidate = Path(value)
        return candidate if candidate.is_absolute() else path.parent / candidate

    def read_bars(value):
        candidate = resolve(value)
        return (
            pd.read_parquet(candidate)
            if candidate.suffix == ".parquet"
            else pd.read_csv(candidate, float_precision="round_trip")
        )

    return BacktestRequest(
        left_instrument=InstrumentRef(**payload["left"]["instrument"]),
        right_instrument=InstrumentRef(**payload["right"]["instrument"]),
        left_bars=read_bars(payload["left"]["bars"]),
        right_bars=read_bars(payload["right"]["bars"]),
        instrument_catalog=InstrumentCatalog.load(resolve(payload["specifications"])),
        initial_cash=payload["initial_cash"],
        pair_budget=payload["pair_budget"],
        max_hedge_error=payload["max_hedge_error"],
        costs=ExecutionCosts(**payload["costs"]),
        risk_policy=PaperRiskPolicy(**payload.get("risk_policy", {})),
        strategy=StrategyConfig(**payload.get("strategy", {})),
        timeframe_seconds=payload.get("timeframe_seconds", 3600),
        from_time=datetime.fromisoformat(payload["from_time"])
        if payload.get("from_time")
        else None,
        to_time=datetime.fromisoformat(payload["to_time"])
        if payload.get("to_time")
        else None,
        tail_rows=payload.get("tail_rows"),
        session_id=payload.get("session_id") or uuid4().hex,
        bootstrap=BootstrapConfig(**payload["bootstrap"])
        if payload.get("bootstrap")
        else None,
        trading_start=datetime.fromisoformat(payload["trading_start"])
        if payload.get("trading_start")
        else None,
        trading_calendar={
            datetime.fromisoformat(at): day
            for at, day in json.loads(
                resolve(payload["trading_calendar"]).read_text()
            ).items()
        }
        if payload.get("trading_calendar")
        else None,
        **(
            {"available_at": datetime.fromisoformat(payload["available_at"])}
            if payload.get("available_at")
            else {}
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run an offline causal RUB futures backtest."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON inputs, historical specifications and explicit assumptions",
    )
    parser.add_argument("--output", required=True, help="Financial result JSON")
    parser.add_argument(
        "--journal", help="New v2 paper journal path for independent replay"
    )
    args = parser.parse_args(argv)
    try:
        if args.journal and Path(args.output).resolve() == Path(args.journal).resolve():
            raise ValueError("Output and journal paths must differ.")
        if Path(args.output).resolve() == Path(args.config).resolve():
            raise ValueError("Output must not overwrite the input configuration.")
        config_path = Path(args.config).resolve()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        input_names = [
            payload["left"]["bars"],
            payload["right"]["bars"],
            payload["specifications"],
        ]
        if payload.get("trading_calendar"):
            input_names.append(payload["trading_calendar"])
        input_files = {config_path} | {
            (config_path.parent / value).resolve() for value in input_names
        }
        for destination in (args.output, args.journal):
            if destination and Path(destination).resolve() in input_files:
                raise ValueError("Outputs must not overwrite research input files.")
        request = _request_from_payload(config_path, payload)
        result = run_backtest(request, journal_path=args.journal)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=output.parent, encoding="utf-8", delete=False
        ) as handle:
            temporary = Path(handle.name)
            try:
                json.dump(
                    result.to_dict(),
                    handle,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    indent=2,
                )
                handle.flush()
                os.fsync(handle.fileno())
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(output)
        print(
            json.dumps(
                {
                    "status": result.status,
                    "output": str(output),
                    "net_pnl": str(result.costs["net_pnl"]),
                    "fills": result.performance["fill_count"],
                }
            )
        )
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Backtest rejected: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
