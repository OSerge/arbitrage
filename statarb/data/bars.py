"""Validated, time-aligned completed bars shared by research entry points."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from statarb.domain.events import _require_aware_datetime

OHLCV = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class AlignedBars:
    left: pd.DataFrame
    right: pd.DataFrame
    timeframe_seconds: int
    diagnostics: tuple[str, ...] = ()

    def __len__(self):
        return len(self.left)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.left.index

    def end_at(self, index: int) -> datetime:
        return (
            self.index[index] + pd.Timedelta(seconds=self.timeframe_seconds)
        ).to_pydatetime()


def validate_bars(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(OHLCV) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    result = frame.copy(deep=True)
    if "bar_start_utc" in result:
        result.index = pd.DatetimeIndex(
            pd.to_datetime(result.pop("bar_start_utc"), utc=True)
        )
    elif "time" in result:
        result.index = pd.DatetimeIndex(
            pd.to_datetime(result.pop("time"), unit="s", utc=True)
        )
    elif isinstance(result.index, pd.DatetimeIndex) and result.index.tz is not None:
        result.index = result.index.tz_convert("UTC")
    else:
        raise ValueError(
            "Bars require epoch time, bar_start_utc, or a timezone-aware datetime index."
        )
    if result.index.hasnans:
        raise ValueError("Bar timestamps must be finite.")
    result = (
        result.loc[:, list(OHLCV)].apply(pd.to_numeric, errors="raise").sort_index()
    )
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("OHLCV values must be finite.")
    if (result.loc[:, ["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("This RUB futures model requires positive prices.")
    if (result.volume < 0).any():
        raise ValueError("Volume must be non-negative.")
    if (
        (result.high < result[["open", "close", "low"]].max(axis=1))
        | (result.low > result[["open", "close", "high"]].min(axis=1))
    ).any():
        raise ValueError("Inconsistent OHLC bounds.")
    duplicate_rows = result[result.index.duplicated(keep=False)]
    if (
        not duplicate_rows.empty
        and (duplicate_rows.groupby(level=0).nunique() > 1).any().any()
    ):
        raise ValueError("Conflicting duplicate bar timestamps.")
    result = result[~result.index.duplicated(keep="first")]
    result.index.name = "bar_start_utc"
    return result


def prepare_pair_bars(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    timeframe_seconds: int,
    from_time: datetime | None = None,
    to_time: datetime | None = None,
    tail_rows: int | None = None,
    available_at: datetime | None = None,
) -> AlignedBars:
    if (
        isinstance(timeframe_seconds, bool)
        or not isinstance(timeframe_seconds, int)
        or timeframe_seconds <= 0
    ):
        raise ValueError("timeframe_seconds must be a positive integer.")
    start = (
        _require_aware_datetime(from_time, "from_time")
        if from_time is not None
        else None
    )
    end = _require_aware_datetime(to_time, "to_time") if to_time is not None else None
    if start is not None and end is not None and start >= end:
        raise ValueError("from_time must precede to_time.")
    if available_at is not None:
        captured = _require_aware_datetime(available_at, "available_at")
        end = min(end, captured) if end is not None else captured
    frames = [validate_bars(left), validate_bars(right)]
    for i, frame in enumerate(frames):
        if (
            frame.index.to_series().diff().dropna()
            < pd.Timedelta(seconds=timeframe_seconds)
        ).any():
            raise ValueError("Overlapping bars or a mismatched timeframe.")
        if start is not None:
            frame = frame.loc[frame.index >= start]
        if end is not None:
            frame = frame.loc[
                (frame.index < end)
                & (frame.index + pd.Timedelta(seconds=timeframe_seconds) <= end)
            ]
        frames[i] = frame
    common = frames[0].index.intersection(frames[1].index).sort_values()
    diagnostics = tuple(
        f"leg-{i + 1}: excluded {len(frame) - len(common)} bars without a simultaneous other leg"
        for i, frame in enumerate(frames)
        if len(frame) != len(common)
    )
    if tail_rows is not None:
        if (
            isinstance(tail_rows, bool)
            or not isinstance(tail_rows, int)
            or tail_rows <= 0
        ):
            raise ValueError("tail_rows must be a positive integer.")
        common = common[-tail_rows:]
    if common.empty:
        raise ValueError("No simultaneous completed bars in the requested range.")
    return AlignedBars(
        frames[0].loc[common].copy(),
        frames[1].loc[common].copy(),
        timeframe_seconds,
        diagnostics,
    )
