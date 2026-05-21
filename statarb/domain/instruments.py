from __future__ import annotations

"""Canonical instrument contracts."""

from dataclasses import dataclass
from enum import StrEnum

from statarb.domain.events import SerializableContract, _require_non_empty


class InstrumentType(StrEnum):
    EQUITY = "equity"
    FUTURE = "future"
    CURRENCY = "currency"
    INDEX = "index"
    ETF = "etf"

    @classmethod
    def coerce(cls, value: "InstrumentType | str") -> "InstrumentType":
        if isinstance(value, cls):
            return value

        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member

        allowed = ", ".join(member.value for member in cls)
        raise ValueError(f"Unsupported instrument type: {value!r}. Expected one of: {allowed}.")


@dataclass(frozen=True, slots=True)
class InstrumentRef(SerializableContract):
    symbol: str
    exchange: str
    instrument_type: InstrumentType

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _require_non_empty(self.symbol, "symbol"))
        object.__setattr__(self, "exchange", _require_non_empty(self.exchange, "exchange"))
        object.__setattr__(self, "instrument_type", InstrumentType.coerce(self.instrument_type))


__all__ = ["InstrumentRef", "InstrumentType"]
