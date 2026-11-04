"""Effective-dated instrument economics; identity remains InstrumentRef."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path
from typing import Iterable

from statarb.domain.events import (
    SerializableContract,
    _coerce_decimal,
    _require_aware_datetime,
    _require_non_empty,
)
from statarb.domain.instruments import InstrumentRef, InstrumentType


@dataclass(frozen=True, slots=True)
class InstrumentSpec(SerializableContract):
    instrument: InstrumentRef
    version: str
    source: str
    valid_from: datetime
    tick_size: Decimal
    tick_value: Decimal
    contract_size: Decimal
    initial_margin_long: Decimal
    initial_margin_short: Decimal
    last_trade_at: datetime
    currency: str = "RUB"
    valid_until: datetime | None = None

    def __post_init__(self):
        if (
            not isinstance(self.instrument, InstrumentRef)
            or self.instrument.instrument_type is not InstrumentType.FUTURE
        ):
            raise ValueError("InstrumentSpec currently supports futures only.")
        for name in ("version", "source"):
            object.__setattr__(
                self, name, _require_non_empty(getattr(self, name), name)
            )
        for name in ("valid_from", "last_trade_at"):
            object.__setattr__(
                self, name, _require_aware_datetime(getattr(self, name), name)
            )
        if self.valid_until is not None:
            object.__setattr__(
                self,
                "valid_until",
                _require_aware_datetime(self.valid_until, "valid_until"),
            )
            if self.valid_until <= self.valid_from:
                raise ValueError("Specification valid_until must follow valid_from.")
        if self.last_trade_at <= self.valid_from:
            raise ValueError("last_trade_at must follow valid_from.")
        if self.currency != "RUB":
            raise ValueError("Only RUB-settled futures are supported.")
        for name in (
            "tick_size",
            "tick_value",
            "contract_size",
            "initial_margin_long",
            "initial_margin_short",
        ):
            value = _coerce_decimal(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, value)

    @property
    def point_value(self) -> Decimal:
        # tick_value already includes contract size; never multiply it a second time.
        return self.tick_value / self.tick_size

    def margin(self, quantity: Decimal) -> Decimal:
        return abs(quantity) * (
            self.initial_margin_long if quantity >= 0 else self.initial_margin_short
        )


class InstrumentCatalog:
    def __init__(self, specs: Iterable[InstrumentSpec] = ()):
        self.specs = tuple(specs)
        if not all(isinstance(spec, InstrumentSpec) for spec in self.specs):
            raise ValueError("InstrumentCatalog requires typed InstrumentSpec entries.")
        for i, spec in enumerate(self.specs):
            for other in self.specs[i + 1 :]:
                if spec.instrument != other.instrument:
                    continue
                if (
                    spec.valid_until is None or other.valid_from < spec.valid_until
                ) and (
                    other.valid_until is None or spec.valid_from < other.valid_until
                ):
                    raise ValueError(
                        f"Overlapping specification versions for {spec.instrument.symbol}."
                    )

    def resolve(
        self, instrument: InstrumentRef, as_of: datetime, *, trading: bool = False
    ) -> InstrumentSpec:
        timestamp = _require_aware_datetime(as_of, "as_of")
        for spec in self.specs:
            if (
                spec.instrument == instrument
                and spec.valid_from <= timestamp
                and (spec.valid_until is None or timestamp < spec.valid_until)
            ):
                if trading and timestamp >= spec.last_trade_at:
                    raise ValueError(
                        f"Instrument {instrument.symbol} is past last_trade_at."
                    )
                return spec
        raise ValueError(
            f"Missing historical InstrumentSpec for {instrument.symbol} at {timestamp.isoformat()}."
        )

    @classmethod
    def load(cls, path: str | Path) -> "InstrumentCatalog":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        specs = []
        for row in payload["instruments"]:
            item = dict(row)
            item["instrument"] = InstrumentRef(**item["instrument"])
            for field in ("valid_from", "valid_until", "last_trade_at"):
                if item.get(field) is not None:
                    item[field] = datetime.fromisoformat(item[field])
            specs.append(InstrumentSpec(**item))
        return cls(specs)
