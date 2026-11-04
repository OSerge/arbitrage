"""Convert timestamped Alor reference snapshots without inventing historical fields."""

from datetime import datetime
from decimal import Decimal

from statarb.domain import (
    InstrumentCatalog,
    InstrumentRef,
    InstrumentSpec,
    InstrumentType,
)


def catalog_from_alor_snapshots(
    records,
    *,
    observed_at: datetime,
    version: str,
    source: str,
    valid_until: datetime | None = None,
) -> InstrumentCatalog:
    specs = []
    for row in records:
        if "future" not in (row.instrument_type or "").lower():
            raise ValueError(f"Reference row {row.symbol} is not an identified future.")
        required = (
            row.min_step,
            row.tick_value,
            row.lot_size,
            row.initial_margin_long,
            row.initial_margin_short,
            row.last_trade_at,
            row.currency,
        )
        if any(value is None for value in required):
            raise ValueError(
                f"Incomplete instrument economics for {row.symbol}; supply a verified historical specification."
            )
        specs.append(
            InstrumentSpec(
                instrument=InstrumentRef(
                    row.short_name or row.symbol, row.exchange, InstrumentType.FUTURE
                ),
                version=version,
                source=source,
                valid_from=observed_at,
                tick_size=Decimal(str(row.min_step)),
                tick_value=Decimal(str(row.tick_value)),
                contract_size=Decimal(str(row.lot_size)),
                initial_margin_long=Decimal(str(row.initial_margin_long)),
                initial_margin_short=Decimal(str(row.initial_margin_short)),
                last_trade_at=datetime.fromisoformat(row.last_trade_at),
                currency=row.currency,
                valid_until=valid_until,
            )
        )
    return InstrumentCatalog(specs)
