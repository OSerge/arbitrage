from datetime import UTC, datetime
from dataclasses import replace

import pytest

from statarb.adapters.alor.http_market_data import normalize_securities_response
from statarb.adapters.alor.mapper import AlorMappingContext, map_alor_trade
from statarb.data.specifications import catalog_from_alor_snapshots
from statarb.domain import InstrumentRef, InstrumentType


def test_alor_economics_retain_tick_value_margin_and_expiration():
    records = normalize_securities_response(
        [
            {
                "symbol": "SBRF-3.26",
                "shortName": "SRH6",
                "exchange": "MOEX",
                "type": "Future",
                "currency": "RUB",
                "lotsize": 100,
                "minstep": 1,
                "pricestep": 10,
                "marginbuy": 1000,
                "marginsell": 1200,
                "cancellation": "2026-03-20T15:00:00Z",
            }
        ]
    )
    catalog = catalog_from_alor_snapshots(
        records,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        version="reference-1",
        source="synthetic test source",
    )
    spec = catalog.specs[0]
    assert spec.instrument.symbol == "SRH6"
    assert (
        spec.point_value == 10
    )  # Already per contract; lot size is not multiplied again.
    assert spec.initial_margin_short == 1200
    assert spec.valid_from == datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="Incomplete instrument economics"):
        catalog_from_alor_snapshots(
            (replace(records[0], tick_value=None),),
            observed_at=datetime(2026, 1, 1, tzinfo=UTC),
            version="bad",
            source="fixture",
        )


def test_mapper_resolves_aliases_and_never_confuses_lots_with_units():
    instrument = InstrumentRef("SRH6", "MOEX", InstrumentType.FUTURE)
    payload = {
        "id": "1",
        "orderno": "order-1",
        "symbol": "SBRF-3.26",
        "side": "buy",
        "qty": 2,
        "price": 30000,
        "date": "2026-01-05T10:00:00Z",
    }
    context = AlorMappingContext(
        "test", instruments={"SBRF-3.26": instrument}, lot_sizes={"SBRF-3.26": 1}
    )
    mapped = map_alor_trade(payload, context=context)
    assert mapped.instrument == instrument
    assert mapped.fill_quantity == 2
    with pytest.raises(ValueError, match="explicit reference lot size"):
        map_alor_trade(payload, context=AlorMappingContext("test"))
    stock = {**payload, "symbol": "SBER"}
    mapped_stock = map_alor_trade(
        stock, context=AlorMappingContext("test", lot_sizes={"SBER": 10})
    )
    assert mapped_stock.fill_quantity == 20
