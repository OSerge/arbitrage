from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from statarb.adapters.alor.mapper import (
    AlorMappingContext,
    CanonicalAlorMapper,
    MappingStatus,
    PlaceholderAlorMapper,
)
from statarb.domain.events import ExecutionContour
from statarb.domain.instruments import InstrumentType
from statarb.domain.orders import OrderSide, OrderStatus, OrderType
from tests.adapters.alor.fixtures.portfolio_payloads import (
    MAPPING_CAPTURED_AT,
    ORDER_WORKING,
    POSITION_APTK,
    SUMMARY_D00014,
    TRADE_GAZP,
)


def _context(*, portfolio: str = "D00014") -> AlorMappingContext:
    return AlorMappingContext(
        portfolio=portfolio,
        exchange="MOEX",
        contour=ExecutionContour.BROKER_TEST,
        captured_at=MAPPING_CAPTURED_AT,
    )


def test_placeholder_mapper_classifies_known_alor_streams() -> None:
    mapper = PlaceholderAlorMapper()

    order_result = mapper.map_payload(
        stream="portfolio",
        opcode="OrdersGetAndSubscribeV2",
        payload={"id": "order-1"},
    )
    trade_result = mapper.map_payload(
        stream="portfolio",
        opcode="TradesGetAndSubscribe",
        payload={"id": "trade-1"},
    )
    position_result = mapper.map_payload(
        stream="portfolio",
        opcode="PositionsGetAndSubscribeV2",
        payload={"symbol": "SBER"},
    )
    account_result = mapper.map_payload(
        stream="portfolio",
        opcode="SummariesGetAndSubscribeV2",
        payload={"portfolio": "D39004"},
    )

    assert order_result.status is MappingStatus.PLACEHOLDER
    assert order_result.events[0].kind.value == "order-event"

    assert trade_result.events[0].kind.value == "fill-event"
    assert position_result.events[0].kind.value == "position-snapshot"
    assert account_result.events[0].kind.value == "account-summary"


def test_placeholder_mapper_preserves_unknown_payloads_for_future_contract_wiring() -> None:
    mapper = PlaceholderAlorMapper()

    result = mapper.map_payload(
        stream="market-data",
        opcode="QuotesSubscribe",
        payload={"symbol": "SBER", "bid": 300.1},
    )

    assert result.status is MappingStatus.DEFERRED
    assert result.events == ()
    assert result.unmapped_reason == "No canonical event contract is frozen for this payload yet."
    assert result.payload["symbol"] == "SBER"


def test_canonical_mapper_maps_real_summary_payload_from_smoke() -> None:
    mapper = CanonicalAlorMapper()
    result = mapper.map_http_surface(
        surface="summary",
        payload=SUMMARY_D00014,
        context=_context(),
    )

    assert result.status is MappingStatus.MAPPED
    assert len(result.events) == 1
    summary = result.events[0]
    assert summary.account_id == "D00014"
    assert summary.equity == Decimal("100000")
    assert summary.available_cash == Decimal("100000")
    assert summary.contour is ExecutionContour.BROKER_TEST


def test_canonical_mapper_maps_empty_portfolio_arrays_from_smoke() -> None:
    mapper = CanonicalAlorMapper()
    context = _context()

    for surface in ("positions", "orders", "trades"):
        result = mapper.map_http_surface(surface=surface, payload=[], context=context)
        assert result.status is MappingStatus.MAPPED
        assert result.events == ()


def test_canonical_mapper_maps_documented_position_order_and_trade_payloads() -> None:
    mapper = CanonicalAlorMapper()
    context = _context(portfolio="D39004")

    position_result = mapper.map_http_surface(
        surface="positions",
        payload=[POSITION_APTK],
        context=context,
    )
    order_result = mapper.map_http_surface(
        surface="orders",
        payload=[ORDER_WORKING],
        context=context,
    )
    trade_result = mapper.map_http_surface(
        surface="trades",
        payload=[TRADE_GAZP],
        context=context,
    )

    position = position_result.events[0]
    order = order_result.events[0]
    fill = trade_result.events[0]

    assert position.instrument.symbol == "APTK"
    assert position.net_quantity == Decimal("60")
    assert position.average_price == Decimal("9.1533")

    assert order.order_id == "2006665023417483347"
    assert order.order_type is OrderType.LIMIT
    assert order.side is OrderSide.BUY
    assert order.status is OrderStatus.WORKING
    assert order.instrument.instrument_type is InstrumentType.FUTURE

    assert fill.fill_id == "4421925358"
    assert fill.order_id == "42030831469"
    assert fill.fill_quantity == Decimal("10")
    assert fill.commission == Decimal("1.36105")


def test_canonical_mapper_skips_currency_positions() -> None:
    mapper = CanonicalAlorMapper()
    result = mapper.map_http_surface(
        surface="positions",
        payload=[{"symbol": "RUB", "exchange": "MOEX", "qtyUnits": 100000, "isCurrency": True}],
        context=_context(),
    )

    assert result.status is MappingStatus.MAPPED
    assert result.events == ()


def test_canonical_mapper_maps_ws_envelope_payloads() -> None:
    mapper = CanonicalAlorMapper()
    context = _context(portfolio="D39004")

    order_result = mapper.map_payload(
        stream="portfolio",
        opcode="OrdersGetAndSubscribeV2",
        payload={"data": ORDER_WORKING, "guid": "orders-read-only"},
        context=context,
    )
    trade_result = mapper.map_payload(
        stream="portfolio",
        opcode="TradesGetAndSubscribe",
        payload={"data": TRADE_GAZP, "guid": "trades-read-only"},
        context=context,
    )

    assert order_result.status is MappingStatus.MAPPED
    assert trade_result.status is MappingStatus.MAPPED
    assert order_result.events[0].broker_order_id == "2006665023417483347"
    assert trade_result.events[0].broker_trade_id == "4421925358"


def test_canonical_mapper_marks_partially_filled_working_orders() -> None:
    mapper = CanonicalAlorMapper()
    payload = dict(ORDER_WORKING)
    payload["filledQtyUnits"] = 1
    payload["status"] = "working"

    result = mapper.map_http_surface(
        surface="orders",
        payload=[payload],
        context=_context(portfolio="D39004"),
    )

    assert result.events[0].status is OrderStatus.PARTIALLY_FILLED
