from __future__ import annotations

from statarb.adapters.alor.mapper import (
    CanonicalEventKind,
    MappingStatus,
    PlaceholderAlorMapper,
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
    assert order_result.events[0].kind is CanonicalEventKind.ORDER_EVENT

    assert trade_result.events[0].kind is CanonicalEventKind.FILL_EVENT
    assert position_result.events[0].kind is CanonicalEventKind.POSITION_SNAPSHOT
    assert account_result.events[0].kind is CanonicalEventKind.ACCOUNT_SUMMARY


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
