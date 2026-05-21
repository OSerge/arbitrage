from __future__ import annotations

from statarb.adapters.alor.ws_portfolio import OrdersSubscription, PositionsSubscription, TradesSubscription


def test_positions_subscription_includes_skip_history_flag() -> None:
    payload = PositionsSubscription(
        guid="positions-guid",
        portfolio="D39004",
        exchange="MOEX",
        skip_history=True,
    ).to_payload(access_token="access-token")

    assert payload == {
        "opcode": "PositionsGetAndSubscribeV2",
        "guid": "positions-guid",
        "token": "access-token",
        "portfolio": "D39004",
        "exchange": "MOEX",
        "format": "Simple",
        "skipHistory": True,
    }


def test_orders_subscription_emits_optional_status_filter() -> None:
    payload = OrdersSubscription(
        guid="orders-guid",
        portfolio="D39004",
        exchange="MOEX",
        order_statuses=("working", "filled"),
    ).to_payload(access_token="access-token")

    assert payload["opcode"] == "OrdersGetAndSubscribeV2"
    assert payload["orderStatuses"] == ["working", "filled"]
    assert payload["skipHistory"] is False


def test_trades_subscription_emits_repo_flag() -> None:
    payload = TradesSubscription(
        guid="trades-guid",
        portfolio="D39004",
        exchange="MOEX",
        with_repo=True,
    ).to_payload(access_token="access-token")

    assert payload["opcode"] == "TradesGetAndSubscribe"
    assert payload["withRepo"] is True
