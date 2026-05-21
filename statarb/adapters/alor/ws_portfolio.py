from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PortfolioSubscription:
    opcode: str = field(init=False)
    guid: str
    portfolio: str
    exchange: str = "MOEX"
    format: str = "Simple"
    skip_history: bool = False

    def __post_init__(self) -> None:
        if not self.guid.strip():
            raise ValueError("Subscription guid cannot be empty.")
        if not self.portfolio.strip():
            raise ValueError("Portfolio identifier cannot be empty.")

    def to_payload(self, *, access_token: str) -> dict[str, object]:
        return {
            "opcode": self.opcode,
            "guid": self.guid,
            "token": access_token,
            "portfolio": self.portfolio,
            "exchange": self.exchange,
            "format": self.format,
            "skipHistory": self.skip_history,
        }


@dataclass(frozen=True, slots=True)
class PositionsSubscription(PortfolioSubscription):
    opcode: str = field(init=False, default="PositionsGetAndSubscribeV2")


@dataclass(frozen=True, slots=True)
class SummariesSubscription(PortfolioSubscription):
    opcode: str = field(init=False, default="SummariesGetAndSubscribeV2")


@dataclass(frozen=True, slots=True)
class OrdersSubscription(PortfolioSubscription):
    opcode: str = field(init=False, default="OrdersGetAndSubscribeV2")
    order_statuses: tuple[str, ...] = ()

    def to_payload(self, *, access_token: str) -> dict[str, object]:
        payload = PortfolioSubscription.to_payload(self, access_token=access_token)
        if self.order_statuses:
            payload["orderStatuses"] = [status for status in self.order_statuses]
        return payload


@dataclass(frozen=True, slots=True)
class TradesSubscription(PortfolioSubscription):
    opcode: str = field(init=False, default="TradesGetAndSubscribe")
    with_repo: bool = False

    def to_payload(self, *, access_token: str) -> dict[str, object]:
        payload = PortfolioSubscription.to_payload(self, access_token=access_token)
        if self.with_repo:
            payload["withRepo"] = True
        return payload


__all__ = [
    "OrdersSubscription",
    "PortfolioSubscription",
    "PositionsSubscription",
    "SummariesSubscription",
    "TradesSubscription",
]
