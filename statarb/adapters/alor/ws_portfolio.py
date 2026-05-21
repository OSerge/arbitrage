from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PortfolioSubscription:
    opcode: str = field(init=False)
    guid: str
    portfolio: str
    exchange: str = "MOEX"
    format: str = "Simple"

    def __post_init__(self) -> None:
        if not self.guid.strip():
            raise ValueError("Subscription guid cannot be empty.")
        if not self.portfolio.strip():
            raise ValueError("Portfolio identifier cannot be empty.")

    def to_payload(self, *, access_token: str) -> dict[str, str]:
        return {
            "opcode": self.opcode,
            "guid": self.guid,
            "token": access_token,
            "portfolio": self.portfolio,
            "exchange": self.exchange,
            "format": self.format,
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


@dataclass(frozen=True, slots=True)
class TradesSubscription(PortfolioSubscription):
    opcode: str = field(init=False, default="TradesGetAndSubscribe")


__all__ = [
    "OrdersSubscription",
    "PortfolioSubscription",
    "PositionsSubscription",
    "SummariesSubscription",
    "TradesSubscription",
]
