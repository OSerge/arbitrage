from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True, slots=True)
class MarketDataSubscription:
    opcode: str = field(init=False)
    guid: str
    code: str
    exchange: str = "MOEX"
    format: str = "Simple"
    depth: int | None = None

    def __post_init__(self) -> None:
        if not self.guid.strip():
            raise ValueError("Subscription guid cannot be empty.")
        if not self.code.strip():
            raise ValueError("Instrument code cannot be empty.")

    def to_payload(self, *, access_token: str) -> dict[str, str | int]:
        payload: dict[str, str | int] = {
            "opcode": self.opcode,
            "guid": self.guid,
            "token": access_token,
            "code": self.code,
            "exchange": self.exchange,
            "format": self.format,
        }
        if self.depth is not None:
            payload["depth"] = self.depth
        return payload


@dataclass(frozen=True, slots=True)
class WsRequestEnvelope:
    url: str
    payload: Mapping[str, str | int]


@dataclass(frozen=True, slots=True)
class QuotesSubscription(MarketDataSubscription):
    opcode: str = field(init=False, default="QuotesSubscribe")


@dataclass(frozen=True, slots=True)
class OrderBookSubscription(MarketDataSubscription):
    opcode: str = field(init=False, default="OrderBookGetAndSubscribe")
    depth: int = 20


__all__ = [
    "MarketDataSubscription",
    "OrderBookSubscription",
    "QuotesSubscription",
    "WsRequestEnvelope",
]
