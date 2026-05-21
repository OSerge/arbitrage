from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
from uuid import uuid4


@dataclass(slots=True)
class GuidFactory:
    _issued: set[str] = field(default_factory=set, init=False, repr=False)

    def reserve(self, guid: str) -> str:
        normalized = guid.strip()
        if not normalized:
            raise ValueError("Command guid cannot be empty.")
        if normalized in self._issued:
            raise ValueError(f"Command guid {normalized!r} has already been issued.")
        self._issued.add(normalized)
        return normalized

    def new(self) -> str:
        while True:
            candidate = uuid4().hex
            if candidate not in self._issued:
                self._issued.add(candidate)
                return candidate


@dataclass(frozen=True, slots=True)
class CwsCommandEnvelope:
    guid: str
    command: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.guid.strip():
            raise ValueError("Command guid cannot be empty.")
        if not self.command.strip():
            raise ValueError("Command name cannot be empty.")

    def to_payload(self) -> dict[str, object]:
        return {
            "guid": self.guid,
            "opcode": self.command,
            **dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class CwsAuthorizeRequest:
    guid: str
    access_token: str

    def __post_init__(self) -> None:
        if not self.guid.strip():
            raise ValueError("Authorization guid cannot be empty.")
        if not self.access_token.strip():
            raise ValueError("Access token cannot be empty.")

    def to_command(self) -> CwsCommandEnvelope:
        return CwsCommandEnvelope(
            guid=self.guid,
            command="authorize",
            payload={"token": self.access_token},
        )


__all__ = [
    "CwsAuthorizeRequest",
    "CwsCommandEnvelope",
    "GuidFactory",
]
