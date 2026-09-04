from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperienceEvent:
    id: int
    kind: str
    payload: Any
    created_at: str


class ExperienceLog:
    def __init__(self, root: str | Path) -> None:
        raise NotImplementedError

    def append(self, kind: str, payload: Any) -> ExperienceEvent:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def read(self, start: int, end: int) -> list[ExperienceEvent]:
        raise NotImplementedError

    def search(self, pattern: str, *, limit: int = 12) -> list[ExperienceEvent]:
        raise NotImplementedError

    @staticmethod
    def format(events: list[ExperienceEvent]) -> str:
        raise NotImplementedError
