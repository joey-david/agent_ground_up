from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class TaskCase:
    id: str
    prompt: str
    verifier: str
    workspace: str | None = None
    split: str = "train"


@dataclass(slots=True)
class TaskFamily:
    name: str
    description: str
    cases: list[TaskCase]
    history: list[float] = field(default_factory=list)

    def by_split(self, split: str) -> list[TaskCase]:
        raise NotImplementedError

    @property
    def mean_score(self) -> float:
        raise NotImplementedError

    @property
    def uncertainty(self) -> float:
        raise NotImplementedError

    def observe(self, score: float) -> None:
        raise NotImplementedError


class Curriculum:
    def __init__(self, families: Iterable[TaskFamily], *, target: float = 0.45) -> None:
        raise NotImplementedError

    def select_frontier(self) -> TaskFamily:
        raise NotImplementedError

    def record(self, family_name: str, score: float) -> None:
        raise NotImplementedError

    def get(self, name: str) -> TaskFamily:
        raise NotImplementedError


def load_families(path: str | Path) -> list[TaskFamily]:
    raise NotImplementedError
