from __future__ import annotations

import json
import math
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
        return [case for case in self.cases if case.split == split]

    @property
    def mean_score(self) -> float:
        if not self.history:
            return 0.5
        return sum(self.history) / len(self.history)

    @property
    def uncertainty(self) -> float:
        return 1.0 / math.sqrt(len(self.history) + 1)

    def observe(self, score: float) -> None:
        self.history.append(float(score))


class Curriculum:
    """Select task families near the current capability frontier."""

    def __init__(self, families: Iterable[TaskFamily], *, target: float = 0.45) -> None:
        self.families = list(families)
        if not self.families:
            raise ValueError("at least one task family is required")
        if not 0.0 <= target <= 1.0:
            raise ValueError("target must be in [0, 1]")
        self.target = target

    def select_frontier(self) -> TaskFamily:
        def priority(family: TaskFamily) -> tuple[float, float, str]:
            distance = abs(family.mean_score - self.target)
            return (distance - 0.15 * family.uncertainty, -family.uncertainty, family.name)

        return min(self.families, key=priority)

    def record(self, family_name: str, score: float) -> None:
        self.get(family_name).observe(score)

    def get(self, name: str) -> TaskFamily:
        for family in self.families:
            if family.name == name:
                return family
        raise KeyError(name)


def load_families(path: str | Path) -> list[TaskFamily]:
    """Load a compact JSON curriculum file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    families: list[TaskFamily] = []
    for item in data["families"]:
        cases = [TaskCase(**case) for case in item["cases"]]
        families.append(
            TaskFamily(
                name=item["name"],
                description=item.get("description", ""),
                cases=cases,
                history=[float(score) for score in item.get("history", [])],
            )
        )
    return families
