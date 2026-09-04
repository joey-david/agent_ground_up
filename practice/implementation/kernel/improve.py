from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .archive import Archive, ArchiveEntry
from .evaluate import EvalReport, Evaluator
from .memory import ConstantMemory
from .tasks import Curriculum, TaskFamily


class Mutator(Protocol):
    def mutate(
        self,
        worktree: Path,
        family: TaskFamily,
        parent_report: EvalReport,
        memory_context: str,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class ImprovementRound:
    family: str
    parent_id: str
    child_id: str | None
    parent_score: float
    train_score: float
    heldout_score: float
    accepted: bool
    reason: str
    mutation_note: str


class SelfImprover:
    def __init__(
        self,
        *,
        archive: Archive,
        curriculum: Curriculum,
        evaluator: Evaluator,
        mutator: Mutator,
        memory: ConstantMemory,
        min_valid_rate: float = 1.0,
        regression_tolerance: float = 0.05,
    ) -> None:
        raise NotImplementedError

    def bootstrap(self, source_root: str | Path, family: TaskFamily) -> ArchiveEntry:
        raise NotImplementedError

    def run_round(self, source_root: str | Path) -> ImprovementRound:
        raise NotImplementedError

    def run(self, source_root: str | Path, rounds: int) -> list[ImprovementRound]:
        raise NotImplementedError
