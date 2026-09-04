from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .tasks import TaskCase, TaskFamily


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    score: float
    valid: bool = True
    details: str = ""


@dataclass(frozen=True, slots=True)
class EvalReport:
    family: str
    split: str
    mean_score: float
    valid_rate: float
    cases: tuple[CaseResult, ...]

    def to_json(self) -> str:
        raise NotImplementedError


class CandidateRunner(Protocol):
    def run(self, candidate: Path, case: TaskCase) -> CaseResult: ...


class Evaluator:
    def __init__(self, runner: CandidateRunner) -> None:
        raise NotImplementedError

    def evaluate(self, candidate: str | Path, family: TaskFamily, split: str) -> EvalReport:
        raise NotImplementedError

    @staticmethod
    def save(report: EvalReport, path: str | Path) -> None:
        raise NotImplementedError


def beats_baseline(candidate: EvalReport, baseline: EvalReport, *, margin: float = 0.0) -> bool:
    raise NotImplementedError
