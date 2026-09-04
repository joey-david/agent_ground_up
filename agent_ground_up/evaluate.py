from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
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
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


class CandidateRunner(Protocol):
    def run(self, candidate: Path, case: TaskCase) -> CaseResult: ...


class Evaluator:
    """Evaluate candidate descendants on explicit train/held-out splits."""

    def __init__(self, runner: CandidateRunner) -> None:
        self.runner = runner

    def evaluate(self, candidate: str | Path, family: TaskFamily, split: str) -> EvalReport:
        candidate_path = Path(candidate).resolve()
        cases = family.by_split(split)
        if not cases:
            raise ValueError(f"task family {family.name!r} has no {split!r} cases")
        results = tuple(self.runner.run(candidate_path, case) for case in cases)
        mean_score = sum(result.score for result in results) / len(results)
        valid_rate = sum(result.valid for result in results) / len(results)
        return EvalReport(family.name, split, mean_score, valid_rate, results)

    @staticmethod
    def save(report: EvalReport, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report.to_json() + "\n", encoding="utf-8")


class LocalCodingRunner:
    """Development-only runner. Candidate Python executes on the host; use only for trusted smoke tests."""

    def __init__(
        self,
        *,
        repository_root: str | Path,
        python: str = sys.executable,
        timeout_s: int = 1800,
        environment: dict[str, str] | None = None,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.python = python
        self.timeout_s = timeout_s
        self.environment = environment or {}

    def run(self, candidate: Path, case: TaskCase) -> CaseResult:
        if case.workspace is None:
            raise ValueError(f"case {case.id!r} needs a workspace template")
        template = (self.repository_root / case.workspace).resolve()
        if not template.exists() or not template.is_dir():
            raise ValueError(f"workspace template does not exist: {template}")
        run_py = candidate / "scripts" / "run.py"
        config = candidate / "configs" / "lamgate.yaml"
        if not run_py.exists() or not config.exists():
            return CaseResult(case.id, 0.0, False, "candidate missing scripts/run.py or configs/lamgate.yaml")

        with tempfile.TemporaryDirectory(prefix=f"eval-{case.id}-") as temp:
            workdir = Path(temp) / "workspace"
            shutil.copytree(template, workdir)
            env = os.environ.copy() | self.environment
            inherited = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(candidate) + (os.pathsep + inherited if inherited else "")
            process = subprocess.run(
                [
                    self.python,
                    str(run_py),
                    case.prompt,
                    "--config",
                    str(config),
                    "--workdir",
                    str(workdir),
                    "--quiet",
                    "--no-memory",
                ],
                cwd=candidate,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.timeout_s,
                check=False,
            )
            if process.returncode != 0:
                return CaseResult(case.id, 0.0, False, _tail(process.stdout))
            verifier = subprocess.run(
                case.verifier,
                cwd=workdir,
                shell=True,
                executable="/bin/bash",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=min(self.timeout_s, 300),
                check=False,
            )
            score = 1.0 if verifier.returncode == 0 else 0.0
            details = _tail(verifier.stdout or process.stdout)
            return CaseResult(case.id, score, True, details)


def beats_baseline(candidate: EvalReport, baseline: EvalReport, *, margin: float = 0.0) -> bool:
    """Require held-out improvement without accepting invalid-task regressions."""
    if candidate.family != baseline.family or candidate.split != baseline.split:
        raise ValueError("reports must describe the same family and split")
    return candidate.valid_rate >= baseline.valid_rate and (
        candidate.mean_score >= baseline.mean_score + margin
    )


def _tail(text: str, limit: int = 1200) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[-limit:]
