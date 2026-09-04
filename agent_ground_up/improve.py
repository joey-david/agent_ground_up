from __future__ import annotations

import tempfile
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
    """DGM-style outer loop: select frontier, mutate a parent, evaluate, archive, repeat."""

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
        self.archive = archive
        self.curriculum = curriculum
        self.evaluator = evaluator
        self.mutator = mutator
        self.memory = memory
        self.min_valid_rate = min_valid_rate
        self.regression_tolerance = regression_tolerance

    def bootstrap(self, source_root: str | Path, family: TaskFamily) -> ArchiveEntry:
        report = self.evaluator.evaluate(source_root, family, "heldout")
        self.curriculum.record(family.name, report.mean_score)
        entry = self.archive.add(
            source_root,
            parent_id=None,
            score=report.mean_score,
            novelty=1.0,
            family=family.name,
            note="frozen baseline",
        )
        self.memory.remember(
            f"Bootstrap {family.name}: held-out score={report.mean_score:.3f}, valid={report.valid_rate:.3f}.",
            tags=("evaluation", family.name),
        )
        return entry

    def run_round(self, source_root: str | Path) -> ImprovementRound:
        family = self.curriculum.select_frontier()
        eligible = [entry for entry in self.archive.entries() if entry.family == family.name]
        if not eligible:
            self.bootstrap(source_root, family)
        parent = self.archive.select_parent(family=family.name)
        parent_path = self.archive.root / parent.id
        parent_train_report = self.evaluator.evaluate(parent_path, family, "train")
        parent_heldout_report = self.evaluator.evaluate(parent_path, family, "heldout")

        with tempfile.TemporaryDirectory(prefix="agent-descendant-") as temp:
            worktree = Path(temp)
            self.archive.materialize(parent.id, worktree)
            before = self.archive.fingerprint(worktree)
            note = self.mutator.mutate(worktree, family, parent_train_report, self.memory.wake())
            after = self.archive.fingerprint(worktree)
            if before == after:
                return self._rejected(parent, family, parent_heldout_report, note, "no source change")

            train_report = self.evaluator.evaluate(worktree, family, "train")
            heldout_report = self.evaluator.evaluate(worktree, family, "heldout")
            valid = min(train_report.valid_rate, heldout_report.valid_rate) >= self.min_valid_rate
            noncatastrophic = heldout_report.mean_score + self.regression_tolerance >= parent_heldout_report.mean_score
            accepted = valid and noncatastrophic
            reason = "archived valid descendant" if accepted else self._failure_reason(valid, noncatastrophic)
            child: ArchiveEntry | None = None
            if valid:
                child = self.archive.add(
                    worktree,
                    parent_id=parent.id,
                    score=heldout_report.mean_score,
                    novelty=self.archive.novelty(worktree),
                    family=family.name,
                    note=note,
                )
            self.curriculum.record(family.name, heldout_report.mean_score)
            self.memory.remember(
                self._memory_line(family, parent, child, train_report, heldout_report, accepted, reason, note),
                tags=("self-improvement", family.name),
            )
            return ImprovementRound(
                family=family.name,
                parent_id=parent.id,
                child_id=child.id if child else None,
                parent_score=parent_heldout_report.mean_score,
                train_score=train_report.mean_score,
                heldout_score=heldout_report.mean_score,
                accepted=accepted,
                reason=reason,
                mutation_note=note,
            )

    def run(self, source_root: str | Path, rounds: int) -> list[ImprovementRound]:
        if rounds < 1:
            return []
        return [self.run_round(source_root) for _ in range(rounds)]

    def _rejected(
        self,
        parent: ArchiveEntry,
        family: TaskFamily,
        parent_report: EvalReport,
        note: str,
        reason: str,
    ) -> ImprovementRound:
        self.memory.remember(
            f"Rejected mutation for {family.name} from {parent.id}: {reason}. Note: {note}",
            tags=("self-improvement", "rejected", family.name),
        )
        return ImprovementRound(
            family=family.name,
            parent_id=parent.id,
            child_id=None,
            parent_score=parent_report.mean_score,
            train_score=0.0,
            heldout_score=parent_report.mean_score,
            accepted=False,
            reason=reason,
            mutation_note=note,
        )

    @staticmethod
    def _failure_reason(valid: bool, noncatastrophic: bool) -> str:
        if not valid:
            return "invalid descendant"
        if not noncatastrophic:
            return "held-out regression beyond tolerance"
        return "rejected"

    @staticmethod
    def _memory_line(
        family: TaskFamily,
        parent: ArchiveEntry,
        child: ArchiveEntry | None,
        train: EvalReport,
        heldout: EvalReport,
        accepted: bool,
        reason: str,
        note: str,
    ) -> str:
        child_id = child.id if child else "not archived"
        return (
            f"Evolution {family.name}: parent={parent.id} child={child_id}; "
            f"train={train.mean_score:.3f}; heldout={heldout.mean_score:.3f}; "
            f"accepted={accepted}; reason={reason}; mutation={note}"
        )


class PromptMutator:
    """Turn any coding-agent callback into a source-code self-mutator."""

    def __init__(self, edit_agent: "EditAgent") -> None:
        self.edit_agent = edit_agent

    def mutate(
        self,
        worktree: Path,
        family: TaskFamily,
        parent_report: EvalReport,
        memory_context: str,
    ) -> str:
        failures = "\n".join(
            f"- {case.case_id}: score={case.score:.3f} valid={case.valid} {case.details}"
            for case in parent_report.cases
        )
        policy_path = worktree / "agent_ground_up" / "improvement_policy.md"
        policy = policy_path.read_text(encoding="utf-8") if policy_path.exists() else "(no policy file)"
        prompt = f"""You are improving your own agent implementation.

Current self-improvement policy (editable by you):
{policy}

Capability family: {family.name}
Description: {family.description}
Current train score: {parent_report.mean_score:.3f}

Training-case evidence:
{failures}

Persistent discoveries:
{memory_context}

Modify the implementation in this workspace to improve generalization on unseen sibling tasks.
You may change the agent loop, tools, generated skills, memory, evaluator-facing behavior, task
selection, and the self-improvement code itself. Do not hard-code benchmark answers. Prefer the
smallest falsifiable improvement. Run focused tests before finishing.

When done, return one terse line describing the hypothesis you implemented.
"""
        return self.edit_agent(worktree, prompt).strip()


class EditAgent(Protocol):
    def __call__(self, worktree: Path, prompt: str) -> str: ...
