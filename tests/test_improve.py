from pathlib import Path

from agent_ground_up.archive import Archive
from agent_ground_up.evaluate import CaseResult, Evaluator
from agent_ground_up.improve import SelfImprover
from agent_ground_up.memory import ConstantMemory
from agent_ground_up.tasks import Curriculum, TaskCase, TaskFamily


class Runner:
    def run(self, candidate: Path, case: TaskCase) -> CaseResult:
        code = (candidate / "agent_ground_up" / "agent.py").read_text()
        score = 0.8 if "improved" in code else 0.4
        return CaseResult(case.id, score, True, code)


class Mutator:
    def mutate(self, worktree: Path, family, parent_report, memory_context: str) -> str:
        path = worktree / "agent_ground_up" / "agent.py"
        path.write_text(path.read_text() + "\nimproved = True\n")
        return "add a reusable improvement"


def test_self_improver_bootstraps_mutates_heldout_evaluates_and_archives(tmp_path: Path) -> None:
    source = tmp_path / "source"
    code = source / "agent_ground_up" / "agent.py"
    code.parent.mkdir(parents=True)
    code.write_text("base = True\n")

    family = TaskFamily(
        "debug",
        "debug sibling repositories",
        [
            TaskCase("t", "", "", split="train"),
            TaskCase("h", "", "", split="heldout"),
        ],
    )
    archive = Archive(tmp_path / "archive", editable=("agent_ground_up/agent.py",))
    improver = SelfImprover(
        archive=archive,
        curriculum=Curriculum([family]),
        evaluator=Evaluator(Runner()),
        mutator=Mutator(),
        memory=ConstantMemory(tmp_path / "memory"),
    )
    result = improver.run_round(source)
    assert result.accepted
    assert result.parent_score == 0.4
    assert result.heldout_score == 0.8
    assert result.child_id is not None
    assert len(archive.entries()) == 2
    assert "Evolution debug" in improver.memory.wake()
