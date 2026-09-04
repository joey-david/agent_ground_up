import json
from pathlib import Path

from kernel.archive import Archive
from kernel.evaluate import CaseResult, Evaluator, beats_baseline
from kernel.improve import SelfImprover
from kernel.memory import ConstantMemory
from kernel.tasks import Curriculum, TaskCase, TaskFamily, load_families


class Runner:
    def run(self, candidate: Path, case: TaskCase) -> CaseResult:
        score = 1.0 if "good" in candidate.name else 0.25
        return CaseResult(case.id, score, True, candidate.name)


class CodeRunner:
    def run(self, candidate: Path, case: TaskCase) -> CaseResult:
        code = (candidate / "kernel" / "agent.py").read_text()
        score = 0.8 if "improved" in code else 0.4
        return CaseResult(case.id, score, True, code)


class Mutator:
    def mutate(self, worktree: Path, family, parent_report, memory_context: str) -> str:
        path = worktree / "kernel" / "agent.py"
        path.write_text(path.read_text() + "\nimproved = True\n")
        return "add reusable improvement"


def test_frontier_eval_archive_and_loader(tmp_path: Path) -> None:
    easy = TaskFamily("easy", "", [TaskCase("e", "", "", split="heldout")], [0.95])
    frontier = TaskFamily("frontier", "", [TaskCase("h", "", "", split="heldout")], [0.50])
    curriculum = Curriculum([easy, frontier], target=0.45)
    assert curriculum.select_frontier().name == "frontier"
    curriculum.record("frontier", 0.4)
    assert curriculum.get("frontier").history[-1] == 0.4

    evaluator = Evaluator(Runner())
    good = tmp_path / "good-agent"
    base = tmp_path / "base-agent"
    good.mkdir()
    base.mkdir()
    good_report = evaluator.evaluate(good, frontier, "heldout")
    base_report = evaluator.evaluate(base, frontier, "heldout")
    assert beats_baseline(good_report, base_report, margin=0.5)

    source = tmp_path / "source"
    file = source / "kernel" / "agent.py"
    file.parent.mkdir(parents=True)
    file.write_text("base")
    archive = Archive(tmp_path / "archive", editable=("kernel/agent.py",))
    parent = archive.add(source, parent_id=None, score=0.25, novelty=1.0, family="frontier")
    file.write_text("child")
    assert archive.novelty(source) == 1.0
    child = archive.add(
        source,
        parent_id=parent.id,
        score=0.5,
        novelty=archive.novelty(source),
        family="frontier",
    )
    destination = tmp_path / "materialized"
    archive.materialize(parent.id, destination)
    assert (destination / "kernel" / "agent.py").read_text() == "base"
    assert {entry.id for entry in archive.entries()} == {parent.id, child.id}
    assert archive.select_parent(family="frontier").id in {parent.id, child.id}

    curriculum_file = tmp_path / "curriculum.json"
    curriculum_file.write_text(
        json.dumps(
            {
                "families": [
                    {
                        "name": "loaded",
                        "description": "demo",
                        "cases": [
                            {
                                "id": "x",
                                "prompt": "p",
                                "verifier": "v",
                                "split": "train",
                            }
                        ],
                    }
                ]
            }
        )
    )
    assert load_families(curriculum_file)[0].by_split("train")[0].id == "x"


def test_self_improver_mutates_evaluates_and_archives(tmp_path: Path) -> None:
    source = tmp_path / "source"
    code = source / "kernel" / "agent.py"
    code.parent.mkdir(parents=True)
    code.write_text("base = True\n")
    family = TaskFamily(
        "debug",
        "sibling debug tasks",
        [
            TaskCase("train", "", "", split="train"),
            TaskCase("held", "", "", split="heldout"),
        ],
    )
    archive = Archive(tmp_path / "archive", editable=("kernel/agent.py",))
    improver = SelfImprover(
        archive=archive,
        curriculum=Curriculum([family]),
        evaluator=Evaluator(CodeRunner()),
        mutator=Mutator(),
        memory=ConstantMemory(tmp_path / "memory"),
    )
    result = improver.run_round(source)
    assert result.accepted
    assert result.parent_score == 0.4
    assert result.heldout_score == 0.8
    assert result.child_id is not None
    assert len(archive.entries()) == 2
