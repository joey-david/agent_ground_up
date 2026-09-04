from pathlib import Path

from kernel.archive import Archive
from kernel.evaluate import CaseResult, Evaluator, beats_baseline
from kernel.tasks import Curriculum, TaskCase, TaskFamily


class Runner:
    def run(self, candidate: Path, case: TaskCase) -> CaseResult:
        score = 1.0 if "good" in candidate.name else 0.25
        return CaseResult(case.id, score, True, candidate.name)


def test_frontier_eval_and_archive(tmp_path: Path) -> None:
    easy = TaskFamily("easy", "", [TaskCase("e", "", "", split="heldout")], [0.95])
    frontier = TaskFamily("frontier", "", [TaskCase("h", "", "", split="heldout")], [0.50])
    curriculum = Curriculum([easy, frontier], target=0.45)
    assert curriculum.select_frontier().name == "frontier"

    evaluator = Evaluator(Runner())
    good = tmp_path / "good-agent"
    base = tmp_path / "base-agent"
    good.mkdir(); base.mkdir()
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
    child = archive.add(source, parent_id=parent.id, score=0.5, novelty=archive.novelty(source), family="frontier")
    assert {entry.id for entry in archive.entries()} == {parent.id, child.id}
    assert archive.select_parent(family="frontier").id in {parent.id, child.id}
