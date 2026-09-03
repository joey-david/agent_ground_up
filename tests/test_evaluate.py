from pathlib import Path

from agent_ground_up.evaluate import CaseResult, Evaluator, beats_baseline
from agent_ground_up.tasks import TaskCase, TaskFamily


class Runner:
    def run(self, candidate: Path, case: TaskCase) -> CaseResult:
        return CaseResult(case.id, 1.0 if "good" in candidate.name else 0.25, True, candidate.name)


def test_evaluator_uses_explicit_splits_and_baseline_gate(tmp_path: Path) -> None:
    family = TaskFamily(
        "debug",
        "",
        [
            TaskCase("train", "", "", split="train"),
            TaskCase("held", "", "", split="heldout"),
        ],
    )
    evaluator = Evaluator(Runner())
    good = tmp_path / "good-agent"
    base = tmp_path / "base-agent"
    good.mkdir()
    base.mkdir()
    candidate = evaluator.evaluate(good, family, "heldout")
    baseline = evaluator.evaluate(base, family, "heldout")
    assert candidate.cases[0].case_id == "held"
    assert beats_baseline(candidate, baseline, margin=0.5)
