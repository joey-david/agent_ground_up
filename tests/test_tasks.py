from agent_ground_up.tasks import Curriculum, TaskCase, TaskFamily


def family(name: str, history: list[float]) -> TaskFamily:
    return TaskFamily(name, name, [TaskCase(name, "p", "true")], history)


def test_curriculum_selects_frontier_not_easiest() -> None:
    curriculum = Curriculum(
        [
            family("easy", [0.95, 1.0]),
            family("frontier", [0.45, 0.5]),
            family("hard", [0.0, 0.05]),
        ],
        target=0.45,
    )
    assert curriculum.select_frontier().name == "frontier"
    curriculum.record("frontier", 0.7)
    assert curriculum.get("frontier").history[-1] == 0.7
