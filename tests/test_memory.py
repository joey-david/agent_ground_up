from pathlib import Path

from agent_ground_up.memory import ConstantMemory


def test_memory_persists_wakes_recalls_and_zooms(tmp_path: Path) -> None:
    memory = ConstantMemory(tmp_path / "memory", wake_records=2, leaf_size=2, summary_chars=200)
    for index in range(5):
        memory.remember(f"episode {index} solved parser bug", tags=("debug",))

    reopened = ConstantMemory(tmp_path / "memory", wake_records=2, leaf_size=2, summary_chars=200)
    assert len(reopened.records()) == 5
    wake = reopened.wake()
    assert "episode 4" in wake and "episode 3" in wake
    assert "episode 0" not in wake.split("Recent memories:", 1)[-1]

    matches = reopened.recall(r"episode [12]")
    assert [record.id for record in matches] == [2, 1]

    root = reopened.root_node()
    assert root is not None and root.level >= 1
    expanded = reopened.zoom(root.id)
    assert "[L" in expanded
    leaf_id = expanded.split("[", 1)[1].split("]", 1)[0]
    if not leaf_id.startswith("L0"):
        leaf_id = reopened.zoom(leaf_id).split("[", 1)[1].split("]", 1)[0]
    assert "parser bug" in reopened.zoom(leaf_id)
