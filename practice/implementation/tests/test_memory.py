from pathlib import Path

from kernel.memory import ConstantMemory


def test_memory_is_durable_searchable_and_bounded_on_wake(tmp_path: Path) -> None:
    memory = ConstantMemory(tmp_path, wake_records=2, leaf_size=2, summary_chars=120)
    for index in range(5):
        memory.remember(f"fact {index}", tags=("demo",))
    assert len(memory.records()) == 5
    assert memory.recall(r"fact [34]")[0].text == "fact 4"
    wake = memory.wake()
    assert "fact 4" in wake and "fact 3" in wake
    root = memory.root_node()
    assert root is not None
    assert memory.zoom(root.id)
