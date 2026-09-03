from pathlib import Path

from agent_ground_up.archive import Archive


def write_candidate(root: Path, value: str) -> None:
    path = root / "agent_ground_up" / "agent.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def test_archive_retains_branches_materializes_and_detects_novelty(tmp_path: Path) -> None:
    editable = ("agent_ground_up/agent.py",)
    archive = Archive(tmp_path / "archive", editable=editable)
    source = tmp_path / "source"
    write_candidate(source, "base")
    base = archive.add(source, parent_id=None, score=0.5, novelty=1.0, family="debug")

    write_candidate(source, "child")
    assert archive.novelty(source) == 1.0
    child = archive.add(source, parent_id=base.id, score=0.4, novelty=1.0, family="debug")
    assert {entry.id for entry in archive.entries()} == {base.id, child.id}

    destination = tmp_path / "materialized"
    archive.materialize(base.id, destination)
    assert (destination / "agent_ground_up" / "agent.py").read_text() == "base"
    assert archive.select_parent(family="debug").id in {base.id, child.id}
