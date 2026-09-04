from pathlib import Path

from agent_ground_up.experience import ExperienceLog


def test_experience_log_is_append_only_searchable_and_range_readable(tmp_path: Path) -> None:
    log = ExperienceLog(tmp_path / "experience")
    log.append("task", {"task": "repair parser"})
    log.append("tool_call", {"name": "bash", "arguments": "pytest -q"})
    log.append("tool_result", {"content": "2 failed, 5 passed"})

    reopened = ExperienceLog(tmp_path / "experience")
    assert reopened.count() == 3
    matches = reopened.search(r"failed|pytest")
    assert [event.id for event in matches] == [2, 1]
    assert [event.id for event in reopened.read(1, 3)] == [1, 2]
    assert "2 failed" in reopened.format(reopened.read(2, 3))


def test_experience_log_does_not_duplicate_embedded_image_bytes(tmp_path: Path) -> None:
    log = ExperienceLog(tmp_path / "experience")
    log.append(
        "tool_result",
        {
            "content": [
                {"type": "text", "text": "image metadata"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET"}},
            ]
        },
    )
    raw = log.path.read_text()
    assert "SECRET" not in raw
    assert "embedded image retained in workspace" in raw
