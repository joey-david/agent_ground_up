from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import pytest


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode())

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode()


@pytest.fixture
def server(monkeypatch, tmp_path: Path):
    task = tmp_path / "tasks" / "example"
    (task / "workspace").mkdir(parents=True)
    (task / "task.json").write_text(
        json.dumps(
            {
                "image": "python:3.12-slim",
                "observation": "ready",
                "verifier": "test -f answer.txt",
                "verifier_timeout_s": 10,
            }
        )
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        """model:
  processor: fake
sandbox:
  tasks_dir: tasks
  token_env: AGENT_ENV_TOKEN
  max_envs: 1
  max_steps: 2
  max_image_bytes: 1024
  max_output_tokens: 8
"""
    )
    monkeypatch.setenv("AGENT_CONFIG", str(config))
    monkeypatch.setenv("AGENT_ENV_TOKEN", "token")
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: CharacterTokenizer()
    )
    import infra.sandbox_server

    return importlib.reload(infra.sandbox_server)


def test_sandbox_command_and_verifier_are_task_results(server, monkeypatch) -> None:
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["docker", "exec"] and command[-1] == "test -f answer.txt":
            return subprocess.CompletedProcess(command, 1, stdout="failed verifier", stderr="")
        if command[:2] == ["docker", "exec"]:
            return subprocess.CompletedProcess(command, 7, stdout="abcdefghijkl", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(server.subprocess, "run", run)
    sandbox = server.Sandbox("example")
    assert sandbox.start() == "ready"
    result = sandbox.bash("exit 7", 10)
    assert result["valid"] is True
    assert "[exit code: 7]" in result["text"]
    assert "tokens omitted" in result["text"]
    assert sandbox.score() == {"reward": 0.0, "verifier_output": "failed verifier"}
    assert any(command[:3] == ["docker", "rm", "-f"] for command in calls)


def test_sandbox_timeout_is_infrastructure_only_for_verifier(server, monkeypatch) -> None:
    def run(command, **kwargs):
        if command[:2] == ["docker", "exec"] and command[-1] == "test -f answer.txt":
            raise subprocess.TimeoutExpired(command, kwargs.get("timeout", 1), output="partial")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(server.subprocess, "run", run)
    sandbox = server.Sandbox("example")
    sandbox.start()
    assert sandbox.score() == {"reward": 0.0, "infrastructure_error": True}
