from __future__ import annotations

import json
from pathlib import Path

from agent_ground_up.remote_env import RemoteCodingEnv, remote_reward


class FakeSocket:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.sent: list[dict] = []
        self.closed = False

    def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    def recv(self) -> str:
        return json.dumps(self.responses.pop(0))

    def close(self) -> None:
        self.closed = True


def configure(monkeypatch, tmp_path: Path, url: str) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(f"sandbox:\n  public_url: {url}\n  token_env: AGENT_ENV_TOKEN\n")
    monkeypatch.setenv("AGENT_CONFIG", str(config))
    monkeypatch.setenv("AGENT_ENV_TOKEN", "token")


def test_remote_reward_scores_and_closes(monkeypatch, tmp_path: Path) -> None:
    configure(monkeypatch, tmp_path, "https://sandbox.example/base")
    environment = RemoteCodingEnv()
    environment.socket = FakeSocket([{"reward": 1.0}])
    environment.invalid_actions = 1
    environment.steps = 2

    assert environment._websocket_url() == "wss://sandbox.example/base/ws"
    assert remote_reward([environment]) == [0.9]
    assert environment.socket is None


def test_infrastructure_failure_is_masked(monkeypatch, tmp_path: Path) -> None:
    configure(monkeypatch, tmp_path, "http://localhost:8001")
    environment = RemoteCodingEnv()
    environment.socket = FakeSocket([{"reward": 0.0, "infrastructure_error": True}])
    assert remote_reward([environment]) == [None]
