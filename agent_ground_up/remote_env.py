from __future__ import annotations

import argparse
import base64
import io
import json
import os
from typing import Any
from urllib.parse import urlparse, urlunparse

from PIL import Image

from .config import load_config, secret, section


class RemoteCodingEnv:
    """TRL environment wrapper for the sandbox service running on the Mac."""

    def __init__(self) -> None:
        values = section(load_config(), "sandbox")
        self.url = values["public_url"]
        self.token = secret(values, "token_env")
        self.access_id = os.getenv("CF_ACCESS_CLIENT_ID")
        self.access_secret = os.getenv("CF_ACCESS_CLIENT_SECRET")
        self.socket: Any = None
        self.reward = 0.0
        self.steps = 0
        self.invalid_actions = 0
        self.hit_limit = False
        self._scored = False

    def reset(self, **row: Any) -> str | None:
        from websockets.sync.client import connect

        self.close()
        headers = {"Authorization": f"Bearer {self.token}"}
        if self.access_id and self.access_secret:
            headers |= {"CF-Access-Client-Id": self.access_id, "CF-Access-Client-Secret": self.access_secret}
        self.socket = connect(self._websocket_url(), additional_headers=headers, open_timeout=30)
        response = self._request({"type": "reset", "task_id": row["task_id"]})
        self.reward = 0.0
        self.steps = self.invalid_actions = 0
        self.hit_limit = self._scored = False
        return response.get("observation")

    def bash(self, command: str, timeout_s: int = 120) -> str:
        """Run a command in the remote disposable workspace.

        Args:
            command: Bash source to execute.
            timeout_s: Maximum runtime in seconds.

        Returns:
            Combined output and exit status from the sandbox.
        """
        response = self._step("bash", {"command": command, "timeout_s": timeout_s})
        return response["text"]

    def view_image(self, path: str) -> list[dict[str, Any]]:
        """Show an image from the remote disposable workspace.

        Args:
            path: Image path relative to the workspace.

        Returns:
            Multimodal image and text content blocks.
        """
        response = self._step("view_image", {"path": path})
        image = Image.open(io.BytesIO(base64.b64decode(response["image_base64"]))).copy()
        return [{"type": "image", "image": image}, {"type": "text", "text": response["text"]}]

    def _step(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        response = self._request({"type": "step", "tool": tool, "arguments": arguments})
        self.steps += 1
        self.invalid_actions += int(not response.get("valid", True))
        self.hit_limit = bool(response.get("episode_limit", False))
        return response

    def _score(self) -> float | None:
        if self._scored:
            return self.reward
        try:
            response = self._request({"type": "score"})
            if response.get("infrastructure_error"):
                return None
            base = float(response.get("reward", 0.0))
            premature = int(base == 0.0 and (self.steps == 0 or self.hit_limit))
            self.reward = base - 0.10 * self.invalid_actions - 0.10 * premature
            self._scored = True
            return self.reward
        finally:
            self.close()

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.socket is None:
            raise RuntimeError("Environment has not been reset")
        self.socket.send(json.dumps(payload))
        response = json.loads(self.socket.recv())
        if response.get("error"):
            raise RuntimeError(f"Sandbox request failed ({response.get('kind', 'unknown')})")
        return response

    def _websocket_url(self) -> str:
        parsed = urlparse(self.url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = parsed.path.rstrip("/") + "/ws"
        return urlunparse((scheme, parsed.netloc, path, "", "", ""))

    def close(self) -> None:
        if self.socket is not None:
            try:
                self.socket.close()
            finally:
                self.socket = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def remote_reward(environments: list[RemoteCodingEnv], **_: Any) -> list[float | None]:
    """Return verifier rewards; None masks infrastructure failures."""
    return [environment._score() for environment in environments]


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the remote coding environment")
    parser.add_argument("task_id")
    parser.add_argument("--command", default="pwd && find . -maxdepth 2 -type f -print")
    parser.add_argument("--config", default=os.getenv("AGENT_CONFIG", "config.yaml"))
    args = parser.parse_args()
    os.environ["AGENT_CONFIG"] = str(args.config)
    environment = RemoteCodingEnv()
    print(environment.reset(task_id=args.task_id) or "")
    print(environment.bash(args.command))
    print(f"reward={environment._score()}")


if __name__ == "__main__":
    main()
