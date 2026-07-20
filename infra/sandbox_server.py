from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import subprocess
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image
from transformers import AutoTokenizer

from agent_ground_up.config import load_config, path, secret, section

CONFIG = load_config()
VALUES = section(CONFIG, "sandbox")
TASKS_DIR = path(CONFIG, VALUES["tasks_dir"])
TOKEN = secret(VALUES, "token_env")
MAX_ENVS = VALUES["max_envs"]
MAX_STEPS = VALUES["max_steps"]
MAX_IMAGE_BYTES = VALUES["max_image_bytes"]
MAX_OUTPUT_TOKENS = VALUES["max_output_tokens"]
TOKENIZER = AutoTokenizer.from_pretrained(section(CONFIG, "model")["processor"], trust_remote_code=False)
SEMAPHORE = asyncio.Semaphore(MAX_ENVS)
app = FastAPI(docs_url=None, redoc_url=None)


class Sandbox:
    def __init__(self, task_id: str) -> None:
        if not task_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in task_id
        ):
            raise ValueError("invalid task id")
        self.task_dir = (TASKS_DIR / task_id).resolve()
        if not self.task_dir.is_relative_to(TASKS_DIR):
            raise ValueError("task escapes task directory")
        self.config = json.loads((self.task_dir / "task.json").read_text())
        self.name = f"agent-ground-up-{uuid.uuid4().hex}"
        self.steps = 0
        self.closed = False

    def start(self) -> str:
        image = self.config["image"]
        self._host(
            [
                "docker",
                "create",
                "--name",
                self.name,
                "--network",
                "none",
                "--cpus",
                str(self.config.get("cpus", 2)),
                "--memory",
                str(self.config.get("memory", "4g")),
                "--pids-limit",
                str(self.config.get("pids", 256)),
                "--workdir",
                "/workspace",
                image,
                "sleep",
                "infinity",
            ]
        )
        self._host(["docker", "start", self.name])
        workspace = self.task_dir / "workspace"
        if workspace.is_dir():
            self._host(["docker", "cp", f"{workspace}/.", f"{self.name}:/workspace"])
        self._host(["docker", "exec", "--user", "0", self.name, "chown", "-R", "65534:65534", "/workspace"])
        return str(self.config.get("observation", "Workspace ready."))

    def bash(self, command: str, timeout_s: int) -> dict[str, Any]:
        if self.steps >= MAX_STEPS:
            return {"text": "Episode step limit reached.", "valid": False, "episode_limit": True}
        if not isinstance(command, str) or not command.strip() or len(command) > 100_000 or not 1 <= timeout_s <= 3600:
            return {"text": "Invalid bash arguments.", "valid": False}
        self.steps += 1
        try:
            result = self._host(
                [
                    "docker",
                    "exec",
                    "--user",
                    "65534:65534",
                    "--workdir",
                    "/workspace",
                    self.name,
                    "/bin/bash",
                    "-lc",
                    command,
                ],
                timeout=timeout_s,
                check=False,
            )
            text = _truncate(result.stdout) + f"\n[exit code: {result.returncode}]"
            return {"text": text, "valid": True, "episode_limit": self.steps >= MAX_STEPS}
        except subprocess.TimeoutExpired as error:
            self.close()
            output = error.stdout or ""
            return {"text": output + "\n[exit code: -1] [timed out]", "valid": True, "episode_limit": True}

    def view_image(self, path: str) -> dict[str, Any]:
        if self.steps >= MAX_STEPS:
            return {"text": "Episode step limit reached.", "valid": False, "episode_limit": True}
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts:
            return {"text": "Invalid image path.", "valid": False}
        self.steps += 1
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / pure.name
            try:
                self._host(["docker", "cp", f"{self.name}:/workspace/{pure}", str(destination)])
            except subprocess.CalledProcessError:
                return {"text": "Image does not exist.", "valid": False}
            if not destination.is_file() or destination.stat().st_size > MAX_IMAGE_BYTES:
                return {"text": "Image is invalid or too large.", "valid": False}
            try:
                with Image.open(destination) as image:
                    image.verify()
            except (OSError, ValueError):
                return {"text": "File is not a valid image.", "valid": False}
            mime = mimetypes.guess_type(destination.name)[0]
            if not mime or not mime.startswith("image/"):
                return {"text": "File is not a supported image.", "valid": False}
            return {
                "text": f"Image: {path} ({mime})",
                "image_base64": base64.b64encode(destination.read_bytes()).decode("ascii"),
                "valid": True,
                "episode_limit": self.steps >= MAX_STEPS,
            }

    def score(self) -> dict[str, Any]:
        verifier = self.config.get("verifier")
        if not verifier or self.closed:
            return {"reward": 0.0, "infrastructure_error": True}
        try:
            result = self._host(
                ["docker", "exec", "--user", "0", "--workdir", "/workspace", self.name, "/bin/bash", "-lc", verifier],
                timeout=int(self.config.get("verifier_timeout_s", 300)),
                check=False,
            )
            return {"reward": float(result.returncode == 0), "verifier_output": result.stdout[-4000:]}
        except subprocess.TimeoutExpired:
            return {"reward": 0.0, "infrastructure_error": True}
        finally:
            self.close()

    def close(self) -> None:
        if self.closed:
            return
        subprocess.run(["docker", "rm", "-f", self.name], text=True, capture_output=True, check=False)
        self.closed = True

    @staticmethod
    def _host(command: list[str], timeout: int = 120, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout, check=check
        )


def _truncate(text: str) -> str:
    tokens = TOKENIZER.encode(text, add_special_tokens=False)
    if len(tokens) <= MAX_OUTPUT_TOKENS:
        return text
    half = MAX_OUTPUT_TOKENS // 2
    omitted = len(tokens) - 2 * half
    head = TOKENIZER.decode(tokens[:half])
    tail = TOKENIZER.decode(tokens[-half:])
    return f"{head}\n... [{omitted} tokens omitted] ...\n{tail}"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    if websocket.headers.get("authorization") != f"Bearer {TOKEN}":
        await websocket.close(code=4401)
        return
    await websocket.accept()
    sandbox: Sandbox | None = None
    acquired = False
    try:
        await SEMAPHORE.acquire()
        acquired = True
        while True:
            message = await websocket.receive_json()
            kind = message.get("type")
            if kind == "reset":
                if sandbox:
                    await asyncio.to_thread(sandbox.close)
                sandbox = Sandbox(message["task_id"])
                observation = await asyncio.to_thread(sandbox.start)
                await websocket.send_json({"observation": observation})
            elif kind == "step" and sandbox:
                arguments = message.get("arguments") or {}
                if message.get("tool") == "bash":
                    response = await asyncio.to_thread(
                        sandbox.bash, arguments.get("command", ""), arguments.get("timeout_s", 120)
                    )
                elif message.get("tool") == "view_image":
                    response = await asyncio.to_thread(sandbox.view_image, arguments.get("path", ""))
                else:
                    response = {"text": "Unknown tool.", "valid": False}
                await websocket.send_json(response)
            elif kind == "score" and sandbox:
                await websocket.send_json(await asyncio.to_thread(sandbox.score))
                sandbox = None
            else:
                await websocket.send_json({"error": True, "kind": "invalid_request"})
    except WebSocketDisconnect:
        pass
    except Exception:
        await websocket.send_json({"error": True, "kind": "sandbox_failure"})
    finally:
        if sandbox:
            await asyncio.to_thread(sandbox.close)
        if acquired:
            SEMAPHORE.release()
