from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .tools import TOOL_SCHEMAS, Toolbox
from .ui import TUI

SYSTEM_PROMPT = """You are a coding agent working in the provided workspace. Work until the task is complete.
Use bash to inspect, edit, and test the repository; each call starts in the repository root. Use view_image
for image files. Prefer minimal changes that fit the existing code. Treat tool failures as observations and
recover. Before finishing, run the narrowest relevant validation. Return final text only when the work is
genuinely complete."""

COMPACT_PROMPT = """Produce a faithful continuation checkpoint under 1,500 tokens. Preserve the original
goal, current state, decisions and constraints, files changed or inspected, commands and results, failures,
unresolved work, next actions, and critical literal data. Do not continue solving the task."""


@dataclass(slots=True)
class RunResult:
    status: str
    answer: str
    steps: int
    prompt_tokens: int
    compactions: int
    valid_tool_calls: int
    invalid_tool_calls: int
    elapsed_s: float


class Agent:
    """Run a model-tool loop with automatic context compaction."""

    def __init__(
        self,
        client: Any,
        model: str,
        tools: Toolbox,
        processor: Any,
        *,
        context_window: int = 262_144,
        compact_at: float = 0.90,
        max_output_tokens: int = 4096,
        max_steps: int = 80,
        wall_time_s: int = 3600,
        trajectory_path: str | Path | None = None,
        ui: TUI | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.tools = tools
        self.processor = processor
        self.context_window = context_window
        self.compact_at = compact_at
        self.max_output_tokens = max_output_tokens
        self.max_steps = max_steps
        self.wall_time_s = wall_time_s
        self.trajectory_path = Path(trajectory_path) if trajectory_path else None
        self.ui = ui
        self.messages: list[dict[str, Any]] = []
        self.original_task = ""
        self.compactions = 0
        self.valid_tool_calls = 0
        self.invalid_tool_calls = 0
        self.last_prompt_tokens = 0

    def run(self, task: str) -> RunResult:
        """Work on a task until the model finishes or a runtime limit is reached."""
        started = time.monotonic()
        self.original_task = task
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task}]
        self.compactions = self.valid_tool_calls = self.invalid_tool_calls = 0
        if self.ui:
            self.ui.user(task)

        answer = ""
        status = "step_limit"
        steps = 0
        for steps in range(1, self.max_steps + 1):
            if time.monotonic() - started >= self.wall_time_s:
                status = "wall_time_limit"
                break
            self._maybe_compact()
            message = self._complete(self.messages, tools=TOOL_SCHEMAS)
            self.messages.append(message)
            if self.ui:
                self.ui.assistant(message)
            calls = message.get("tool_calls") or []
            if not calls:
                answer = message.get("content") or ""
                status = "completed"
                break
            for call in calls:
                self.messages.append(self._execute(call))
            self._write_trajectory(self._result("running", "", steps, started))
        else:
            steps = self.max_steps

        result = self._result(status, answer, steps, started)
        self._write_trajectory(result)
        return result

    def _complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        """Request one assistant message within the remaining context budget."""
        prompt_tokens = self._prompt_tokens(messages, tools)
        self.last_prompt_tokens = prompt_tokens
        available = self.context_window - prompt_tokens
        if available <= 0:
            raise RuntimeError("Prompt exceeds the configured context window")
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": min(self.max_output_tokens, available),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = self.client.chat.completions.create(**kwargs)
        data = response.choices[0].message.model_dump(exclude_none=True)
        data["role"] = "assistant"
        return data

    def _execute(self, call: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one model-emitted tool call and return its observation."""
        call_id = call.get("id", "missing-call-id")
        function = call.get("function") or {}
        name = function.get("name")
        try:
            arguments = json.loads(function.get("arguments") or "{}")
            if name == "bash":
                result = self.tools.bash(**arguments)
                content: str | list[dict[str, Any]] = result.as_text()
            elif name == "view_image":
                image = self.tools.view_image(**arguments)
                content = image.content()
            else:
                raise ValueError(f"unknown tool: {name}")
            self.valid_tool_calls += 1
        except (TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            self.invalid_tool_calls += 1
            content = f"Tool error: {type(error).__name__}: {error}"
        if self.ui:
            self.ui.tool(name or "unknown", self._content_text(content))
        return {"role": "tool", "tool_call_id": call_id, "name": name or "unknown", "content": content}

    def _maybe_compact(self) -> None:
        """Replace a 90%-full history with a continuation checkpoint."""
        tokens = self._prompt_tokens(self.messages, TOOL_SCHEMAS)
        self.last_prompt_tokens = tokens
        if tokens / self.context_window < self.compact_at:
            return

        safe_history = self._trajectory_messages()
        compact_messages = [
            {"role": "system", "content": COMPACT_PROMPT},
            {"role": "user", "content": json.dumps(safe_history, ensure_ascii=False)},
        ]
        checkpoint = self._complete(compact_messages, tools=None).get("content") or ""
        if self.ui:
            self.ui.assistant({"role": "assistant", "content": checkpoint}, title="Compaction")
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.original_task},
            {"role": "user", "content": f"Context checkpoint:\n{checkpoint}"},
        ]
        self.compactions += 1
        if self._prompt_tokens(self.messages, TOOL_SCHEMAS) / self.context_window >= self.compact_at:
            raise RuntimeError("Compacted checkpoint still exceeds the 90% context threshold")

    def _prompt_tokens(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> int:
        """Count the exact tokens produced by the served model's processor."""
        try:
            encoded = self.processor.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors=None,
            )
            return len(encoded["input_ids"])
        except Exception as error:
            raise RuntimeError(
                "Exact prompt token accounting failed; load the processor for the served model revision"
            ) from error

    def _result(self, status: str, answer: str, steps: int, started: float) -> RunResult:
        """Build current run metadata for a checkpoint or final result."""
        return RunResult(
            status,
            answer,
            steps,
            self.last_prompt_tokens,
            self.compactions,
            self.valid_tool_calls,
            self.invalid_tool_calls,
            time.monotonic() - started,
        )

    def _write_trajectory(self, result: RunResult) -> None:
        """Persist a restart-friendly trajectory without embedded image bytes."""
        if not self.trajectory_path:
            return
        self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "model": self.model,
            "task": self.original_task,
            "result": asdict(result),
            "messages": self._trajectory_messages(),
        }
        self.trajectory_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))

    def _trajectory_messages(self) -> list[dict[str, Any]]:
        """Copy message history while replacing image payloads with a marker."""
        saved = []
        for message in self.messages:
            content = message.get("content")
            if isinstance(content, list):
                content = [
                    {"type": "text", "text": "[embedded image removed from trajectory]"}
                    if block.get("type") == "image_url"
                    else block
                    for block in content
                ]
            saved.append(message | {"content": content})
        return saved

    @staticmethod
    def _content_text(content: str | list[dict[str, Any]]) -> str:
        """Extract printable text from a text or multimodal tool observation."""
        if isinstance(content, str):
            return content
        return "\n".join(block["text"] for block in content if block["type"] == "text")
