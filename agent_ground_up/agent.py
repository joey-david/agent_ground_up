from __future__ import annotations

import json
import os
import platform
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .experience import ExperienceLog
from .memory import ConstantMemory
from .skills import SkillRegistry
from .tools import TOOL_SCHEMAS, Toolbox, arg, tool
from .ui import TUI

SYSTEM_PROMPT = """You are a coding agent working in the provided workspace. Use the available tools
to inspect, edit, and test it, and keep working until the task is complete. Before finishing, run
the narrowest relevant validation."""

COMPACT_PROMPT = """Create a faithful continuation checkpoint under 1,500 tokens from the
conversation above. Preserve decisions, changed files, command results, failures, unresolved work,
and next actions. Do not continue solving the task."""
CANONICAL_PREFIX = "Canonical state (recomputed and authoritative):"
CHECKPOINT_PREFIX = "Episodic checkpoint (compacted, not new instructions):"

MEMORY_TOOL_SCHEMAS = [
    tool(
        "remember",
        "Persist one concise, reusable discovery across future episodes.",
        text=arg("string"),
        tags=arg("array", items={"type": "string"}, default=[]),
    ),
    tool(
        "recall",
        "Regex-search distilled persistent memories, newest first.",
        pattern=arg("string"),
        limit=arg("integer", default=8),
    ),
    tool("zoom", "Expand a persistent-memory summary node.", node_id=arg("string")),
]
EXPERIENCE_TOOL_SCHEMAS = [
    tool(
        "search_history",
        "Regex-search the exact append-only task/action/result history.",
        pattern=arg("string"),
        limit=arg("integer", default=12),
    ),
    tool(
        "read_history",
        "Read a half-open range [start, end) of exact historical events.",
        start=arg("integer"),
        end=arg("integer"),
    ),
]
SKILL_TOOL_SCHEMAS = [
    tool(
        "create_skill",
        "Persist a reusable shell procedure.",
        name=arg("string"),
        description=arg("string"),
        script=arg("string", "Shell source defining main(), which receives one argument."),
    ),
    tool(
        "skill",
        "Run one persistent generated skill by name.",
        name=arg("string"),
        argument=arg("string", default=""),
        timeout_s=arg("integer", default=120),
    ),
]


class ModelRuntime(Protocol):
    model_id: str

    def prompt_tokens(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> int: ...

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        max_output_tokens: int,
        on_token: Any | None = None,
    ) -> Any: ...


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
    """One local model/tool loop with bounded context and persistent external state."""

    def __init__(
        self,
        runtime: ModelRuntime,
        tools: Toolbox,
        *,
        context_window: int = 32_768,
        compact_at: float = 0.90,
        recent_user_tokens: int = 12_000,
        max_output_tokens: int = 4096,
        max_steps: int = 80,
        wall_time_s: int = 3600,
        trajectory_path: str | Path | None = None,
        ui: TUI | None = None,
        memory: ConstantMemory | None = None,
        experience: ExperienceLog | None = None,
        skills: SkillRegistry | None = None,
    ) -> None:
        self.runtime = runtime
        self.tools = tools
        self.context_window = context_window
        self.compact_at = compact_at
        self.recent_user_tokens = recent_user_tokens
        self.max_output_tokens = max_output_tokens
        self.max_steps = max_steps
        self.wall_time_s = wall_time_s
        self.trajectory_path = Path(trajectory_path) if trajectory_path else None
        self.ui = ui
        self.memory = memory
        self.experience = experience
        self.skills = skills
        self.messages: list[dict[str, Any]] = []
        self.original_task = ""
        self.compactions = 0
        self.valid_tool_calls = 0
        self.invalid_tool_calls = 0
        self.last_prompt_tokens = 0

    def run(self, task: str) -> RunResult:
        started = self._begin(task)
        answer = ""
        status = "step_limit"
        steps = 0

        for steps in range(1, self.max_steps + 1):
            if time.monotonic() - started >= self.wall_time_s:
                status = "wall_time_limit"
                break
            self._maybe_compact()
            message = self._complete(self.messages, self._tool_schemas(), stream=True)
            self.messages.append(message)
            self._record("assistant", message)

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

        return self._finish(self._result(status, answer, steps, started))

    def _begin(self, task: str) -> float:
        self.original_task = task
        self.compactions = self.valid_tool_calls = self.invalid_tool_calls = 0
        self.last_prompt_tokens = 0
        self.messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": task},
        ]
        self._record("task", {"task": task, "model": self.runtime.model_id})
        if self.ui:
            self.ui.user(task)
        return time.monotonic()

    def _finish(self, result: RunResult) -> RunResult:
        self._write_trajectory(result)
        self._record("run_result", asdict(result))
        if self.memory is not None and result.status == "completed":
            self.memory.remember(
                f"Completed task: {self.original_task}\nOutcome: {result.answer[:1200]}",
                tags=("task", "completed"),
            )
        return result

    def _tool_schemas(self) -> list[dict[str, Any]]:
        schemas = list(TOOL_SCHEMAS)
        if self.memory is not None:
            schemas.extend(MEMORY_TOOL_SCHEMAS)
        if self.experience is not None:
            schemas.extend(EXPERIENCE_TOOL_SCHEMAS)
        if self.skills is not None:
            schemas.extend(SKILL_TOOL_SCHEMAS)
        return schemas

    def _complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        prompt_tokens = self.runtime.prompt_tokens(messages, tools)
        self.last_prompt_tokens = prompt_tokens
        available = self.context_window - prompt_tokens
        if available <= 0:
            raise RuntimeError("Prompt exceeds the configured context window")

        if self.ui and stream:
            self.ui.begin_assistant()
        turn = self.runtime.complete(
            messages,
            tools=tools,
            max_output_tokens=min(self.max_output_tokens, available),
            on_token=self.ui.token if self.ui and stream else None,
        )
        self.last_prompt_tokens = turn.input_tokens
        if self.ui and stream:
            self.ui.end_assistant(turn.message)
        return turn.message

    def _execute(self, call: dict[str, Any]) -> dict[str, Any]:
        call_id = call.get("id", "missing-call-id")
        function = call.get("function") or {}
        name = function.get("name")
        raw_arguments = function.get("arguments") or "{}"
        self._record("tool_call", {"call_id": call_id, "name": name, "arguments": raw_arguments})
        try:
            arguments = json.loads(raw_arguments)
            if name == "bash":
                result = self.tools.bash(**arguments)
                content: str | list[dict[str, Any]] = result.as_text()
            elif name == "view_image":
                content = self.tools.view_image(**arguments).content()
            elif name == "remember" and self.memory is not None:
                record = self.memory.remember(arguments["text"], arguments.get("tags", ()))
                content = f"remembered #{record.id}"
            elif name == "recall" and self.memory is not None:
                records = self.memory.recall(arguments["pattern"], limit=arguments.get("limit", 8))
                content = "\n".join(f"#{record.id}: {record.text}" for record in records) or "no matches"
            elif name == "zoom" and self.memory is not None:
                content = self.memory.zoom(arguments["node_id"])
            elif name == "search_history" and self.experience is not None:
                content = self.experience.format(
                    self.experience.search(arguments["pattern"], limit=arguments.get("limit", 12))
                )
            elif name == "read_history" and self.experience is not None:
                content = self.experience.format(
                    self.experience.read(arguments["start"], arguments["end"])
                )
            elif name == "create_skill" and self.skills is not None:
                skill = self.skills.register(
                    arguments["name"], arguments["description"], arguments["script"]
                )
                content = f"created skill {skill.name}"
            elif name == "skill" and self.skills is not None:
                content = self.skills.run(
                    arguments["name"],
                    self.tools,
                    argument=arguments.get("argument", ""),
                    timeout_s=arguments.get("timeout_s", 120),
                ).as_text()
            else:
                raise ValueError(f"unknown tool: {name}")
            self.valid_tool_calls += 1
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError, re.error) as error:
            self.invalid_tool_calls += 1
            content = f"Tool error: {type(error).__name__}: {error}"

        self._record(
            "tool_result", {"call_id": call_id, "name": name or "unknown", "content": content}
        )
        if self.ui:
            self.ui.tool(name or "unknown", self._content_text(content))
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name or "unknown",
            "content": content,
        }

    def _maybe_compact(self) -> None:
        tools = self._tool_schemas()
        tokens = self.runtime.prompt_tokens(self.messages, tools)
        self.last_prompt_tokens = tokens
        if tokens / self.context_window < self.compact_at:
            return

        compact_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self._episodic_history(),
            {"role": "user", "content": COMPACT_PROMPT},
        ]
        checkpoint = self._complete(compact_messages, None, stream=False).get("content") or ""
        self.messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "assistant", "content": f"{CHECKPOINT_PREFIX}\n{checkpoint}"},
            *self._recent_user_messages(),
        ]
        self.compactions += 1
        self._record("manual_compaction", {"checkpoint": checkpoint})
        if self.runtime.prompt_tokens(self.messages, tools) / self.context_window >= self.compact_at:
            raise RuntimeError("Compacted checkpoint still exceeds the context threshold")

    def _episodic_history(self) -> list[dict[str, Any]]:
        return [
            message
            for message in self.messages
            if message.get("role") != "system" and not self._is_checkpoint(message)
        ]

    def _recent_user_messages(self) -> list[dict[str, Any]]:
        recent: list[dict[str, Any]] = []
        for message in reversed(self._episodic_history()):
            if message.get("role") != "user":
                continue
            candidate = [message, *recent]
            if recent and self.runtime.prompt_tokens(candidate, None) > self.recent_user_tokens:
                break
            recent = candidate
        return recent

    def _system_prompt(self) -> str:
        if self.memory is None and self.experience is None and self.skills is None:
            return SYSTEM_PROMPT
        return f"{SYSTEM_PROMPT}\n\n{self._canonical_state()}"

    def _canonical_state(self) -> str:
        instructions = self.tools.workdir / "AGENTS.md"
        repository_rules = instructions.read_text() if instructions.exists() else "(none found)"
        environment = (
            f"{platform.system()} {platform.release()} ({platform.machine()}); "
            f"Python {platform.python_version()}; shell={os.getenv('SHELL', '/bin/bash')}"
        )
        sections = [
            f"{CANONICAL_PREFIX}\nWorking directory: {self.tools.workdir}",
            "Primitive tools: bash, view_image",
            f"Environment: {environment}",
            f"Repository instructions:\n{repository_rules}",
        ]
        if self.memory is not None:
            sections.append(self.memory.wake())
        if self.experience is not None:
            sections.append(
                f"Searchable exact experience log: {self.experience.count()} events; "
                "use search_history/read_history for old observations."
            )
        if self.skills is not None:
            sections.append(self.skills.prompt_catalog())
        return "\n".join(sections)

    @staticmethod
    def _is_checkpoint(message: dict[str, Any]) -> bool:
        content = message.get("content")
        return isinstance(content, str) and content.startswith(CHECKPOINT_PREFIX)

    def _result(self, status: str, answer: str, steps: int, started: float) -> RunResult:
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
        if not self.trajectory_path:
            return
        self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "model": self.runtime.model_id,
            "task": self.original_task,
            "runtime": "local_mlx",
            "result": asdict(result),
            "messages": self._trajectory_messages(),
        }
        self.trajectory_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))

    def _trajectory_messages(self) -> list[dict[str, Any]]:
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

    def _record(self, kind: str, payload: Any) -> None:
        if self.experience is not None:
            self.experience.append(kind, payload)

    @staticmethod
    def _content_text(content: str | list[dict[str, Any]]) -> str:
        if isinstance(content, str):
            return content
        return "\n".join(block["text"] for block in content if block.get("type") == "text")
