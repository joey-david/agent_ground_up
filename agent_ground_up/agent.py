from __future__ import annotations

import json
import os
import platform
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .experience import ExperienceLog
from .memory import ConstantMemory, MemoryRecord
from .runtime import ContinuousResponsesRuntime
from .skills import SkillRegistry
from .tools import TOOL_SCHEMAS, Toolbox, arg, tool
from .ui import TUI

SYSTEM_PROMPT = """You are a coding agent working in the provided workspace. Work until the task is
complete. Before finishing, run the narrowest relevant validation. Return final text only when the
work is genuinely complete."""

COMPACT_PROMPT = """Create a continuation checkpoint under 1,500 tokens from the conversation
above. Do not continue solving the task."""
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
    tool(
        "zoom",
        "Expand a persistent-memory summary node into finer summaries or raw memories.",
        node_id=arg("string"),
    ),
]

EXPERIENCE_TOOL_SCHEMAS = [
    tool(
        "search_history",
        "Regex-search the exact append-only task/action/result history, newest first.",
        pattern=arg("string"),
        limit=arg("integer", default=12),
    ),
    tool(
        "read_history",
        "Read a half-open range [start, end) of exact historical events by ID.",
        start=arg("integer"),
        end=arg("integer"),
    ),
]

SKILL_TOOL_SCHEMAS = [
    tool(
        "create_skill",
        "Persist a reusable shell procedure as a generated skill.",
        name=arg("string"),
        description=arg("string"),
        script=arg("string", "Shell source defining main(), which receives one argument."),
    ),
    tool(
        "skill",
        "Run one persistent generated skill by name inside the workspace.",
        name=arg("string"),
        argument=arg("string", default=""),
        timeout_s=arg("integer", default=120),
    ),
]

ALL_TOOL_SCHEMAS = (
    *TOOL_SCHEMAS,
    *MEMORY_TOOL_SCHEMAS,
    *EXPERIENCE_TOOL_SCHEMAS,
    *SKILL_TOOL_SCHEMAS,
)
SCHEMAS = {schema["function"]["name"]: schema for schema in ALL_TOOL_SCHEMAS}


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
    """Model-tool loop with native continuous state, external memory, and generated skills.

    `runtime=None` preserves the original OpenAI-compatible Chat Completions path used by local
    vLLM. Supplying `ContinuousResponsesRuntime` switches the same agent/tool loop to stateless
    Responses replay with encrypted reasoning and provider-native compaction.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        tools: Toolbox,
        processor: Any | None,
        *,
        context_window: int = 262_144,
        compact_at: float = 0.90,
        recent_user_tokens: int = 12_000,
        max_output_tokens: int = 4096,
        max_steps: int = 80,
        wall_time_s: int = 3600,
        system_prompt: str = SYSTEM_PROMPT,
        compact_prompt: str = COMPACT_PROMPT,
        temperature: float | None = None,
        top_p: float | None = None,
        trajectory_path: str | Path | None = None,
        ui: TUI | None = None,
        memory: ConstantMemory | None = None,
        experience: ExperienceLog | None = None,
        skills: SkillRegistry | None = None,
        runtime: ContinuousResponsesRuntime | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.tools = tools
        self.processor = processor
        self.context_window = context_window
        self.compact_at = compact_at
        self.recent_user_tokens = recent_user_tokens
        self.max_output_tokens = max_output_tokens
        self.max_steps = max_steps
        self.wall_time_s = wall_time_s
        self.system_prompt = system_prompt
        self.compact_prompt = compact_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.trajectory_path = Path(trajectory_path) if trajectory_path else None
        self.ui = ui
        self.memory = memory
        self.experience = experience
        self.skills = skills
        self.runtime = runtime
        self.messages: list[dict[str, Any]] = []
        self.original_task = ""
        self.compactions = 0
        self.valid_tool_calls = 0
        self.invalid_tool_calls = 0
        self.last_prompt_tokens = 0
        self.handlers = self._build_handlers()

    def run(self, task: str) -> RunResult:
        """Work on a task until the model finishes or a runtime limit is reached."""
        started = self._begin(task)
        return self._finish(self._run(started))

    def _begin(self, task: str) -> float:
        started = time.monotonic()
        self.original_task = task
        self.messages = [{"role": "system", "content": self.system_prompt}]
        if self.memory is not None or self.experience is not None or self.skills is not None:
            self.messages.append({"role": "system", "content": self._canonical_state()})
        self.messages.append({"role": "user", "content": task})
        self.compactions = self.valid_tool_calls = self.invalid_tool_calls = 0
        self.last_prompt_tokens = 0
        if self.runtime is not None:
            self.runtime.reset(task)
        self._record("task", {"task": task, "model": self.model})
        if self.ui:
            self.ui.user(task)
        return started

    def _run(self, started: float) -> RunResult:
        """The loop, identical for both runtimes: think, act on every tool call, repeat."""
        answer = ""
        status = "step_limit"
        steps = 0
        for steps in range(1, self.max_steps + 1):
            if time.monotonic() - started >= self.wall_time_s:
                status = "wall_time_limit"
                break
            message = self._turn()
            self.messages.append(message)
            self._record("assistant", message)
            if self.ui:
                self.ui.assistant(message)
            calls = message.get("tool_calls") or []
            if not calls:
                answer = message.get("content") or ""
                status = "completed"
                break
            for call in calls:
                observation = self._execute(call)
                self.messages.append(observation)
                if self.runtime is not None:
                    self.runtime.submit_tool_output(
                        call_id=observation["tool_call_id"],
                        name=observation["name"],
                        content=observation["content"],
                    )
            self._write_trajectory(self._result("running", "", steps, started))
        else:
            steps = self.max_steps
        return self._result(status, answer, steps, started)

    def _turn(self) -> dict[str, Any]:
        """Ask for one assistant message, and keep context bounded the way this runtime does.

        The chat path owns its context: it compacts locally into a written checkpoint before
        asking. The continuous path hands that job to the provider and just reports back what
        the turn cost.
        """
        if self.runtime is None:
            self._maybe_compact()
            return self._complete(self.messages, tools=self._tool_schemas())
        turn = self.runtime.complete(
            instructions=self._runtime_instructions(),
            tools=self._tool_schemas(),
            max_output_tokens=self.max_output_tokens,
        )
        self.last_prompt_tokens = turn.input_tokens
        self.compactions = turn.compactions
        return turn.message

    def _finish(self, result: RunResult) -> RunResult:
        self._write_trajectory(result)
        self._record("run_result", asdict(result))
        if self.memory is not None and result.status == "completed":
            self.memory.remember(
                f"Completed task: {self.original_task}\nOutcome: {result.answer[:1200]}",
                tags=("task", "completed"),
            )
        return result

    def _build_handlers(self) -> dict[str, Callable[..., Any]]:
        """Bind every tool this agent can actually run to the call that answers it.

        This table is the single source of truth for both halves of a tool: `_tool_schemas`
        advertises exactly these names to the model and `_execute` dispatches exactly these
        names, so the two can never drift. Handlers forward keyword arguments untouched, which
        leaves each argument's real default in one place — the subsystem being called.
        """
        handlers: dict[str, Callable[..., Any]] = {
            "bash": lambda **kwargs: self.tools.bash(**kwargs).as_text(),
            "view_image": lambda **kwargs: self.tools.view_image(**kwargs).content(),
        }
        if (memory := self.memory) is not None:
            handlers |= {
                "remember": lambda **kwargs: f"remembered #{memory.remember(**kwargs).id}",
                "recall": lambda **kwargs: self._format_memories(memory.recall(**kwargs)),
                "zoom": lambda **kwargs: memory.zoom(**kwargs),
            }
        if (experience := self.experience) is not None:
            handlers |= {
                "search_history": lambda **kwargs: experience.format(experience.search(**kwargs)),
                "read_history": lambda **kwargs: experience.format(experience.read(**kwargs)),
            }
        if (skills := self.skills) is not None:
            handlers |= {
                "create_skill": lambda **kwargs: f"created skill {skills.register(**kwargs).name}",
                "skill": lambda **kwargs: skills.run(runner=self.tools, **kwargs).as_text(),
            }
        return handlers

    def _tool_schemas(self) -> list[dict[str, Any]]:
        return [SCHEMAS[name] for name in self.handlers]

    @staticmethod
    def _collapse_leading_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Join the leading system messages into one before anything leaves the agent.

        The kernel keeps the base prompt and the canonical state as two separate system
        messages so that `_is_compaction_message` can recognise the canonical block and drop
        it from episodic history. Qwen chat templates reject any system message that is not
        the very first one, and vLLM renders that same template server-side, so both the
        token count and the request itself must see a single merged block.
        """
        leading: list[str] = []
        rest_index = 0
        for index, message in enumerate(messages):
            if message.get("role") != "system" or not isinstance(message.get("content"), str):
                rest_index = index
                break
            leading.append(message["content"])
        else:
            rest_index = len(messages)
        if len(leading) < 2:
            return messages
        merged = {"role": "system", "content": "\n\n".join(leading)}
        return [merged, *messages[rest_index:]]

    def _complete(
        self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        """Request one Chat Completions assistant message within the remaining context budget."""
        wire_messages = self._collapse_leading_system(messages)
        prompt_tokens = self._prompt_tokens(messages, tools)
        self.last_prompt_tokens = prompt_tokens
        available = self.context_window - prompt_tokens
        if available <= 0:
            raise RuntimeError("Prompt exceeds the configured context window")
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": wire_messages,
            "max_tokens": min(self.max_output_tokens, available),
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = self.client.chat.completions.create(**kwargs)
        data = response.choices[0].message.model_dump(exclude_none=True)
        data["role"] = "assistant"
        return data

    def _execute(self, call: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one model-emitted tool call and return its observation.

        Every failure the model can cause — bad JSON, a missing argument, a bad regex, an
        unreadable file — comes back as a readable observation rather than an exception, so a
        wrong call costs the agent one step instead of the whole run.
        """
        call_id = call.get("id", "missing-call-id")
        function = call.get("function") or {}
        name = function.get("name")
        raw_arguments = function.get("arguments") or "{}"
        self._record("tool_call", {"call_id": call_id, "name": name, "arguments": raw_arguments})
        content: str | list[dict[str, Any]]
        try:
            handler = self.handlers.get(name or "")
            if handler is None:
                raise ValueError(f"unknown tool: {name}")
            content = handler(**json.loads(raw_arguments))
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

    @staticmethod
    def _format_memories(records: list[MemoryRecord]) -> str:
        return "\n".join(f"#{record.id}: {record.text}" for record in records) or "no matches"

    def _maybe_compact(self) -> None:
        """Chat path only: replace a nearly-full history with a continuation checkpoint."""
        tools = self._tool_schemas()
        tokens = self._prompt_tokens(self.messages, tools)
        self.last_prompt_tokens = tokens
        if self._has_room(tokens):
            return

        checkpoint = self._complete(self._compaction_request(), tools=None).get("content") or ""
        if self.ui:
            self.ui.assistant({"role": "assistant", "content": checkpoint}, title="Compaction")
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": self._canonical_state()},
            {"role": "assistant", "content": f"{CHECKPOINT_PREFIX}\n{checkpoint}"},
            *self._recent_user_messages(),
        ]
        self.compactions += 1
        self._record("manual_compaction", {"checkpoint": checkpoint})
        if not self._has_room(self._prompt_tokens(self.messages, tools)):
            raise RuntimeError("Compacted checkpoint still exceeds the context threshold")

    def _has_room(self, tokens: int) -> bool:
        """Whether a prompt of this size can still be answered without compacting first.

        Two conditions, not one. The ratio is the ordinary trigger, but a single step can add
        several large observations at once and clear the threshold in one jump, so the request
        must also retain enough room to generate a reply into. Checking only the ratio leaves a
        window where the next completion has no output budget and fails outright.
        """
        below_threshold = tokens / self.context_window < self.compact_at
        # Cap the reserve at a quarter of the window: a configuration whose output budget rivals
        # its context would otherwise demand compaction on every step and never make progress.
        reserve = min(self.max_output_tokens, self.context_window // 4)
        return below_threshold and self.context_window - tokens >= reserve

    def _compaction_request(self) -> list[dict[str, Any]]:
        """Episodic history plus the compaction instruction, trimmed to leave room for the reply.

        The checkpoint is written *from* this request, so the request has to fit the window and
        still leave space to generate into. Without the trim, a history that has grown past the
        window makes the one call that could rescue it the call that fails. Surplus history is
        dropped oldest-first: the canonical state is reinjected separately, and a continuation
        needs the newest turns. A leading system message is kept, and any `tool` message left
        without the assistant turn that requested it is dropped so the wire format stays valid.
        The newest turn always survives, so a window too small to hold even that still produces a
        well-formed request rather than an empty one.
        """
        instruction = {"role": "user", "content": self.compact_prompt}
        history = self._episodic_history()
        head = history[:1] if history and history[0].get("role") == "system" else []
        body = history[len(head) :]
        budget = self.context_window - self.max_output_tokens
        while len(body) > 1 and (
            body[0].get("role") == "tool"
            or self._prompt_tokens([*head, *body, instruction], None) > budget
        ):
            body = body[1:]
        return [*head, *body, instruction]

    def _episodic_history(self) -> list[dict[str, Any]]:
        return [message for message in self.messages if not self._is_compaction_message(message)]

    def _recent_user_messages(self) -> list[dict[str, Any]]:
        recent: list[dict[str, Any]] = []
        for message in reversed(self._episodic_history()):
            if message.get("role") != "user":
                continue
            candidate = [message, *recent]
            if recent and self._prompt_tokens(candidate, None) > self.recent_user_tokens:
                break
            recent = candidate
        return recent

    def _runtime_instructions(self) -> str:
        return f"{self.system_prompt}\n\n{self._canonical_state()}"

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
            sections.append(
                f"{self.memory.wake()}\nRemember only reusable discoveries; "
                "prefer recall/zoom over restating old history."
            )
        if self.experience is not None:
            sections.append(
                f"Searchable exact experience log: {self.experience.count()} events; "
                "use search_history/read_history for old observations and tool results."
            )
        if self.skills is not None:
            sections.append(
                f"{self.skills.prompt_catalog()}\nPrefer a reliable existing skill over "
                "re-deriving the same procedure."
            )
        return "\n".join(sections)

    @staticmethod
    def _is_compaction_message(message: dict[str, Any]) -> bool:
        content = message.get("content")
        return isinstance(content, str) and content.startswith(
            (CANONICAL_PREFIX, CHECKPOINT_PREFIX)
        )

    def _prompt_tokens(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> int:
        if self.processor is None:
            raise RuntimeError("Chat Completions runtime requires the served model's processor")
        try:
            processor_messages = []
            for message in self._collapse_leading_system(messages):
                calls = []
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    if not isinstance(arguments, dict):
                        arguments = {}
                    calls.append(call | {"function": function | {"arguments": arguments}})
                processor_messages.append(message | ({"tool_calls": calls} if calls else {}))
            encoded = self.processor.apply_chat_template(
                processor_messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors=None,
            )
            input_ids = encoded["input_ids"]
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            return len(input_ids)
        except Exception as error:
            raise RuntimeError(
                "Exact prompt token accounting failed; load the processor for the served model revision"
            ) from error

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
            "model": self.model,
            "task": self.original_task,
            "runtime": "responses_continuous" if self.runtime is not None else "chat_completions",
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
