"""Compare this repository's agent kernel against a deliberately minimal harness.

Three arms answer three different questions, all on the same benchmark, the same model and the
same limits, so the only variable is the harness:

    kernel       the full agent: native tool calls, persistent memory, append-only experience,
                 generated skills, and local compaction when the window fills
    kernel-bare  the same loop and the same tool schemas with every subsystem removed -- the
                 scaffolding's contribution, isolated from the loop's
    mini         a faithful reimplementation of the mini-swe-agent protocol: no tool API at
                 all, one bash command per turn parsed out of a fenced block, linear history,
                 no truncation and no compaction
    oneshot      no agent at all: one multimodal completion holding the task and, for an image
                 task, the picture -- the floor a harness has to beat to justify its existence

Every episode is independent: a fresh copy of the case workspace, fresh subsystem state, and a
verifier whose exit code is the only score.

    API_KEY=EMPTY HF_HUB_OFFLINE=1 uv run python scripts/harness_ab.py --split hard2 --repeats 2
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from openai import OpenAI
from transformers import AutoProcessor

from agent_ground_up.agent import Agent
from agent_ground_up.config import DEFAULT_CONFIG, load_config, section
from agent_ground_up.experience import ExperienceLog
from agent_ground_up.memory import ConstantMemory
from agent_ground_up.skills import SkillRegistry
from agent_ground_up.tasks import load_families
from agent_ground_up.tools import Toolbox

DEFAULT_ENDPOINTS = "http://127.0.0.1:8020/v1"
ARMS = ("kernel", "kernel-bare", "mini", "oneshot")

# The mini-swe-agent contract, kept deliberately close to the original: the model has no tool
# API, so the protocol lives entirely in the prompt, and a turn that does not obey it is
# answered with a format complaint rather than an execution.
MINI_SYSTEM_PROMPT = """You are a helpful assistant that can interact with a computer.

Your response must contain exactly ONE bash code block with ONE command (or commands connected
with && or ||). Include a THOUGHT section before your command where you explain your reasoning.

Format your response as shown below:

THOUGHT: your reasoning here

```bash
your_command_here
```

Failure to follow these rules will cause your response to be rejected."""

MINI_INSTANCE_TEMPLATE = """Please solve this task in the current directory:

{task}

## Recommended workflow

1. Look at the files in the working directory to understand the situation.
2. Make the smallest edit that solves the task, writing files with a heredoc.
3. Run the relevant check yourself before you submit.

## Important rules

1. Every response must contain exactly one bash code block with exactly one command.
2. Non-interactive commands only: no vim, no nano, no pagers, no commands that wait for input.
3. The working directory is fixed; `cd` does not persist between commands.
4. When you have solved the task, run exactly this command and nothing else:

```bash
echo {sentinel}
```"""

MINI_SENTINEL = "MINI_AGENT_FINAL_OUTPUT"
MINI_FORMAT_ERROR = """Your last response did not contain exactly one bash code block.
Respond with a THOUGHT section and exactly one ```bash block containing exactly one command."""

BASH_BLOCK = re.compile(r"```bash\s*\n(.*?)\n?```", re.DOTALL)


@dataclass(slots=True)
class Outcome:
    arm: str
    case_id: str
    family: str
    repeat: int
    episode_index: int
    passed: bool
    status: str
    steps: int
    prompt_tokens: int
    valid_tool_calls: int
    invalid_tool_calls: int
    elapsed_s: float
    compactions: int = 0
    detail: str = ""
    tools_used: dict[str, int] | None = None


@dataclass(slots=True)
class MiniResult:
    status: str
    steps: int
    prompt_tokens: int
    valid_tool_calls: int
    invalid_tool_calls: int


class MiniAgent:
    """The dumb baseline: one bash command per turn, parsed out of the model's own text.

    There is no tool schema, no state beyond the transcript, and no recovery when the window
    fills -- overflowing it ends the episode, which is exactly the failure mode the kernel's
    compaction exists to prevent.
    """

    def __init__(
        self,
        client,
        model: str,
        workdir: Path,
        processor,
        *,
        context_window: int,
        max_output_tokens: int,
        max_steps: int,
        wall_time_s: int,
        temperature: float | None,
        top_p: float | None,
        command_timeout_s: int = 120,
    ) -> None:
        self.client = client
        self.model = model
        self.workdir = workdir
        self.processor = processor
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.max_steps = max_steps
        self.wall_time_s = wall_time_s
        self.temperature = temperature
        self.top_p = top_p
        self.command_timeout_s = command_timeout_s
        self.messages: list[dict[str, str]] = []
        self.last_prompt_tokens = 0

    def run(self, task: str) -> MiniResult:
        started = time.monotonic()
        self.messages = [
            {"role": "system", "content": MINI_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": MINI_INSTANCE_TEMPLATE.format(task=task, sentinel=MINI_SENTINEL),
            },
        ]
        status, valid, invalid = "step_limit", 0, 0
        steps = 0
        for steps in range(1, self.max_steps + 1):
            if time.monotonic() - started >= self.wall_time_s:
                status = "wall_time_limit"
                break
            content = self._complete()
            self.messages.append({"role": "assistant", "content": content})
            blocks = BASH_BLOCK.findall(content)
            if len(blocks) != 1:
                invalid += 1
                self.messages.append({"role": "user", "content": MINI_FORMAT_ERROR})
                continue
            command = blocks[0].strip()
            valid += 1
            if MINI_SENTINEL in command:
                status = "completed"
                break
            self.messages.append({"role": "user", "content": self._observe(command)})
        else:
            steps = self.max_steps
        return MiniResult(status, steps, self.last_prompt_tokens, valid, invalid)

    def _complete(self) -> str:
        """Ask for one message, billed against the same window the kernel is held to.

        The kernel compacts when this budget runs low; the minimal harness has nothing to do
        about it, so the episode ends here. That asymmetry is the point of the comparison, but
        it is only meaningful if both arms are measured against the same window.
        """
        encoded = self.processor.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        prompt_tokens = len(input_ids)
        self.last_prompt_tokens = max(self.last_prompt_tokens, prompt_tokens)
        available = self.context_window - prompt_tokens
        if available <= 0:
            raise RuntimeError("Prompt exceeds the configured context window")
        kwargs = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": min(self.max_output_tokens, available),
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _observe(self, command: str) -> str:
        """Run one command and report it back in mini-swe-agent's tagged form, untruncated."""
        try:
            completed = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=self.command_timeout_s,
            )
            output = completed.stdout + completed.stderr
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            output, returncode = f"command timed out after {self.command_timeout_s}s", 124
        return f"<returncode>{returncode}</returncode>\n<output>\n{output}</output>"


ONESHOT_SYSTEM_PROMPT = """You are answering a question about a workspace you cannot interact
with. Everything you are given is in this message. Answer directly and concisely."""


@dataclass(slots=True)
class OneShotResult:
    status: str
    answer: str
    steps: int
    prompt_tokens: int
    valid_tool_calls: int
    invalid_tool_calls: int


class OneShotAgent:
    """The no-harness floor: a single completion, then whatever it said is the answer.

    It is given the same task text and the same picture the kernel's image tool would produce,
    so the difference between this arm and the kernel is the loop -- being able to look again,
    crop, enlarge, and check -- rather than what the model was allowed to see once.
    """

    def __init__(
        self,
        client,
        model: str,
        workdir: Path,
        toolbox: Toolbox,
        *,
        max_output_tokens: int,
        temperature: float | None,
        top_p: float | None,
    ) -> None:
        self.client = client
        self.model = model
        self.workdir = workdir
        self.toolbox = toolbox
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.messages: list[dict[str, Any]] = []
        self.last_prompt_tokens = 0

    def run(self, task: str) -> OneShotResult:
        content: list[dict[str, Any]] = [{"type": "text", "text": self._task_text(task)}]
        for image in sorted(self.workdir.glob("*.jpg")) + sorted(self.workdir.glob("*.png")):
            content.extend(self.toolbox.view_image(image.name).content())
        self.messages = [
            {"role": "system", "content": ONESHOT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        response = self.client.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_prompt_tokens = usage.prompt_tokens
        answer = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": answer})
        # There is no shell in this arm, so the harness writes the file the task asks for and
        # the model's single reply is what gets graded.
        self._write_answer(answer)
        return OneShotResult(
            "completed" if answer.strip() else "empty_completion",
            answer,
            1,
            self.last_prompt_tokens,
            0,
            0,
        )

    def _task_text(self, task: str) -> str:
        """Inline the workspace's text files: this arm has no way to open them itself."""
        parts = [task, ""]
        for path in sorted(self.workdir.rglob("*")):
            if path.is_file() and path.suffix in {".md", ".txt"} and path.name != "answer.txt":
                parts.append(f"--- {path.relative_to(self.workdir)} ---")
                parts.append(path.read_text(encoding="utf-8", errors="replace")[:20000])
        return "\n".join(parts)

    def _write_answer(self, answer: str) -> None:
        (self.workdir / "answer.txt").write_text(answer.strip(), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", default="benchmarks/coding/curriculum.json")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", default="heldout", help="comma-separated splits")
    parser.add_argument("--arms", default=",".join(ARMS), help="comma-separated arms to run")
    parser.add_argument("--cases", default="", help="comma-separated case ids to restrict to")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--endpoints", default=DEFAULT_ENDPOINTS)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--wall-time-s", type=int, default=420)
    parser.add_argument("--context-window", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="a reasoning model spends this on thought before it emits a call")
    parser.add_argument("--max-tool-output-tokens", type=int, default=None,
                        help="cap tool output; must stay well under a shrunken context window")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--out", default="runs/harness_ab.json")
    parser.add_argument("--verifier-timeout-s", type=int, default=300,
                        help="a hidden-test verifier runs many cases and outlives a unit test")
    parser.add_argument("--persistent-state", default="",
                        help="carry memory/experience/skills across episodes, in a fixed order, "
                             "so an arm can learn from the cases it has already seen")
    parser.add_argument("--transcripts", default="",
                        help="directory to write one message transcript per episode")
    return parser.parse_args()


def run_case(
    *,
    arm: str,
    case,
    family: str,
    repeat: int,
    client: OpenAI,
    processor,
    model: str,
    config,
    agent_config,
    repository_root: Path,
    args: argparse.Namespace,
    episode_index: int = 0,
    state_root: Path | None = None,
) -> Outcome:
    started = time.monotonic()
    context_window = args.context_window or section(config, "model")["context_window"]
    with tempfile.TemporaryDirectory(prefix=f"ab-{arm}-{case.id}-") as temp:
        root = Path(temp)
        workdir = root / "workspace"
        shutil.copytree(repository_root / case.workspace, workdir)
        prompts = [case.prompt] + ([case.followup] if case.followup else [])
        status, steps, tokens, valid, invalid, detail = "completed", 0, 0, 0, 0, ""
        tools_used: dict[str, int] = {}
        compactions = 0

        if arm == "oneshot":
            agent = OneShotAgent(
                client,
                model,
                workdir,
                Toolbox(
                    workdir,
                    max_output_tokens=args.max_tool_output_tokens
                    or agent_config["max_tool_output_tokens"],
                    patch_size=section(config, "model").get("patch_size", 32),
                    token_counter=lambda text: len(
                        processor.tokenizer.encode(text, add_special_tokens=False)
                    ),
                ),
                max_output_tokens=args.max_output_tokens
                or agent_config["max_output_tokens"],
                temperature=args.temperature,
                top_p=args.top_p,
            )
        elif arm == "mini":
            agent = MiniAgent(
                client,
                model,
                workdir,
                processor,
                context_window=context_window,
                max_output_tokens=args.max_output_tokens
                or agent_config["max_output_tokens"],
                max_steps=args.max_steps,
                wall_time_s=args.wall_time_s,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        else:
            toolbox = Toolbox(
                workdir,
                max_output_tokens=args.max_tool_output_tokens
                or agent_config["max_tool_output_tokens"],
                patch_size=section(config, "model").get("patch_size", 32),
                token_counter=lambda text: len(
                    processor.tokenizer.encode(text, add_special_tokens=False)
                ),
            )
            bare = arm == "kernel-bare"
            subsystem_root = state_root if state_root is not None else root
            agent = Agent(
                client,
                model,
                toolbox,
                processor,
                context_window=context_window,
                compact_at=agent_config["compact_at"],
                recent_user_tokens=agent_config["recent_user_tokens"],
                max_output_tokens=args.max_output_tokens
                or agent_config["max_output_tokens"],
                max_steps=args.max_steps,
                wall_time_s=args.wall_time_s,
                temperature=args.temperature,
                top_p=args.top_p,
                # With --persistent-state these directories outlive the episode, so what the
                # agent remembered, logged and built is still there for the next case.
                memory=None if bare else ConstantMemory(subsystem_root / "memory"),
                experience=None if bare else ExperienceLog(subsystem_root / "experience"),
                skills=None if bare else SkillRegistry(subsystem_root / "skills"),
            )

        try:
            for episode_prompt in prompts:
                result = agent.run(episode_prompt)
                status = result.status
                # A reasoning model that spends its whole output budget on thought returns an
                # empty message with no tool calls, which the loop cannot distinguish from a
                # deliberate finish. Scoring that as "completed" hides the real failure.
                # `mini` cannot hit this: it only completes on an explicit sentinel command,
                # and an empty message there is just a format error it retries.
                if arm not in {"mini", "oneshot"} and status == "completed" and not (result.answer or "").strip():
                    status = "empty_completion"
                steps += result.steps
                tokens = max(tokens, result.prompt_tokens)
                valid += result.valid_tool_calls
                invalid += result.invalid_tool_calls
                compactions += getattr(result, "compactions", 0)
                for message in getattr(agent, "messages", []):
                    for call in message.get("tool_calls") or []:
                        name = (call.get("function") or {}).get("name", "?")
                        tools_used[name] = tools_used.get(name, 0) + 1
        except Exception as error:  # a crashed episode is a failed episode, not a dead sweep
            frames = traceback.extract_tb(error.__traceback__)
            where = " <- ".join(f"{f.name}:{f.lineno}" for f in frames[-4:])
            # The dumb harness has no answer to a full window, so it fails here rather than
            # inside compaction; keeping the distinction makes the comparison readable.
            overflow = "context" in str(error).lower() or "maximum" in str(error).lower()
            status = "context_overflow" if overflow else "error"
            detail = f"{type(error).__name__}: {error} | {where} | steps={steps}"[:400]

        if args.transcripts:
            # The arms fail for different reasons, so the transcript is usually the only way to
            # tell a wrong answer from a harness that never let the model see the evidence.
            directory = Path(args.transcripts)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{arm}-{case.id}-{repeat}.json").write_text(
                json.dumps(getattr(agent, "messages", []), indent=2, default=str) + "\n"
            )

        passed = False
        if status not in {"error", "context_overflow"}:
            try:
                check = subprocess.run(
                    case.verifier,
                    shell=True,
                    executable="/bin/bash",
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    timeout=args.verifier_timeout_s,
                )
            except subprocess.TimeoutExpired:
                # One stuck verifier is a failed episode; it must not take the sweep with it.
                status, detail = "verifier_timeout", case.verifier[:200]
            else:
                passed = check.returncode == 0
                if not passed:
                    detail = (check.stdout + check.stderr).strip()[-300:]

    return Outcome(
        arm=arm,
        case_id=case.id,
        family=family,
        repeat=repeat,
        episode_index=episode_index,
        passed=passed,
        status=status,
        steps=steps,
        prompt_tokens=tokens,
        valid_tool_calls=valid,
        invalid_tool_calls=invalid,
        elapsed_s=time.monotonic() - started,
        compactions=compactions,
        detail=detail,
        tools_used=tools_used,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_config = section(config, "model")
    agent_config = section(config, "agent")
    repository_root = config["_root"]

    arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in arms if name not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from {list(ARMS)}")

    families = load_families(args.curriculum)
    splits = [name.strip() for name in args.split.split(",") if name.strip()]
    cases = [
        (family.name, case)
        for family in families
        for split in splits
        for case in family.by_split(split)
    ]
    if args.cases:
        wanted = {cid.strip() for cid in args.cases.split(",")}
        cases = [(f, c) for f, c in cases if c.id in wanted]
    if not cases:
        raise SystemExit(f"no cases in split {args.split!r}")

    processor = AutoProcessor.from_pretrained(model_config["processor"], trust_remote_code=False)
    endpoints = [url.strip() for url in args.endpoints.split(",") if url.strip()]
    clients = [OpenAI(base_url=url, api_key="EMPTY", timeout=600.0) for url in endpoints]
    print(f"{len(arms)} arms x {len(cases)} cases x {args.repeats} repeats "
          f"= {len(arms) * len(cases) * args.repeats} episodes over {len(endpoints)} endpoints")

    jobs = [
        (arm, family, case, repeat)
        for arm in arms
        for family, case in cases
        for repeat in range(args.repeats)
    ]
    counter = itertools.count()
    lock = threading.Lock()
    done = [0]
    finished: list[Outcome] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def work(job) -> Outcome:
        index, (arm, family, case, repeat) = job
        with lock:
            client = clients[next(counter) % len(clients)]
        state_root = None
        if args.persistent_state:
            # One state directory per arm: the arms must learn separately or the comparison
            # measures one shared brain rather than each harness's own.
            state_root = Path(args.persistent_state) / arm
            state_root.mkdir(parents=True, exist_ok=True)
        outcome = run_case(
            arm=arm,
            case=case,
            family=family,
            repeat=repeat,
            episode_index=index,
            state_root=state_root,
            client=client,
            processor=processor,
            model=model_config["served_name"],
            config=config,
            agent_config=agent_config,
            repository_root=repository_root,
            args=args,
        )
        with lock:
            done[0] += 1
            # Persist after every episode: losing a long sweep to an interrupted process costs
            # more than rewriting a small JSON file each time.
            finished.append(outcome)
            out_path.write_text(json.dumps([asdict(o) for o in finished], indent=2) + "\n")
            mark = "PASS" if outcome.passed else "FAIL"
            print(f"[{done[0]:>3}/{len(jobs)}] {mark} {arm:<12} {outcome.case_id:<14} "
                  f"steps={outcome.steps:<3} {outcome.elapsed_s:6.1f}s {outcome.status}",
                  flush=True)
        return outcome

    # Accumulated state only means anything if the episodes happen in order, so persistence
    # forces the sweep to run one case at a time.
    workers = 1 if args.persistent_state else args.concurrency
    if args.persistent_state:
        jobs.sort(key=lambda job: (job[0], job[3]))
    # Number within the arm, not across the run: the question is whether an arm improves over
    # the cases *it* has seen.
    seen: Counter[str] = Counter()
    numbered = []
    for job in jobs:
        numbered.append((seen[job[0]], job))
        seen[job[0]] += 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(work, numbered))

    out_path.write_text(json.dumps([asdict(o) for o in outcomes], indent=2) + "\n")
    report(outcomes, arms)
    print(f"\nwrote {out_path}")


def report(outcomes: list[Outcome], arms: list[str]) -> None:
    print(f"\n{'arm':<14} {'pass':>7} {'steps':>6} {'tokens':>8} {'compact':>8} "
          f"{'bad_calls':>10} {'sec':>7}")
    print("-" * 64)
    for arm in arms:
        rows = [o for o in outcomes if o.arm == arm]
        if not rows:
            continue
        rate = sum(o.passed for o in rows) / len(rows)
        print(f"{arm:<14} {rate:>6.0%} {mean(o.steps for o in rows):>6.1f} "
              f"{mean(o.prompt_tokens for o in rows):>8.0f} "
              f"{mean(o.compactions for o in rows):>8.1f} "
              f"{sum(o.invalid_tool_calls for o in rows):>10} "
              f"{mean(o.elapsed_s for o in rows):>7.1f}")

    print("\nper-case pass rate")
    case_ids = sorted({o.case_id for o in outcomes})
    print(f"{'arm':<14} " + " ".join(f"{c[:7]:>7}" for c in case_ids))
    for arm in arms:
        cells = []
        for cid in case_ids:
            rows = [o for o in outcomes if o.arm == arm and o.case_id == cid]
            cells.append(f"{sum(o.passed for o in rows) / len(rows):>7.0%}" if rows else f"{'-':>7}")
        print(f"{arm:<14} " + " ".join(cells))

    if any(o.episode_index for o in outcomes):
        print("\nlearning curve (pass rate over the sequence each arm saw)")
        for arm in arms:
            rows = sorted(
                (o for o in outcomes if o.arm == arm), key=lambda o: o.episode_index
            )
            if len(rows) < 4:
                continue
            middle = len(rows) // 2
            first = sum(o.passed for o in rows[:middle]) / middle
            second = sum(o.passed for o in rows[middle:]) / (len(rows) - middle)
            print(f"{arm:<14} first half {first:>5.0%}   second half {second:>5.0%}   "
                  f"delta {second - first:+.0%}")

    print("\nterminal status counts")
    for arm in arms:
        counts = Counter(o.status for o in outcomes if o.arm == arm)
        print(f"{arm:<14} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
