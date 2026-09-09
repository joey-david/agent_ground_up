"""Score system-prompt variants on a verifiable coding benchmark.

Each (variant, case) pair is one independent episode: a fresh copy of the case workspace, fresh
memory/experience/skill state, and the variant installed as the agent's system prompt. The case is
scored by its verifier's exit code, so the only thing being compared is the prompt.

    uv run python scripts/prompt_search.py --split train --repeats 1
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from openai import OpenAI
from transformers import AutoProcessor

from agent_ground_up.agent import COMPACT_PROMPT, SYSTEM_PROMPT, Agent
from agent_ground_up.config import DEFAULT_CONFIG, load_config, section
from agent_ground_up.experience import ExperienceLog
from agent_ground_up.memory import ConstantMemory
from agent_ground_up.skills import SkillRegistry
from agent_ground_up.tasks import load_families
from agent_ground_up.tools import Toolbox

DEFAULT_ENDPOINTS = "http://127.0.0.1:8020/v1,http://127.0.0.1:8022/v1"


@dataclass(slots=True)
class Outcome:
    variant: str
    case_id: str
    family: str
    repeat: int
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", default="benchmarks/coding/curriculum.json")
    parser.add_argument("--variants", default="benchmarks/prompt_variants.json")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", default="train")
    parser.add_argument("--only", default="", help="comma-separated variant names to run")
    parser.add_argument("--cases", default="", help="comma-separated case ids to restrict to")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--endpoints", default=DEFAULT_ENDPOINTS)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--wall-time-s", type=int, default=420)
    parser.add_argument("--vary", choices=["system", "compact"], default="system",
                        help="which prompt the variants file replaces")
    parser.add_argument("--context-window", type=int, default=None,
                        help="shrink the window so compaction actually fires")
    parser.add_argument("--compact-at", type=float, default=None,
                        help="fraction of the window that triggers compaction")
    parser.add_argument("--max-tool-output-tokens", type=int, default=None,
                        help="cap tool output; must stay well under a shrunken context window")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--out", default="runs/prompt_search.json")
    return parser.parse_args()


def run_case(
    *,
    variant_name: str,
    system_prompt: str,
    case,
    family: str,
    repeat: int,
    client: OpenAI,
    processor,
    model: str,
    config,
    agent_config,
    repository_root: Path,
    max_steps: int,
    wall_time_s: int,
    temperature: float | None,
    top_p: float | None,
    vary: str,
    context_window: int | None,
    max_tool_output_tokens: int | None,
    compact_at: float | None,
) -> Outcome:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"bench-{case.id}-") as temp:
        root = Path(temp)
        workdir = root / "workspace"
        shutil.copytree(repository_root / case.workspace, workdir)
        toolbox = Toolbox(
            workdir,
            max_output_tokens=max_tool_output_tokens or agent_config["max_tool_output_tokens"],
            token_counter=lambda text: len(
                processor.tokenizer.encode(text, add_special_tokens=False)
            ),
        )
        agent = Agent(
            client,
            model,
            toolbox,
            processor,
            context_window=context_window or section(config, "model")["context_window"],
            compact_at=compact_at or agent_config["compact_at"],
            recent_user_tokens=agent_config["recent_user_tokens"],
            max_output_tokens=agent_config["max_output_tokens"],
            max_steps=max_steps,
            wall_time_s=wall_time_s,
            system_prompt=system_prompt if vary == "system" else SYSTEM_PROMPT,
            compact_prompt=system_prompt if vary == "compact" else COMPACT_PROMPT,
            temperature=temperature,
            top_p=top_p,
            memory=ConstantMemory(root / "memory"),
            experience=ExperienceLog(root / "experience"),
            skills=SkillRegistry(root / "skills"),
        )
        prompts = [case.prompt] + ([case.followup] if case.followup else [])
        status, steps, tokens, valid, invalid, detail = "completed", 0, 0, 0, 0, ""
        tools_used: dict[str, int] = {}
        compactions = 0
        try:
            for episode_prompt in prompts:
                # A followup reuses this agent, so memory, experience and skills carry over while
                # the conversation does not -- which is exactly what the subsystem tools are for.
                result = agent.run(episode_prompt)
                status = result.status
                steps += result.steps
                tokens = max(tokens, result.prompt_tokens)
                valid += result.valid_tool_calls
                invalid += result.invalid_tool_calls
                compactions += result.compactions
                for message in agent.messages:
                    for call in message.get("tool_calls") or []:
                        name = (call.get("function") or {}).get("name", "?")
                        tools_used[name] = tools_used.get(name, 0) + 1
        except Exception as error:  # a crashed episode is a failed episode, not a dead sweep
            # Keep the frame that raised: "prompt exceeds the window" can come from the main
            # completion or from inside compaction, and the message alone cannot tell them apart.
            frames = traceback.extract_tb(error.__traceback__)
            where = " <- ".join(f"{f.name}:{f.lineno}" for f in frames[-4:])
            status = "error"
            detail = f"{type(error).__name__}: {error} | {where} | steps={steps}"[:400]

        passed = False
        if status != "error":
            check = subprocess.run(
                case.verifier,
                shell=True,
                executable="/bin/bash",
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            passed = check.returncode == 0
            if not passed:
                detail = (check.stdout + check.stderr).strip()[-300:]

    return Outcome(
        variant=variant_name,
        case_id=case.id,
        family=family,
        repeat=repeat,
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

    variants = json.loads(Path(args.variants).read_text())
    if args.only:
        wanted = [name.strip() for name in args.only.split(",")]
        variants = {name: variants[name] for name in wanted}

    families = load_families(args.curriculum)
    cases = [(family.name, case) for family in families for case in family.by_split(args.split)]
    if args.cases:
        wanted_cases = {cid.strip() for cid in args.cases.split(",")}
        cases = [(f, c) for f, c in cases if c.id in wanted_cases]
    if not cases:
        raise SystemExit(f"no cases in split {args.split!r}")

    processor = AutoProcessor.from_pretrained(model_config["processor"], trust_remote_code=False)
    endpoints = [url.strip() for url in args.endpoints.split(",") if url.strip()]
    clients = [OpenAI(base_url=url, api_key="EMPTY", timeout=600.0) for url in endpoints]
    print(f"{len(variants)} variants x {len(cases)} cases x {args.repeats} repeats "
          f"= {len(variants) * len(cases) * args.repeats} episodes over {len(endpoints)} endpoints")

    jobs = [
        (name, prompt, family, case, repeat)
        for name, prompt in variants.items()
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
        name, prompt, family, case, repeat = job
        with lock:
            client = clients[next(counter) % len(clients)]
        outcome = run_case(
            variant_name=name,
            system_prompt=prompt,
            case=case,
            family=family,
            repeat=repeat,
            client=client,
            processor=processor,
            model=model_config["served_name"],
            config=config,
            agent_config=agent_config,
            repository_root=repository_root,
            max_steps=args.max_steps,
            wall_time_s=args.wall_time_s,
            temperature=args.temperature,
            top_p=args.top_p,
            vary=args.vary,
            context_window=args.context_window,
            max_tool_output_tokens=args.max_tool_output_tokens,
            compact_at=args.compact_at,
        )
        with lock:
            done[0] += 1
            # Persist after every episode: a sweep is long enough that losing it to an
            # interrupted process costs more than rewriting a small JSON file each time.
            finished.append(outcome)
            out_path.write_text(json.dumps([asdict(o) for o in finished], indent=2) + "\n")
            mark = "PASS" if outcome.passed else "FAIL"
            print(f"[{done[0]:>3}/{len(jobs)}] {mark} {name:<22} {outcome.case_id:<12} "
                  f"steps={outcome.steps:<3} {outcome.elapsed_s:6.1f}s {outcome.status}",
                  flush=True)
        return outcome

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        outcomes = list(pool.map(work, jobs))

    out_path.write_text(json.dumps([asdict(o) for o in outcomes], indent=2) + "\n")
    report(outcomes, variants)
    print(f"\nwrote {out_path}")


def report(outcomes: list[Outcome], variants: dict[str, str]) -> None:
    print(f"\n{'variant':<24} {'pass':>7} {'words':>6} {'steps':>6} {'tokens':>8} "
          f"{'compact':>8} {'bad_calls':>10} {'sec':>7}")
    print("-" * 84)
    for name in variants:
        rows = [o for o in outcomes if o.variant == name]
        if not rows:
            continue
        rate = sum(o.passed for o in rows) / len(rows)
        print(f"{name:<24} {rate:>6.0%} {len(variants[name].split()):>6} "
              f"{mean(o.steps for o in rows):>6.1f} {mean(o.prompt_tokens for o in rows):>8.0f} "
              f"{mean(o.compactions for o in rows):>8.1f} "
              f"{sum(o.invalid_tool_calls for o in rows):>10} "
              f"{mean(o.elapsed_s for o in rows):>7.1f}")

    print("\nper-case pass rate")
    case_ids = sorted({o.case_id for o in outcomes})
    print(f"{'variant':<24} " + " ".join(f"{c[:6]:>6}" for c in case_ids))
    for name in variants:
        cells = []
        for cid in case_ids:
            rows = [o for o in outcomes if o.variant == name and o.case_id == cid]
            cells.append(f"{sum(o.passed for o in rows) / len(rows):>6.0%}" if rows else f"{'-':>6}")
        print(f"{name:<24} " + " ".join(cells))


if __name__ == "__main__":
    main()
