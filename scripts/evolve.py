from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from agent_ground_up.agent import Agent
from agent_ground_up.archive import Archive
from agent_ground_up.config import DEFAULT_CONFIG, load_config, section
from agent_ground_up.evaluate import Evaluator, LocalCodingRunner
from agent_ground_up.improve import PromptMutator, SelfImprover
from agent_ground_up.memory import ConstantMemory
from agent_ground_up.runtime import LocalBonsaiRuntime
from agent_ground_up.tasks import Curriculum, load_families
from agent_ground_up.tools import Toolbox

DEFAULT_CURRICULUM = "tests/fixtures/evolution/smoke_curriculum.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursively specialize the local coding agent")
    parser.add_argument("--curriculum", default=DEFAULT_CURRICULUM)
    parser.add_argument("--config", default=os.getenv("AGENT_CONFIG", str(DEFAULT_CONFIG)))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--target", type=float, default=0.45)
    parser.add_argument("--archive", default="state/archive")
    parser.add_argument("--memory", default="state/evolution-memory")
    parser.add_argument("--repository-root", default=".")
    parser.add_argument(
        "--unsafe-local",
        action="store_true",
        help="allow self-modified candidate Python to execute on the host",
    )
    return parser.parse_args()


def build_edit_agent(config_path: str | Path):
    config = load_config(config_path)
    model_config = section(config, "model")
    agent_config = section(config, "agent")
    runtime = LocalBonsaiRuntime(
        model_config["id"],
        adapter_path=model_config.get("adapter"),
        temperature=float(model_config.get("temperature", 1.0)),
        top_p=float(model_config.get("top_p", 0.95)),
        top_k=int(model_config.get("top_k", 20)),
    )

    def edit_agent(worktree: Path, prompt: str) -> str:
        tools = Toolbox(
            worktree,
            max_output_tokens=int(agent_config["max_tool_output_tokens"]),
            patch_size=runtime.patch_size,
            token_counter=runtime.count_text,
        )
        agent = Agent(
            runtime,
            tools,
            context_window=int(model_config["context_window"]),
            compact_at=float(agent_config["compact_at"]),
            recent_user_tokens=int(agent_config["recent_user_tokens"]),
            max_output_tokens=int(agent_config["max_output_tokens"]),
            max_steps=int(agent_config["max_steps"]),
            wall_time_s=int(agent_config["wall_time_s"]),
        )
        return agent.run(prompt).answer

    return edit_agent


def main() -> None:
    args = parse_args()
    if not args.unsafe_local:
        raise SystemExit(
            "Refusing to execute self-modified candidate Python on the host. "
            "Use --unsafe-local only for trusted curricula."
        )
    repository_root = Path(args.repository_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    curriculum = Curriculum(load_families(args.curriculum), target=args.target)
    archive = Archive(repository_root / args.archive)
    memory = ConstantMemory(repository_root / args.memory)
    improver = SelfImprover(
        archive=archive,
        curriculum=curriculum,
        evaluator=Evaluator(LocalCodingRunner(repository_root=repository_root)),
        mutator=PromptMutator(build_edit_agent(config_path)),
        memory=memory,
    )
    for result in improver.run(repository_root, args.rounds):
        print(json.dumps(asdict(result), ensure_ascii=False))


if __name__ == "__main__":
    main()
