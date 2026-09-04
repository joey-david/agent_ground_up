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
from agent_ground_up.factory import build_model_runtime
from agent_ground_up.improve import PromptMutator, SelfImprover
from agent_ground_up.memory import ConstantMemory
from agent_ground_up.tasks import Curriculum, load_families
from agent_ground_up.tools import Toolbox

DEFAULT_CURRICULUM = "tests/fixtures/evolution/smoke_curriculum.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursively specialize the agent on frontier tasks")
    parser.add_argument("--curriculum", default=DEFAULT_CURRICULUM, help="JSON task-family file")
    parser.add_argument("--config", default=os.getenv("AGENT_CONFIG", str(DEFAULT_CONFIG)))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--target", type=float, default=0.45)
    parser.add_argument("--archive", default="state/archive")
    parser.add_argument("--memory", default="state/evolution-memory")
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--unsafe-local", action="store_true", help="allow candidate Python to execute on the host; smoke tests only")
    return parser.parse_args()


def build_edit_agent(config_path: str | Path):
    config = load_config(config_path)
    model_config = section(config, "model")
    agent_config = section(config, "agent")
    bundle = build_model_runtime(model_config)

    def edit_agent(worktree: Path, prompt: str) -> str:
        local_bundle = build_model_runtime(model_config) if bundle.runtime is not None else bundle
        agent = Agent(
            local_bundle.client,
            model_config["served_name"],
            Toolbox(worktree, max_output_tokens=agent_config["max_tool_output_tokens"], token_counter=local_bundle.token_counter),
            local_bundle.processor,
            context_window=model_config["context_window"],
            compact_at=agent_config["compact_at"],
            recent_user_tokens=agent_config["recent_user_tokens"],
            max_output_tokens=agent_config["max_output_tokens"],
            max_steps=agent_config["max_steps"],
            wall_time_s=agent_config["wall_time_s"],
            runtime=local_bundle.runtime,
        )
        return agent.run(prompt).answer

    return edit_agent


def main() -> None:
    args = parse_args()
    if not args.unsafe_local:
        raise SystemExit(
            "Refusing to execute self-modified candidate Python on the host. "
            "Use --unsafe-local only for the bundled trusted smoke curriculum."
        )
    repository_root = Path(args.repository_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    families = load_families(args.curriculum)
    curriculum = Curriculum(families, target=args.target)
    archive = Archive(repository_root / args.archive)
    memory = ConstantMemory(repository_root / args.memory)
    runner = LocalCodingRunner(repository_root=repository_root)
    improver = SelfImprover(
        archive=archive,
        curriculum=curriculum,
        evaluator=Evaluator(runner),
        mutator=PromptMutator(build_edit_agent(config_path)),
        memory=memory,
    )
    for result in improver.run(repository_root, args.rounds):
        print(json.dumps(asdict(result), ensure_ascii=False))


if __name__ == "__main__":
    main()
