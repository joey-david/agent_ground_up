from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from openai import OpenAI
from transformers import AutoProcessor

from agent_ground_up.agent import Agent
from agent_ground_up.archive import Archive
from agent_ground_up.config import load_config, secret, section
from agent_ground_up.evaluate import Evaluator, LocalCodingRunner
from agent_ground_up.improve import PromptMutator, SelfImprover
from agent_ground_up.memory import ConstantMemory
from agent_ground_up.tasks import Curriculum, load_families
from agent_ground_up.tools import Toolbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursively specialize the agent on frontier tasks")
    parser.add_argument("--curriculum", required=True, help="JSON task-family file")
    parser.add_argument("--config", default=os.getenv("AGENT_CONFIG", "config.yaml"))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--target", type=float, default=0.45)
    parser.add_argument("--archive", default="state/archive")
    parser.add_argument("--memory", default="state/evolution-memory")
    parser.add_argument("--repository-root", default=".")
    return parser.parse_args()


def build_edit_agent(config_path: str | Path):
    config = load_config(config_path)
    model_config = section(config, "model")
    agent_config = section(config, "agent")
    processor = AutoProcessor.from_pretrained(model_config["processor"], trust_remote_code=False)
    client = OpenAI(base_url=model_config["base_url"], api_key=secret(model_config, "api_key_env"))

    def token_counter(text: str) -> int:
        return len(processor.tokenizer.encode(text, add_special_tokens=False))

    def edit_agent(worktree: Path, prompt: str) -> str:
        agent = Agent(
            client,
            model_config["served_name"],
            Toolbox(
                worktree,
                max_output_tokens=agent_config["max_tool_output_tokens"],
                token_counter=token_counter,
            ),
            processor,
            context_window=model_config["context_window"],
            compact_at=agent_config["compact_at"],
            recent_user_tokens=agent_config["recent_user_tokens"],
            max_output_tokens=agent_config["max_output_tokens"],
            max_steps=agent_config["max_steps"],
            wall_time_s=agent_config["wall_time_s"],
        )
        return agent.run(prompt).answer

    return edit_agent


def main() -> None:
    args = parse_args()
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
    results = improver.run(repository_root, args.rounds)
    for result in results:
        print(json.dumps(asdict(result), ensure_ascii=False))


if __name__ == "__main__":
    main()
