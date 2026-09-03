from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from openai import OpenAI
from transformers import AutoProcessor

from agent_ground_up.agent import Agent
from agent_ground_up.config import load_config, path, secret, section
from agent_ground_up.memory import ConstantMemory
from agent_ground_up.skills import SkillRegistry
from agent_ground_up.tools import Toolbox
from agent_ground_up.ui import TUI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the self-evolving coding-agent kernel")
    parser.add_argument("task")
    parser.add_argument("--config", default=os.getenv("AGENT_CONFIG", "config.yaml"))
    parser.add_argument("--workdir", help="override agent.workdir")
    parser.add_argument("--trajectory", help="override agent.trajectory")
    parser.add_argument("--quiet", action="store_true", help="disable TUI output")
    parser.add_argument("--json", action="store_true", help="emit final RunResult as JSON")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--no-skills", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_config = section(config, "model")
    agent_config = section(config, "agent")
    ui_config = section(config, "ui")
    if not model_config.get("base_url"):
        raise SystemExit("Set model.base_url in config.yaml")
    processor = AutoProcessor.from_pretrained(model_config["processor"], trust_remote_code=False)

    def token_counter(text: str) -> int:
        return len(processor.tokenizer.encode(text, add_special_tokens=False))

    workdir = (
        Path(args.workdir).expanduser().resolve()
        if args.workdir
        else path(config, agent_config["workdir"])
    )
    trajectory = (
        Path(args.trajectory).expanduser().resolve()
        if args.trajectory
        else path(config, agent_config["trajectory"])
    )
    memory = None
    if not args.no_memory:
        memory = ConstantMemory(
            path(config, agent_config.get("memory_dir", "state/memory")),
            wake_records=int(agent_config.get("wake_records", 6)),
            leaf_size=int(agent_config.get("memory_leaf_size", 8)),
            summary_chars=int(agent_config.get("memory_summary_chars", 500)),
        )
    skills = None
    if not args.no_skills:
        skills = SkillRegistry(path(config, agent_config.get("skills_dir", "skills")))

    ui = None if args.quiet else TUI(max_lines=ui_config["max_lines"])
    agent = Agent(
        OpenAI(base_url=model_config["base_url"], api_key=secret(model_config, "api_key_env")),
        model_config["served_name"],
        Toolbox(
            workdir,
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
        trajectory_path=trajectory,
        ui=ui,
        memory=memory,
        skills=skills,
    )
    result = agent.run(args.task)
    if ui:
        ui.status(f"status={result.status} steps={result.steps} compactions={result.compactions}")
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False))


if __name__ == "__main__":
    main()
