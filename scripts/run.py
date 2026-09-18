from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from agent_ground_up.agent import Agent
from agent_ground_up.config import DEFAULT_CONFIG, load_config, path, section
from agent_ground_up.experience import ExperienceLog
from agent_ground_up.memory import ConstantMemory
from agent_ground_up.runtime import LocalBonsaiRuntime
from agent_ground_up.skills import SkillRegistry
from agent_ground_up.tools import Toolbox
from agent_ground_up.ui import TUI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Agent Ground Up on local Bonsai/MLX")
    parser.add_argument("task")
    parser.add_argument("--config", default=os.getenv("AGENT_CONFIG", str(DEFAULT_CONFIG)))
    parser.add_argument("--workdir", help="override agent.workdir")
    parser.add_argument("--trajectory", help="override agent.trajectory")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--no-experience", action="store_true")
    parser.add_argument("--no-skills", action="store_true")
    return parser.parse_args()


def build_runtime(model_config: dict) -> LocalBonsaiRuntime:
    return LocalBonsaiRuntime(
        model_config["id"],
        adapter_path=model_config.get("adapter"),
        temperature=float(model_config.get("temperature", 1.0)),
        top_p=float(model_config.get("top_p", 0.95)),
        top_k=int(model_config.get("top_k", 20)),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_config = section(config, "model")
    agent_config = section(config, "agent")
    ui_config = section(config, "ui")
    runtime = build_runtime(model_config)

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
    experience = (
        None
        if args.no_experience
        else ExperienceLog(path(config, agent_config.get("experience_dir", "state/experience")))
    )
    skills = (
        None
        if args.no_skills
        else SkillRegistry(path(config, agent_config.get("skills_dir", "skills")))
    )
    ui = None if args.quiet else TUI(max_lines=int(ui_config.get("max_lines", 40)))
    tools = Toolbox(
        workdir,
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
        trajectory_path=trajectory,
        ui=ui,
        memory=memory,
        experience=experience,
        skills=skills,
    )
    result = agent.run(args.task)
    if ui:
        ui.status(
            f"status={result.status} steps={result.steps} compactions={result.compactions}"
        )
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False))


if __name__ == "__main__":
    main()
