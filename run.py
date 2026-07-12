from __future__ import annotations

import argparse
import os

from openai import OpenAI
from transformers import AutoProcessor

from agent_ground_up.agent import Agent
from agent_ground_up.config import load_config, path, secret, section
from agent_ground_up.tools import Toolbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the small multimodal coding agent")
    parser.add_argument("task")
    parser.add_argument("--config", default=os.getenv("AGENT_CONFIG", "config.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_config = section(config, "model")
    agent_config = section(config, "agent")
    if not model_config.get("base_url"):
        raise SystemExit("Set model.base_url in config.yaml")
    processor = AutoProcessor.from_pretrained(model_config["name"], trust_remote_code=False)

    def token_counter(text: str) -> int:
        return len(processor.tokenizer.encode(text, add_special_tokens=False))

    agent = Agent(
        OpenAI(base_url=model_config["base_url"], api_key=secret(model_config, "api_key_env")),
        model_config["name"],
        Toolbox(
            path(config, agent_config["workdir"]),
            max_output_tokens=agent_config["max_tool_output_tokens"],
            token_counter=token_counter,
        ),
        processor,
        context_window=model_config["context_window"],
        compact_at=agent_config["compact_at"],
        max_output_tokens=agent_config["max_output_tokens"],
        max_steps=agent_config["max_steps"],
        wall_time_s=agent_config["wall_time_s"],
        trajectory_path=path(config, agent_config["trajectory"]),
    )
    result = agent.run(args.task)
    print(result.answer)
    print(f"\nstatus={result.status} steps={result.steps} compactions={result.compactions}")


if __name__ == "__main__":
    main()
