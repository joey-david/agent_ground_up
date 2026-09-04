from __future__ import annotations

import argparse
import os
import subprocess

from agent_ground_up.config import DEFAULT_CONFIG, load_config, section


def main() -> None:
    """Start TRL's rollout server used during RL."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.getenv("AGENT_CONFIG", str(DEFAULT_CONFIG)))
    args = parser.parse_args()
    values = section(load_config(args.config), "rollout_server")
    command = [
        "trl",
        "vllm-serve",
        "--model",
        values["model"],
        "--host",
        values["host"],
        "--port",
        str(values["port"]),
        "--max-model-len",
        str(values["max_model_len"]),
        "--gpu-memory-utilization",
        str(values["gpu_memory_utilization"]),
        "--disable-uvicorn-access-log",
    ]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
