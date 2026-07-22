from __future__ import annotations

import argparse
import os

import uvicorn

from agent_ground_up.config import load_config, section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.getenv("AGENT_CONFIG", "config.yaml"))
    args = parser.parse_args()
    os.environ["AGENT_CONFIG"] = args.config
    values = section(load_config(args.config), "sandbox")
    uvicorn.run(
        "infra.sandbox_server:app", host=values["host"], port=values["port"], access_log=False
    )


if __name__ == "__main__":
    main()
