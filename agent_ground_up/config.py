from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or os.getenv("AGENT_CONFIG", "config.yaml")).expanduser().resolve(strict=True)
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    data["_root"] = config_path.parent
    data["_path"] = config_path
    return data


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing config section: {name}")
    return value


def path(config: dict[str, Any], value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else config["_root"] / candidate


def secret(section_data: dict[str, Any], key: str) -> str:
    environment_name = section_data[key]
    value = os.getenv(environment_name)
    if value is None:
        raise ValueError(f"Required environment variable is not set: {environment_name}")
    return value
