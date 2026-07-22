from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML and remember the file paths used to resolve relative values."""
    config_path = (
        Path(path or os.getenv("AGENT_CONFIG", "config.yaml")).expanduser().resolve(strict=True)
    )
    data = yaml.safe_load(config_path.read_text())
    data["_root"] = config_path.parent
    data["_path"] = config_path
    return data


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one named YAML section."""
    return config[name]


def path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve a config path relative to its YAML file."""
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else config["_root"] / candidate


def secret(section_data: dict[str, Any], key: str) -> str:
    """Read a secret from the environment variable named in YAML."""
    environment_name = section_data[key]
    return os.environ[environment_name]
