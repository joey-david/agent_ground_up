from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/lamgate.yaml")


def _project_root(config_path: Path) -> Path:
    """Resolve project-relative paths independently of where YAML files live."""
    for candidate in config_path.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML and remember the project root used for relative values."""
    raw_path = path or os.getenv("AGENT_CONFIG") or DEFAULT_CONFIG
    config_path = Path(raw_path).expanduser().resolve(strict=True)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    data["_root"] = _project_root(config_path)
    data["_path"] = config_path
    return data


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one named YAML section."""
    return config[name]


def path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve a path relative to the project root, not the configs directory."""
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else config["_root"] / candidate


def secret(section_data: dict[str, Any], key: str) -> str:
    """Read a secret from the environment variable named in YAML."""
    environment_name = section_data[key]
    return os.environ[environment_name]
