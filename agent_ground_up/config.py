from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/local.yaml")


def _project_root(config_path: Path) -> Path:
    for candidate in config_path.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    raw_path = path or os.getenv("AGENT_CONFIG") or DEFAULT_CONFIG
    config_path = Path(raw_path).expanduser().resolve(strict=True)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    data["_root"] = _project_root(config_path)
    data["_path"] = config_path
    return data


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    return config[name]


def path(config: dict[str, Any], value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else config["_root"] / candidate
