from __future__ import annotations

from pathlib import Path
import tomllib

import yaml

HERE = Path(__file__).resolve()
TARGET = HERE.parents[1]
REFERENCE = HERE.parents[2] / "implementation"


def yaml_paths(value, prefix=()):
    if isinstance(value, dict):
        result = set()
        for key, child in value.items():
            result |= yaml_paths(child, (*prefix, str(key)))
        return result
    return {prefix}


def toml_paths(value, prefix=()):
    if isinstance(value, dict):
        result = set()
        for key, child in value.items():
            result |= toml_paths(child, (*prefix, str(key)))
        return result
    return {prefix}


def test_yaml_schema_only() -> None:
    expected = yaml.safe_load((REFERENCE / "config.yaml").read_text())
    actual = yaml.safe_load((TARGET / "config.yaml").read_text()) or {}
    assert yaml_paths(actual) == yaml_paths(expected)


def test_toml_schema_only() -> None:
    expected = tomllib.loads((REFERENCE / "pyproject.toml").read_text())
    actual_text = (TARGET / "pyproject.toml").read_text()
    actual = tomllib.loads(actual_text) if actual_text.strip() else {}
    assert toml_paths(actual) == toml_paths(expected)
