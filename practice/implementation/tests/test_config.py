from pathlib import Path
import tomllib

import yaml

ROOT = Path(__file__).parents[1]


def test_yaml_runtime_schema_is_filled() -> None:
    data = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert data["model"]["runtime"] in {"chat_completions", "responses_continuous"}
    assert data["model"]["served_name"] != "<fill>"
    assert data["model"]["base_url"].startswith(("http://", "https://"))
    assert data["model"]["context_window"] >= 4096
    assert data["agent"]["max_steps"] > 0
    assert 0 < data["evolution"]["target"] < 1
    assert 0 < data["training"]["epsilon_low"] < data["training"]["epsilon_high"] < 1


def test_toml_declares_core_and_training_dependencies() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = data["project"]
    assert project["name"] == "agent-ground-up-practice"
    dependencies = " ".join(project["dependencies"])
    assert "openai" in dependencies and "pyyaml" in dependencies
    train = " ".join(project["optional-dependencies"]["train"])
    assert "torch" in train and "trl" in train
    assert "pytest" in " ".join(project["optional-dependencies"]["dev"])
