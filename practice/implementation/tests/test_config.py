from pathlib import Path
import tomllib

import yaml

ROOT = Path(__file__).parents[1]


def test_yaml_local_runtime_schema_is_filled() -> None:
    data = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert data["model"]["id"] != "<fill>"
    assert data["model"]["context_window"] >= 4096
    assert 0 < data["model"]["top_p"] <= 1
    assert data["model"]["top_k"] > 0
    assert data["agent"]["max_steps"] > 0
    assert 0 < data["evolution"]["target"] < 1
    assert data["training"]["rank"] > 0
    assert data["training"]["alpha"] > 0
    assert data["training"]["learning_rate"] > 0


def test_toml_declares_local_mlx_dependencies() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = data["project"]
    assert project["name"] == "agent-ground-up-practice"
    local = " ".join(project["optional-dependencies"]["local"])
    assert "mlx-vlm" in local and "transformers" in local
    assert "pytest" in " ".join(project["optional-dependencies"]["dev"])
