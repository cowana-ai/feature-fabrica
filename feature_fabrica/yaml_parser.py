# mypy: ignore-errors
import yaml
from typing import Any
from pathlib import Path


def load_yaml(file_path: Path) -> dict[str, Any] | None:
    with open(file_path) as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
            return None


def validate_feature_spec(spec):
    # Add validation logic here (e.g., check required fields)
    pass
