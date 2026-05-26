from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration."""

    with Path(path).open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must be a mapping: {path}")
    return config


def deep_update(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Recursively merge two dictionaries without mutating inputs."""

    result = deepcopy(base)
    if not override:
        return result
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def require_keys(config: dict[str, Any], keys: list[str], context: str = "config") -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"Missing required {context} key(s): {missing}")
