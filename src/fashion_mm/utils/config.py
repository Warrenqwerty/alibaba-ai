from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


ConfigDict = dict[str, Any]


def load_yaml(path: str | Path) -> ConfigDict:
    """Load a YAML config file into a dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> ConfigDict:
    """Recursively merge two dictionaries and return a new dictionary."""
    merged: ConfigDict = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(*paths: str | Path) -> ConfigDict:
    """Load and merge one or more YAML config files from left to right."""
    if not paths:
        raise ValueError("At least one config path is required.")

    config: ConfigDict = {}
    for path in paths:
        config = deep_merge(config, load_yaml(path))
    return config
