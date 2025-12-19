"""Argument parsing and YAML-based configuration utilities."""

import argparse
from typing import Any

from ruamel import yaml


class AttrDict(dict[str, Any]):
    """Dictionary with attribute-style access (recursive)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e

        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError as e:
            raise AttributeError(key) from e


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


def _set_nested(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def parse_arguments(hyperparameters):
    """Parse command-line arguments overriding given hyperparameters."""
    parser = argparse.ArgumentParser(description="Hyperparameters")

    def _infer_list_type(sample):
        """Infer the element type for a list-valued hyperparameter."""
        if not sample:
            return str
        return type(sample[0])

    for key, value in hyperparameters.items():
        if isinstance(value, bool):
            # Allow both --flag and --no-flag (works whether default True or False)
            group = parser.add_mutually_exclusive_group(required=False)
            group.add_argument(f"--{key}", dest=key, action="store_true")
            group.add_argument(f"--no-{key}", dest=key, action="store_false")
            parser.set_defaults(**{key: value})

        elif isinstance(value, list):
            parser.add_argument(
                f"--{key}",
                nargs="+",
                type=_infer_list_type(value),
                default=value,
            )
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    return vars(args)


def get_configuration(yaml_file):
    """Load hyperparameters from YAML and merge with CLI overrides."""
    with open(yaml_file, "r", encoding="utf-8") as f:
        yaml_loader = yaml.YAML()
        hyperparameters = yaml_loader.load(f)

    # 1) flatten nested YAML to leaf keys
    flat = _flatten_dict(hyperparameters)

    # 2) parse CLI only for keys present in YAML
    flat_overrides = parse_arguments(flat)

    # 3) apply overrides back into the nested dict
    for k, v in flat_overrides.items():
        _set_nested(hyperparameters, k, v)

    return AttrDict(hyperparameters)
