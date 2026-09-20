"""Config loading with ``extends`` composition.

Configs may declare ``extends: other.yaml``; the referenced file is loaded
first (recursively) and the current file's keys override/produce a merged
mapping. This replaces the earlier prose-only "reuse binary.yaml" comments so
that QAT/distillation configs actually inherit layer selection.

Paths in ``extends`` are resolved relative to the including file.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Raised for malformed configs (bad extends target, cycles, etc.)."""


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict:
    """Load a YAML config, resolving ``extends`` chains.

    Args:
        path: config file path (may be relative to the current directory).
        _seen: internal recursion guard for cycle detection.

    Raises:
        ConfigError: on missing file, missing extends target, or cycles.
    """
    path = Path(path).resolve()
    if path in _seen:
        chain = " -> ".join(str(p) for p in (*_seen, path))
        raise ConfigError(f"Config 'extends' cycle detected: {chain}")
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")

    parent_ref = data.pop("extends", None)
    if parent_ref is None:
        merged = data
    else:
        parent_path = (path.parent / parent_ref).resolve()
        parent = load_config(parent_path, (*_seen, path))
        merged = _deep_merge(parent, data)

    # Record the *requested* (leaf) config, not the root of the ``extends``
    # chain. Each recursion level sets this, so the outermost call wins; using
    # ``setdefault`` would instead leave the first (base) parent's path.
    merged["_config_path"] = str(path)
    return merged
