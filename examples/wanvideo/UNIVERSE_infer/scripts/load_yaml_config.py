#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required to load UNIVERSE YAML configs. Install pyyaml.") from exc


_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VAR_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def _as_env_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return ",".join(_as_env_value(item) for item in value)
    return str(value)


def _entry_parts(entry: Any) -> Tuple[Any, list[str]]:
    if isinstance(entry, dict):
        return entry.get("default"), _as_list(entry.get("aliases"))
    return entry, []


def _expand_refs(value: str, env: Dict[str, str]) -> str:
    expanded = value
    for _ in range(10):
        next_value = _VAR_REF_RE.sub(lambda match: env.get(match.group(1), match.group(0)), expanded)
        if next_value == expanded:
            return next_value
        expanded = next_value
    return expanded


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"UNIVERSE config must be a mapping: {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit shell exports from a UNIVERSE YAML config.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    data = _load_config(args.config)
    env = dict(os.environ)
    exports: list[Tuple[str, str]] = []

    for name, entry in data.items():
        if not isinstance(name, str) or not _ENV_NAME_RE.match(name):
            raise ValueError(f"Invalid environment variable name in {args.config}: {name!r}")

        default, aliases = _entry_parts(entry)
        value = os.environ.get(name)
        if value is None or value == "":
            for alias in aliases:
                if not _ENV_NAME_RE.match(alias):
                    raise ValueError(f"Invalid alias for {name}: {alias!r}")
                alias_value = os.environ.get(alias)
                if alias_value is not None and alias_value != "":
                    value = alias_value
                    break

        if value is None or value == "":
            value = _as_env_value(default)
        else:
            value = _as_env_value(value)

        value = _expand_refs(value, env)
        env[name] = value
        exports.append((name, value))

    for name, value in exports:
        sys.stdout.write(f"export {name}={shlex.quote(value)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
