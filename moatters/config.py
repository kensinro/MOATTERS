"""Portable path and runtime configuration for MOATTERS.

Configuration precedence:
1. Explicit environment variables.
2. JSON file specified by ``MOATTERS_CONFIG``.
3. Repository-local ``config/moatters_config.json``.
4. Platform-neutral defaults under the repository root.

Relative paths in a JSON configuration file are resolved from the repository
root. Relative paths supplied through environment variables are resolved from
the caller's current working directory. This distinction keeps the bundled
configuration stable while retaining conventional shell behaviour.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "moatters_config.json"


def _config_path() -> Path:
    raw = os.environ.get("MOATTERS_CONFIG")
    if not raw:
        return DEFAULT_CONFIG
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


CONFIG_PATH = _config_path()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Unable to parse MOATTERS config: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"MOATTERS config must contain a JSON object: {path}")
    return payload


_CONFIG = _load_json(CONFIG_PATH)


def _resolved_path(raw: str | os.PathLike[str], *, base: Path) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _root(env_name: str, key: str, fallback: Path) -> Path:
    env_value = os.environ.get(env_name)
    if env_value:
        return _resolved_path(env_value, base=Path.cwd())
    config_value = _CONFIG.get(key)
    if config_value:
        # Repository-relative configuration remains stable regardless of cwd.
        return _resolved_path(config_value, base=REPO_ROOT)
    return fallback.resolve()


DATA_ROOT = _root("MOATTERS_DATA_ROOT", "data_root", REPO_ROOT / "data")
OUTPUT_ROOT = _root("MOATTERS_OUTPUT_ROOT", "output_root", REPO_ROOT / "outputs")
RSCRIPT = os.environ.get("MOATTERS_RSCRIPT") or _CONFIG.get("rscript") or "Rscript"


def _parts(value: str | Path | None) -> tuple[str, ...]:
    if value is None:
        return ()
    text = str(value).replace("\\", "/").strip("/")
    return tuple(part for part in text.split("/") if part)


def data_path(relative: str | Path | None = None) -> Path:
    return DATA_ROOT.joinpath(*_parts(relative))


def output_path(relative: str | Path | None = None) -> Path:
    return OUTPUT_ROOT.joinpath(*_parts(relative))


def runtime_summary() -> dict[str, str]:
    return {
        "repository_root": str(REPO_ROOT),
        "config_path": str(CONFIG_PATH),
        "data_root": str(DATA_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "rscript": str(RSCRIPT),
    }
