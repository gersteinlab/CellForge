"""Canonical filesystem locations for CellForge runtime artifacts.

The source/package location is not a writable runtime workspace after an
installed deployment. Runtime paths therefore resolve from
``CELLFORGE_WORKSPACE_DIR`` when set, otherwise from the caller's current
working directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, os.PathLike]
PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent


def workspace_root() -> Path:
    configured = os.getenv("CELLFORGE_WORKSPACE_DIR", "").strip()
    return Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()


def resolve_workspace_path(path: PathLike) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (workspace_root() / candidate).resolve()


def data_path(*parts: str) -> Path:
    return workspace_root().joinpath("data", *parts)


def config_path(path: Optional[PathLike] = None) -> Path:
    configured = path or os.getenv("CELLFORGE_CONFIG", "config.json")
    return resolve_workspace_path(configured)
