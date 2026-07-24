"""Backend registry for CellForge code generation."""

import os
from typing import Dict, Type

from .base import CodeGenerationBackend
from .codex_backend import CodexCodeGenerator
from .legacy import LegacyBackendUnavailableError


BACKENDS: Dict[str, Type[CodeGenerationBackend]] = {
    "codex": CodexCodeGenerator,
}


def list_backends():
    """Return only backends that implement the current generation contract."""

    return sorted(BACKENDS)


def create_backend(name=None) -> CodeGenerationBackend:
    backend_name = (name or os.getenv("CODEGEN_BACKEND") or "codex").strip().lower()
    if backend_name in {"openhands", "openhands-legacy", "legacy-openhands"}:
        raise LegacyBackendUnavailableError.for_backend(backend_name)
    try:
        backend_cls = BACKENDS[backend_name]
    except KeyError as exc:
        available = ", ".join(list_backends())
        raise ValueError(f"Unknown code generation backend '{backend_name}'. Available backends: {available}") from exc
    return backend_cls()
