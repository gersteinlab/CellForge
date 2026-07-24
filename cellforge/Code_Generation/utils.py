"""Utilities shared by code generation backends."""

import os
import subprocess
from pathlib import Path
from typing import Mapping, Optional, Sequence


def load_project_env(project_root: Path) -> None:
    try:
        from dotenv import load_dotenv

        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except Exception:
        pass


def truthy_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def extract_code(model_output: str) -> str:
    if "```python" in model_output:
        start_idx = model_output.find("```python") + len("```python")
        end_idx = model_output.find("```", start_idx)
        if end_idx != -1:
            return model_output[start_idx:end_idx].strip()
        return model_output[start_idx:].strip()
    if "```" in model_output:
        start_idx = model_output.find("```") + len("```")
        end_idx = model_output.find("```", start_idx)
        if end_idx != -1:
            return model_output[start_idx:end_idx].strip()
        return model_output[start_idx:].strip()
    return model_output.strip()


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: Optional[Mapping[str, str]] = None,
    prompt: Optional[str] = None,
    timeout: int = 1800,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
