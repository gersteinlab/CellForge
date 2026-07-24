"""Workspace lifecycle for the Codex CLI code-generation backend."""

import json
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .contracts import AgentRunRequest, AgentRunResult
from .prompts import build_generation_prompt, build_repair_prompt


DEFAULT_ACCEPTANCE_CONTRACT = {
    "entrypoint": "result.py",
    "required_cli": {
        "training_data": ["--adata-path", "--adata_path", "--data_path", "--data", "positional"],
        "validation_data": ["--val-adata-path", "--val_data", "--val-path"],
        "metrics_output": ["--output-metrics", "--metrics-path", "--output"],
    },
    "constraints": {
        "paths_must_not_be_hardcoded": True,
        "network_downloads": False,
        "cluster_submission": False,
        "test_or_ood_data_access": False,
    },
    "required_artifacts": ["result.py"],
}


def _safe_name(value: Optional[str]) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "full_plan").strip("._")
    return cleaned or "full_plan"


def create_task_workspace(output_dir: str, backend: str, task_id: Optional[str]) -> Path:
    """Create a persistent, task-scoped directory rather than a temporary cwd."""

    token = uuid.uuid4().hex[:10]
    workspace = Path(output_dir) / ".cellforge_workspaces" / backend / f"{_safe_name(task_id)}_{token}"
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace.resolve()


def prepare_workspace(request: AgentRunRequest) -> None:
    """Materialize machine-readable inputs that the agent may inspect."""

    request.workspace.mkdir(parents=True, exist_ok=True)
    plan_path = request.workspace / "research_plan.json"
    if not plan_path.exists():
        plan_path.write_text(
            json.dumps(request.research_plan, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    contract = dict(DEFAULT_ACCEPTANCE_CONTRACT)
    contract["entrypoint"] = request.entrypoint
    contract["required_artifacts"] = [request.entrypoint]
    contract_path = request.workspace / "acceptance_contract.json"
    if not contract_path.exists():
        contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    instructions_path = request.workspace / "AGENTS.md"
    if not instructions_path.exists():
        instructions_path.write_text(
            "# CellForge coding task\n\n"
            "Work only in this directory. Implement `research_plan.json` under the rules in "
            "`acceptance_contract.json`. Do not access test/OOD data, submit cluster jobs, "
            "download dependencies, or alter the scientific evaluation protocol.\n",
            encoding="utf-8",
        )


def _snapshot(workspace: Path) -> Dict[str, Tuple[int, int]]:
    snapshot: Dict[str, Tuple[int, int]] = {}
    for path in workspace.rglob("*"):
        if path.is_file():
            stat = path.stat()
            snapshot[str(path.relative_to(workspace))] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def _changed_files(before: Dict[str, Tuple[int, int]], workspace: Path) -> List[str]:
    after = _snapshot(workspace)
    return sorted(name for name, signature in after.items() if before.get(name) != signature)


def _append_event(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def _artifact_path(request: AgentRunRequest) -> Path:
    if Path(request.entrypoint).is_absolute():
        raise ValueError("entrypoint must be relative to the task workspace")
    artifact = (request.workspace / request.entrypoint).resolve()
    try:
        artifact.relative_to(request.workspace.resolve())
    except ValueError as exc:
        raise ValueError("entrypoint must stay inside the task workspace") from exc
    return artifact


CommandFactory = Callable[[AgentRunRequest, Path], Tuple[List[str], dict]]


def execute_agent(
    request: AgentRunRequest,
    *,
    backend: str,
    command_factory: CommandFactory,
) -> AgentRunResult:
    """Run one CLI agent turn and verify that it delivered the requested file."""

    artifact = _artifact_path(request)
    prepare_workspace(request)
    logs_dir = request.workspace / "logs"
    logs_dir.mkdir(exist_ok=True)
    event_log = logs_dir / "agent_events.jsonl"
    final_message_path = logs_dir / f"{backend}_attempt_{request.attempt:02d}.txt"
    before = _snapshot(request.workspace)
    prompt = (
        build_repair_prompt(request.feedback, request.entrypoint)
        if request.feedback
        else build_generation_prompt(request.research_plan, request.entrypoint)
    )
    command, env = command_factory(request, final_message_path)
    started = datetime.now(timezone.utc).isoformat()
    _append_event(
        event_log,
        {
            "event": "agent_started",
            "backend": backend,
            "attempt": request.attempt,
            "started_at": started,
            "command": command,
        },
    )

    try:
        completed = subprocess.run(
            command,
            cwd=request.workspace,
            env=env,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=request.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        message = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        error = f"{backend} timed out after {request.timeout_seconds} seconds"
        status = "timeout"
        returncode = None
    except OSError as exc:
        message = ""
        error = str(exc)
        status = "failed"
        returncode = None
    else:
        returncode = completed.returncode
        message = ""
        if final_message_path.exists():
            message = final_message_path.read_text(encoding="utf-8", errors="replace")
        if not message:
            message = completed.stdout or completed.stderr or ""
        if completed.returncode != 0:
            status = "failed"
            error = (completed.stderr or completed.stdout or "").strip() or (
                f"{backend} exited with status {completed.returncode}"
            )
        elif not artifact.is_file() or artifact.stat().st_size == 0:
            status = "failed"
            error = f"{backend} did not create non-empty {request.entrypoint}"
        else:
            status = "completed"
            error = None

    changed = _changed_files(before, request.workspace)
    _append_event(
        event_log,
        {
            "event": "agent_finished",
            "backend": backend,
            "attempt": request.attempt,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "returncode": returncode,
            "changed_files": changed,
            "error": error,
        },
    )
    return AgentRunResult(
        status=status,
        backend=backend,
        workspace=request.workspace,
        entrypoint=request.entrypoint,
        session_id=request.workspace.name,
        attempt=request.attempt,
        changed_files=changed,
        final_message=message,
        event_log=event_log,
        error=error,
    )


def continuation_request(previous: AgentRunResult, feedback: str, research_plan: dict) -> AgentRunRequest:
    """Create a repair turn in the same workspace with prior artifacts intact."""

    return AgentRunRequest(
        research_plan=research_plan,
        workspace=previous.workspace,
        entrypoint=previous.entrypoint,
        attempt=previous.attempt + 1,
        feedback=feedback,
    )


def publish_compatibility_artifact(result: AgentRunResult, destination: Path) -> Optional[str]:
    """Keep the historical API while treating the workspace as source of truth."""

    if result.status != "completed":
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result.output_path, destination)
    return str(destination)
