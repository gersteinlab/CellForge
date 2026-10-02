"""Workspace lifecycle for the Codex CLI code-generation backend."""

import json
import os
import re
import shutil
import subprocess
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .codex_events import AgentEventLog, CodexJsonlRecorder, safe_text
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
            relative = path.relative_to(workspace)
            if "logs" in relative.parts or "__pycache__" in relative.parts:
                continue
            stat = path.stat()
            snapshot[str(relative)] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def _changed_files(before: Dict[str, Tuple[int, int]], workspace: Path) -> List[str]:
    after = _snapshot(workspace)
    return sorted(name for name, signature in after.items() if before.get(name) != signature)


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
    os.chmod(logs_dir, 0o700)
    event_log = logs_dir / "agent_events.jsonl"
    raw_event_log = logs_dir / f"{backend}_events_raw_attempt_{request.attempt:02d}.jsonl"
    stderr_log = logs_dir / f"{backend}_stderr_attempt_{request.attempt:02d}.log"
    final_message_path = logs_dir / f"{backend}_attempt_{request.attempt:02d}.txt"
    recorder = AgentEventLog(event_log)
    before = _snapshot(request.workspace)
    prompt = (
        build_repair_prompt(request.feedback, request.entrypoint)
        if request.feedback
        else build_generation_prompt(request.research_plan, request.entrypoint)
    )
    command, env = command_factory(request, final_message_path)
    recorder.append(
        "agent_started",
        backend=backend,
        attempt=request.attempt,
        command=command,
    )

    stream_recorder = CodexJsonlRecorder(
        raw_log=raw_event_log,
        event_log=recorder,
        backend=backend,
        attempt=request.attempt,
    )
    stderr_tail: deque[str] = deque(maxlen=200)
    stderr_log.write_text("", encoding="utf-8")
    os.chmod(stderr_log, 0o600)
    reader_errors: list[BaseException] = []
    process: Optional[subprocess.Popen] = None
    stdout_thread: Optional[threading.Thread] = None
    stderr_thread: Optional[threading.Thread] = None

    def read_stdout() -> None:
        try:
            assert process is not None and process.stdout is not None
            for line in process.stdout:
                stream_recorder.record_line(line)
        except BaseException as exc:  # surfaced on the controlling thread below
            reader_errors.append(exc)

    def read_stderr() -> None:
        try:
            assert process is not None and process.stderr is not None
            with stderr_log.open("a", encoding="utf-8") as stream:
                for line in process.stderr:
                    stream.write(line)
                    stream.flush()
                    stderr_tail.append(line)
        except BaseException as exc:  # surfaced on the controlling thread below
            reader_errors.append(exc)

    try:
        process = subprocess.Popen(
            command,
            cwd=request.workspace,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stdout_thread = threading.Thread(
            target=read_stdout, name="codex-stdout", daemon=True
        )
        stderr_thread = threading.Thread(
            target=read_stderr, name="codex-stderr", daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()
        assert process.stdin is not None
        process.stdin.write(prompt)
        process.stdin.close()
        returncode = process.wait(timeout=request.timeout_seconds)
        stdout_thread.join()
        stderr_thread.join()
        if reader_errors:
            raise OSError(f"Failed to record Codex output: {reader_errors[0]}")
    except subprocess.TimeoutExpired:
        assert process is not None
        process.kill()
        process.wait()
        assert stdout_thread is not None and stderr_thread is not None
        stdout_thread.join()
        stderr_thread.join()
        message = ""
        error = f"{backend} timed out after {request.timeout_seconds} seconds"
        status = "timeout"
        returncode = None
    except OSError as exc:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        for thread in (stdout_thread, stderr_thread):
            if thread is not None:
                thread.join()
        message = ""
        error = str(exc)
        status = "failed"
        returncode = None
    else:
        message = ""
        if final_message_path.exists():
            message = final_message_path.read_text(encoding="utf-8", errors="replace")
        if not message:
            message = "".join(stderr_tail)
        if returncode != 0:
            status = "failed"
            error = "".join(stderr_tail).strip() or (
                f"{backend} exited with status {returncode}"
            )
        elif not artifact.is_file() or artifact.stat().st_size == 0:
            status = "failed"
            error = f"{backend} did not create non-empty {request.entrypoint}"
        else:
            status = "completed"
            error = None

    if final_message_path.exists():
        os.chmod(final_message_path, 0o600)
    stream_state = stream_recorder.state
    changed = _changed_files(before, request.workspace)
    recorder.append(
        "agent_finished",
        backend=backend,
        attempt=request.attempt,
        status=status,
        returncode=returncode,
        changed_files=changed,
        error=safe_text(error),
        raw_event_count=stream_state.raw_event_count,
        malformed_line_count=stream_state.malformed_line_count,
    )
    return AgentRunResult(
        status=status,
        backend=backend,
        workspace=request.workspace,
        entrypoint=request.entrypoint,
        session_id=stream_state.session_id or request.workspace.name,
        attempt=request.attempt,
        changed_files=changed,
        final_message=message,
        event_log=event_log,
        raw_event_log=raw_event_log,
        stderr_log=stderr_log,
        event_count=stream_state.raw_event_count,
        usage=stream_state.usage,
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
