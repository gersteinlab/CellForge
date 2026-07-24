"""Structured contracts for coding-agent executions."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


AgentRunStatus = Literal["completed", "failed", "timeout"]


@dataclass
class AgentRunRequest:
    """Everything a coding agent needs for one task-scoped implementation."""

    research_plan: Dict[str, Any]
    workspace: Path
    task_id: Optional[str] = None
    entrypoint: str = "result.py"
    model: Optional[str] = None
    timeout_seconds: int = 1800
    attempt: int = 1
    feedback: Optional[str] = None


@dataclass
class AgentRunResult:
    """Auditable result of one coding-agent invocation."""

    status: AgentRunStatus
    backend: str
    workspace: Path
    entrypoint: str
    session_id: str
    attempt: int
    changed_files: List[str] = field(default_factory=list)
    final_message: str = ""
    event_log: Optional[Path] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def output_path(self) -> Path:
        return self.workspace / self.entrypoint
