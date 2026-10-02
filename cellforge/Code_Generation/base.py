"""Shared interface for CellForge code generation backends."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from ..paths import workspace_root
from .contracts import AgentRunRequest, AgentRunResult


class CodeGenerationBackend(ABC):
    """Backend contract used by both code_generation and autorun."""

    name = "base"

    def __init__(self):
        self.project_root = workspace_root()
        self.last_result: Optional[AgentRunResult] = None

    @abstractmethod
    def generate_code(
        self,
        research_plan: Dict[str, Any],
        output_dir: str = "data/codes",
        task_id: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a Python file and return its path."""

    def scoped_plan(self, research_plan: Dict[str, Any], task_id: Optional[str]) -> Dict[str, Any]:
        if not task_id:
            return research_plan

        task_blocks = research_plan.get("task_wise_plan", []) or []
        target = None
        for item in task_blocks:
            if str(item.get("task_id", "")).strip() == str(task_id).strip():
                target = item
                break
        if target is None:
            raise ValueError(f"Task id not found in task_wise_plan: {task_id}")

        scoped = dict(research_plan)
        scoped["selected_task"] = target
        scoped["selected_task_id"] = task_id
        return scoped

    def output_file(self, output_dir: str, task_id: Optional[str]) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"result_{task_id}.py" if task_id else "result.py"
        return output_path / filename

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        """Run a structured coding-agent task.

        Legacy backends may continue to expose only ``generate_code``.
        """

        raise NotImplementedError(f"{self.__class__.__name__} does not support structured agent runs")

    def continue_run(self, previous: AgentRunResult, feedback: str) -> AgentRunResult:
        """Repair an artifact in the same workspace."""

        raise NotImplementedError(f"{self.__class__.__name__} does not support repair continuation")
