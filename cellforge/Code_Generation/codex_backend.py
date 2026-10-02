"""Codex CLI coding-agent backend."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import CodeGenerationBackend
from .contracts import AgentRunRequest, AgentRunResult
from .orchestrator import run_with_verification
from .runner import (
    continuation_request,
    create_task_workspace,
    execute_agent,
    publish_compatibility_artifact,
)
from .utils import load_project_env, truthy_env

logger = logging.getLogger(__name__)


class CodexCodeGenerator(CodeGenerationBackend):
    """Run Codex as a workspace-editing agent with its OS sandbox enabled."""

    name = "codex"

    def __init__(self):
        super().__init__()
        load_project_env(self.project_root)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.codex_api_key = os.getenv("CODEX_API_KEY")
        # An empty value lets the installed Codex CLI select its current
        # supported default. MODEL_NAME belongs to the general LLM pipeline
        # and may not be a valid Codex CLI model.
        self.model = os.getenv("CODEX_MODEL", "").strip() or None
        self.codex_bin = os.getenv("CODEX_CLI_BIN", "codex")
        self.auth_mode = os.getenv("CODEX_AUTH_MODE", "local").strip().lower()
        self.pass_openai_env = truthy_env("CODEX_PASS_OPENAI_ENV")
        if self.auth_mode not in {"local", "api"}:
            raise ValueError("CODEX_AUTH_MODE must be 'local' or 'api'")

    def _environment(self) -> dict:
        env = os.environ.copy()
        if self.auth_mode == "local":
            # Default to the interactive `codex login` session even when the
            # broader CellForge process has provider keys configured.
            env.pop("OPENAI_API_KEY", None)
            env.pop("OPENAI_BASE_URL", None)
            return env

        if self.codex_api_key:
            # CODEX_API_KEY is CellForge's configuration alias; Codex itself
            # consumes the standard OpenAI credential variable.
            env["OPENAI_API_KEY"] = self.codex_api_key
            if self.base_url:
                env["OPENAI_BASE_URL"] = self.base_url
            else:
                env.pop("OPENAI_BASE_URL", None)
        elif self.pass_openai_env:
            if self.api_key:
                env["OPENAI_API_KEY"] = self.api_key
            if self.base_url:
                env["OPENAI_BASE_URL"] = self.base_url
        else:
            raise ValueError(
                "CODEX_AUTH_MODE=api requires CODEX_API_KEY, or "
                "CODEX_PASS_OPENAI_ENV=true with OPENAI_API_KEY"
            )
        return env

    def _command(self, request: AgentRunRequest, final_message: Path) -> Tuple[List[str], dict]:
        command = [
            self.codex_bin,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--json",
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
            "-C",
            str(request.workspace),
        ]
        if request.model:
            command.extend(["-m", request.model])
        command.extend(["-o", str(final_message), "-"])
        return command, self._environment()

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        request.model = request.model or self.model
        self.last_result = execute_agent(request, backend=self.name, command_factory=self._command)
        return self.last_result

    def continue_run(self, previous: AgentRunResult, feedback: str) -> AgentRunResult:
        plan_path = previous.workspace / "research_plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        request = continuation_request(previous, feedback, plan)
        request.model = self.model
        return self.run(request)

    def generate_code(
        self,
        research_plan: Dict[str, Any],
        output_dir: str = "data/codes",
        task_id: Optional[str] = None,
    ) -> Optional[str]:
        """Compatibility entrypoint returning the published Python file path."""

        try:
            scoped_plan = self.scoped_plan(research_plan, task_id)
            workspace = create_task_workspace(output_dir, self.name, task_id)
            request = AgentRunRequest(
                research_plan=scoped_plan,
                workspace=workspace,
                task_id=task_id,
                model=self.model,
            )
            result, _verification = run_with_verification(self, request)
            if result.status != "completed":
                logger.error("Codex generation failed: %s", result.error)
            return publish_compatibility_artifact(result, self.output_file(output_dir, task_id))
        except Exception as exc:
            logger.error("Error during Codex code generation: %s", exc)
            return None
