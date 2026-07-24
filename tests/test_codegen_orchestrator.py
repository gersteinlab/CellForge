from pathlib import Path

from cellforge.Code_Generation.base import CodeGenerationBackend
from cellforge.Code_Generation.contracts import AgentRunRequest, AgentRunResult
from cellforge.Code_Generation.orchestrator import run_with_verification


class RepairingBackend(CodeGenerationBackend):
    name = "fake"

    def generate_code(self, research_plan, output_dir="data/codes", task_id=None):
        raise AssertionError("compatibility API is not used in this test")

    def run(self, request):
        request.workspace.mkdir(parents=True, exist_ok=True)
        request.workspace.joinpath(request.entrypoint).write_text(
            "def broken(:\n",
            encoding="utf-8",
        )
        return self._result(request)

    def continue_run(self, previous, feedback):
        assert "python_compile" in feedback
        previous.output_path.write_text(
            "import argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.parse_args()\n",
            encoding="utf-8",
        )
        request = AgentRunRequest(
            research_plan={},
            workspace=previous.workspace,
            entrypoint=previous.entrypoint,
            attempt=previous.attempt + 1,
        )
        return self._result(request)

    def _result(self, request):
        result = AgentRunResult(
            status="completed",
            backend=self.name,
            workspace=Path(request.workspace),
            entrypoint=request.entrypoint,
            session_id="cellforge-test-session",
            attempt=request.attempt,
        )
        self.last_result = result
        return result


class FailingBackend(CodeGenerationBackend):
    name = "failing"

    def __init__(self):
        super().__init__()
        self.run_calls = 0
        self.continue_calls = 0

    def generate_code(self, research_plan, output_dir="data/codes", task_id=None):
        raise AssertionError("compatibility API is not used in this test")

    def run(self, request):
        self.run_calls += 1
        request.workspace.mkdir(parents=True, exist_ok=True)
        return AgentRunResult(
            status="failed",
            backend=self.name,
            workspace=request.workspace,
            entrypoint=request.entrypoint,
            session_id="cellforge-failed-session",
            attempt=request.attempt,
            error="provider authentication failed",
        )

    def continue_run(self, previous, feedback):
        self.continue_calls += 1
        raise AssertionError("process failures must not trigger a repair turn")


def test_orchestrator_repairs_in_same_workspace(tmp_path):
    backend = RepairingBackend()
    request = AgentRunRequest(research_plan={}, workspace=tmp_path)

    result, verification = run_with_verification(
        backend,
        request,
        max_rounds=2,
    )

    assert result.status == "completed"
    assert result.attempt == 2
    assert result.workspace == tmp_path
    assert verification.passed
    assert (tmp_path / "logs" / "verification_attempt_01.json").exists()
    assert (tmp_path / "logs" / "verification_attempt_02.json").exists()


def test_orchestrator_does_not_retry_or_mask_process_failure(tmp_path):
    backend = FailingBackend()
    request = AgentRunRequest(research_plan={}, workspace=tmp_path)

    result, verification = run_with_verification(
        backend,
        request,
        max_rounds=5,
    )

    assert result.status == "failed"
    assert result.error == "provider authentication failed"
    assert result.attempt == 1
    assert backend.run_calls == 1
    assert backend.continue_calls == 0
    assert not verification.passed
    assert (tmp_path / "logs" / "verification_attempt_01.json").exists()
