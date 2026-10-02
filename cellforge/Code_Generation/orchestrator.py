"""Bounded generate-verify-repair orchestration for coding-agent backends."""

import json
import os
from pathlib import Path
from typing import Optional, Tuple

from .base import CodeGenerationBackend
from .codex_events import AgentEventLog
from .contracts import AgentRunRequest, AgentRunResult
from .verifier import (
    DEFAULT_MAX_REPAIR_ROUNDS,
    VerificationResult,
    verify_generated_code,
)


def configured_max_repair_rounds() -> int:
    """Read and validate the global repair budget."""

    raw = os.getenv("CODEGEN_MAX_REPAIR_ROUNDS", str(DEFAULT_MAX_REPAIR_ROUNDS))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("CODEGEN_MAX_REPAIR_ROUNDS must be an integer") from exc
    if value < 1 or value > DEFAULT_MAX_REPAIR_ROUNDS:
        raise ValueError(
            f"CODEGEN_MAX_REPAIR_ROUNDS must be between 1 and "
            f"{DEFAULT_MAX_REPAIR_ROUNDS}"
        )
    return value


def _write_verification_report(
    result: AgentRunResult,
    verification: VerificationResult,
) -> Path:
    logs_dir = result.workspace / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    report_path = logs_dir / f"verification_attempt_{result.attempt:02d}.json"
    report_path.write_text(
        json.dumps(verification.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(report_path, 0o600)
    return report_path


def _record_orchestration_event(result: AgentRunResult, event: str, **details) -> None:
    if result.event_log is None:
        return
    AgentEventLog(result.event_log).append(
        event,
        backend=result.backend,
        attempt=result.attempt,
        **details,
    )


def run_with_verification(
    backend: CodeGenerationBackend,
    request: AgentRunRequest,
    *,
    max_rounds: Optional[int] = None,
) -> Tuple[AgentRunResult, VerificationResult]:
    """Run an agent and independently verify/repair its artifact.

    Repair turns reuse the same task-scoped workspace. The provider may implement
    a native conversation resume later; correctness does not depend on it
    because the workspace, plan, contract, prior artifact, and verifier
    feedback remain available on every turn.
    """

    repair_budget = max_rounds or configured_max_repair_rounds()
    if repair_budget < 1 or repair_budget > DEFAULT_MAX_REPAIR_ROUNDS:
        raise ValueError(
            f"max_rounds must be between 1 and {DEFAULT_MAX_REPAIR_ROUNDS}"
        )

    result = backend.run(request)
    _record_orchestration_event(result, "verification_started")
    verification = verify_generated_code(
        result.output_path,
        workspace=result.workspace,
    )
    report_path = _write_verification_report(result, verification)
    _record_orchestration_event(
        result,
        "verification_completed",
        passed=verification.passed,
        report=str(report_path.relative_to(result.workspace)),
        failed_checks=[check.name for check in verification.failures],
    )

    # Repair prompts are for artifacts that the agent successfully delivered
    # but that failed deterministic checks. Authentication, CLI startup,
    # timeout, and transport failures need operator action and must not consume
    # additional paid repair turns.
    while (
        result.status == "completed"
        and not verification.passed
        and result.attempt < repair_budget
    ):
        feedback = verification.repair_feedback_text(
            round_number=result.attempt,
            max_rounds=repair_budget,
        )
        _record_orchestration_event(
            result,
            "repair_started",
            next_attempt=result.attempt + 1,
            failed_checks=[check.name for check in verification.failures],
        )
        result = backend.continue_run(result, feedback)
        _record_orchestration_event(result, "verification_started")
        verification = verify_generated_code(
            result.output_path,
            workspace=result.workspace,
        )
        report_path = _write_verification_report(result, verification)
        _record_orchestration_event(
            result,
            "verification_completed",
            passed=verification.passed,
            report=str(report_path.relative_to(result.workspace)),
            failed_checks=[check.name for check in verification.failures],
        )

    if result.status == "completed" and not verification.passed:
        result.status = "failed"
        result.error = (
            f"Artifact failed deterministic verification after "
            f"{result.attempt} attempt(s)"
        )
    _record_orchestration_event(
        result,
        "generation_finished",
        status=result.status,
        passed=verification.passed,
        error=result.error,
    )
    return result, verification
