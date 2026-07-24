"""Deterministic verification for generated code artifacts.

The verifier deliberately does not ask an LLM to judge its own output.  It
checks a small, explicit contract and returns machine-readable feedback that a
code-generation runner can send back to the same agent session for repair.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


DEFAULT_MAX_REPAIR_ROUNDS = 5


@dataclass(frozen=True)
class VerificationCommand:
    """A command executed relative to the isolated generation workspace."""

    name: str
    argv: Sequence[str]
    timeout_seconds: int = 60

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Verification command name cannot be empty")
        if not self.argv:
            raise ValueError(f"Verification command '{self.name}' has no argv")
        if self.timeout_seconds <= 0:
            raise ValueError("Verification command timeout_seconds must be positive")


@dataclass(frozen=True)
class VerificationContract:
    """Files and commands that a generated artifact must satisfy."""

    entrypoint: str = "result.py"
    required_files: Sequence[str] = field(default_factory=tuple)
    commands: Optional[Sequence[VerificationCommand]] = None
    result_json: Optional[str] = None
    required_result_fields: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class VerificationCheck:
    """Result of one deterministic check."""

    name: str
    passed: bool
    message: str
    command: Optional[List[str]] = None
    exit_code: Optional[int] = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationResult:
    """Aggregate verifier result, suitable for persistence as JSON."""

    passed: bool
    workspace: str
    entrypoint: str
    checks: Sequence[VerificationCheck]

    @property
    def failures(self) -> List[VerificationCheck]:
        return [check for check in self.checks if not check.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "workspace": self.workspace,
            "entrypoint": self.entrypoint,
            "checks": [check.to_dict() for check in self.checks],
        }

    @property
    def report(self) -> Dict[str, Any]:
        """Machine-readable report alias for runner integrations."""

        return self.to_dict()

    def repair_feedback(
        self,
        round_number: int,
        max_rounds: int = DEFAULT_MAX_REPAIR_ROUNDS,
    ) -> Dict[str, Any]:
        """Return bounded, structured feedback for an agent repair turn."""

        if max_rounds < 1 or max_rounds > DEFAULT_MAX_REPAIR_ROUNDS:
            raise ValueError(
                f"max_rounds must be between 1 and {DEFAULT_MAX_REPAIR_ROUNDS}"
            )
        if round_number < 1 or round_number > max_rounds:
            raise ValueError("round_number must be between 1 and max_rounds")
        return {
            "kind": "code_generation_verification",
            "round": round_number,
            "max_rounds": max_rounds,
            "passed": self.passed,
            "remaining_rounds": max_rounds - round_number,
            "failures": [check.to_dict() for check in self.failures],
        }

    def repair_feedback_text(
        self,
        round_number: int,
        max_rounds: int = DEFAULT_MAX_REPAIR_ROUNDS,
    ) -> str:
        """Serialize repair feedback for a text-based CLI/ACP prompt."""

        return json.dumps(
            self.repair_feedback(round_number, max_rounds),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


class CodeGenerationVerifier:
    """Verify generated files without modifying source or using an LLM."""

    def verify(
        self,
        workspace: Union[Path, str],
        contract: Optional[VerificationContract] = None,
    ) -> VerificationResult:
        contract = contract or VerificationContract()
        workspace_path = Path(workspace).resolve()
        checks: List[VerificationCheck] = []

        if not workspace_path.is_dir():
            check = VerificationCheck(
                name="workspace_exists",
                passed=False,
                message=f"Workspace does not exist or is not a directory: {workspace_path}",
            )
            return VerificationResult(
                passed=False,
                workspace=str(workspace_path),
                entrypoint=contract.entrypoint,
                checks=[check],
            )

        entrypoint = self._resolve_inside(workspace_path, contract.entrypoint)
        if entrypoint is None:
            checks.append(
                VerificationCheck(
                    name="entrypoint_path",
                    passed=False,
                    message=f"Entrypoint escapes workspace: {contract.entrypoint}",
                )
            )
            return self._result(workspace_path, contract, checks)

        required = list(dict.fromkeys([contract.entrypoint, *contract.required_files]))
        for relative_path in required:
            target = self._resolve_inside(workspace_path, relative_path)
            passed = target is not None and target.is_file()
            checks.append(
                VerificationCheck(
                    name=f"file_exists:{relative_path}",
                    passed=passed,
                    message=(
                        f"Required file exists: {relative_path}"
                        if passed
                        else f"Required file is missing or outside workspace: {relative_path}"
                    ),
                )
            )

        if entrypoint.is_file():
            checks.append(
                self._run_command(
                    workspace_path,
                    VerificationCommand(
                        name="python_compile",
                        argv=(sys.executable, "-m", "py_compile", contract.entrypoint),
                    ),
                )
            )

            commands = contract.commands
            if commands is None:
                commands = (
                    VerificationCommand(
                        name="cli_help",
                        argv=(sys.executable, contract.entrypoint, "--help"),
                    ),
                )
            for command in commands:
                checks.append(self._run_command(workspace_path, command))

        if contract.result_json:
            checks.append(self._check_result_json(workspace_path, contract))

        return self._result(workspace_path, contract, checks)

    @staticmethod
    def _resolve_inside(workspace: Path, relative_path: str) -> Optional[Path]:
        candidate = (workspace / relative_path).resolve()
        try:
            candidate.relative_to(workspace)
        except ValueError:
            return None
        return candidate

    @staticmethod
    def _tail(value: str, limit: int = 4000) -> str:
        return value[-limit:]

    def _run_command(
        self,
        workspace: Path,
        command: VerificationCommand,
    ) -> VerificationCheck:
        argv = [str(value) for value in command.argv]
        env = {
            **os.environ,
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        try:
            completed = subprocess.run(
                argv,
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=command.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return VerificationCheck(
                name=command.name,
                passed=False,
                message=f"Command timed out after {command.timeout_seconds} seconds",
                command=argv,
                stdout_tail=self._tail(self._text(exc.stdout)),
                stderr_tail=self._tail(self._text(exc.stderr)),
                details={"timeout_seconds": command.timeout_seconds},
            )
        except OSError as exc:
            return VerificationCheck(
                name=command.name,
                passed=False,
                message=f"Command could not start: {exc}",
                command=argv,
                details={"exception": type(exc).__name__},
            )

        passed = completed.returncode == 0
        return VerificationCheck(
            name=command.name,
            passed=passed,
            message=(
                "Command completed successfully"
                if passed
                else f"Command exited with status {completed.returncode}"
            ),
            command=argv,
            exit_code=completed.returncode,
            stdout_tail=self._tail(completed.stdout),
            stderr_tail=self._tail(completed.stderr),
        )

    def _check_result_json(
        self,
        workspace: Path,
        contract: VerificationContract,
    ) -> VerificationCheck:
        assert contract.result_json is not None
        result_path = self._resolve_inside(workspace, contract.result_json)
        if result_path is None or not result_path.is_file():
            return VerificationCheck(
                name="result_json",
                passed=False,
                message=f"Result JSON is missing or outside workspace: {contract.result_json}",
            )
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return VerificationCheck(
                name="result_json",
                passed=False,
                message=f"Result JSON is not valid UTF-8 JSON: {exc}",
                details={"exception": type(exc).__name__},
            )
        if not isinstance(payload, Mapping):
            return VerificationCheck(
                name="result_json",
                passed=False,
                message="Result JSON must contain an object at the top level",
                details={"actual_type": type(payload).__name__},
            )
        missing = [field for field in contract.required_result_fields if field not in payload]
        return VerificationCheck(
            name="result_json",
            passed=not missing,
            message=(
                "Result JSON satisfies the required structure"
                if not missing
                else f"Result JSON is missing fields: {', '.join(missing)}"
            ),
            details={"missing_fields": missing},
        )

    @staticmethod
    def _text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    @staticmethod
    def _result(
        workspace: Path,
        contract: VerificationContract,
        checks: Sequence[VerificationCheck],
    ) -> VerificationResult:
        return VerificationResult(
            passed=bool(checks) and all(check.passed for check in checks),
            workspace=str(workspace),
            entrypoint=contract.entrypoint,
            checks=checks,
        )


def verify_generated_code(
    code_file: Union[Path, str],
    *,
    workspace: Optional[Union[Path, str]] = None,
    acceptance_commands: Optional[Sequence[Sequence[str]]] = None,
    required_files: Sequence[str] = (),
    result_json: Optional[str] = None,
    required_result_fields: Sequence[str] = (),
    command_timeout_seconds: int = 60,
) -> VerificationResult:
    """Convenience API for runners that have only a generated code path.

    ``acceptance_commands=None`` applies the default ``--help`` check. Passing
    an empty sequence intentionally disables runtime commands while retaining
    file and compile checks.
    """

    code_path = Path(code_file).resolve()
    workspace_path = Path(workspace).resolve() if workspace is not None else code_path.parent
    try:
        entrypoint = str(code_path.relative_to(workspace_path))
    except ValueError:
        entrypoint = str(code_path)

    commands: Optional[Sequence[VerificationCommand]]
    if acceptance_commands is None:
        commands = None
    else:
        commands = tuple(
            VerificationCommand(
                name=f"acceptance_{index:02d}",
                argv=tuple(argv),
                timeout_seconds=command_timeout_seconds,
            )
            for index, argv in enumerate(acceptance_commands, start=1)
        )

    return CodeGenerationVerifier().verify(
        workspace_path,
        VerificationContract(
            entrypoint=entrypoint,
            required_files=required_files,
            commands=commands,
            result_json=result_json,
            required_result_fields=required_result_fields,
        ),
    )


__all__ = [
    "DEFAULT_MAX_REPAIR_ROUNDS",
    "CodeGenerationVerifier",
    "VerificationCheck",
    "VerificationCommand",
    "VerificationContract",
    "VerificationResult",
    "verify_generated_code",
]
