"""Verified coding-agent backends for CellForge.

The compatibility helpers still return ``result*.py`` paths, while each
backend now runs in an isolated workspace and exposes its structured result as
``backend.last_result``.
"""

from .base import CodeGenerationBackend
from .codex_backend import CodexCodeGenerator
from .contracts import AgentRunRequest, AgentRunResult
from .orchestrator import run_with_verification
from .registry import create_backend, list_backends
from .verifier import (
    CodeGenerationVerifier,
    VerificationContract,
    VerificationResult,
    verify_generated_code,
)


def generate_code_from_plan(research_plan, output_dir="data/codes", backend=None):
    generator = create_backend(backend)
    return generator.generate_code(research_plan, output_dir=output_dir)


def generate_code_from_plan_task(research_plan, task_id, output_dir="data/codes", backend=None):
    generator = create_backend(backend)
    return generator.generate_code(research_plan, output_dir=output_dir, task_id=task_id)


__all__ = [
    "CodeGenerationBackend",
    "CodexCodeGenerator",
    "AgentRunRequest",
    "AgentRunResult",
    "CodeGenerationVerifier",
    "VerificationContract",
    "VerificationResult",
    "create_backend",
    "generate_code_from_plan",
    "generate_code_from_plan_task",
    "list_backends",
    "run_with_verification",
    "verify_generated_code",
]
