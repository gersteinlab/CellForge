"""Deprecated import shim for the removed OpenHands backend.

This module remains importable for downstream code that referenced the class,
but the registry does not advertise it and construction fails immediately with
an actionable migration error.
"""

from ..base import CodeGenerationBackend
from . import LegacyBackendUnavailableError


class OpenHandsLegacyCodeGenerator(CodeGenerationBackend):
    """Import-compatible tombstone for the non-functional legacy adapter."""

    name = "openhands-legacy"

    def __init__(self):
        raise LegacyBackendUnavailableError.for_backend(self.name)

    def generate_code(self, research_plan, output_dir="data/codes", task_id=None):
        """Satisfy the abstract interface for introspection-only imports."""

        raise LegacyBackendUnavailableError.for_backend(self.name)
