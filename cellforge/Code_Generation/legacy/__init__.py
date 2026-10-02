"""Compatibility errors for removed code-generation integrations."""


class LegacyBackendUnavailableError(ValueError):
    """A configured legacy backend cannot satisfy the current contract."""

    @classmethod
    def for_backend(cls, backend_name: str) -> "LegacyBackendUnavailableError":
        return cls(
            f"Code generation backend '{backend_name}' is no longer supported. "
            "The legacy OpenHands Docker/UI integration never implemented the "
            "current research-plan-to-artifact contract. Use 'codex'. "
            "For a new OpenHands integration, implement a coding-agent "
            "runner with an isolated workspace and deterministic verification; "
            "see cellforge/Code_Generation/legacy/README.md."
        )


__all__ = ["LegacyBackendUnavailableError"]
