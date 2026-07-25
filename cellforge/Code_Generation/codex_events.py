"""Normalize Codex CLI JSONL events into CellForge's audit timeline."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


_SECRET_ASSIGNMENT = re.compile(
    r"(?i)([\"']?(?:[a-z0-9_]*api[_-]?key|[a-z0-9_]*token|password)"
    r"[\"']?\s*[:=]\s*[\"']?)([^\"'\s,}]+)"
)
_AUTHORIZATION = re.compile(
    r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?)(?:bearer\s+)?([^\"'\s,}]+)"
)
_MAX_TEXT = 4000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_text(value: Any, limit: int = _MAX_TEXT) -> str:
    """Return bounded text with common inline credentials removed."""

    text = str(value or "")
    text = _AUTHORIZATION.sub(r"\1[REDACTED]", text)
    text = _SECRET_ASSIGNMENT.sub(r"\1[REDACTED]", text)
    if len(text) > limit:
        return text[:limit] + f"\n...[truncated {len(text) - limit} characters]"
    return text


class AgentEventLog:
    """Append ordered, provider-neutral events to one JSONL timeline."""

    def __init__(self, path: Path):
        self.path = path
        self.sequence = self._last_sequence()

    def _last_sequence(self) -> int:
        if not self.path.exists():
            return 0
        last = 0
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                last = max(last, int(json.loads(line).get("sequence", 0)))
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
        return last

    def append(self, event: str, *, backend: str, attempt: int, **details: Any) -> dict:
        self.sequence += 1
        payload = {
            "sequence": self.sequence,
            "timestamp": utc_now(),
            "event": event,
            "backend": backend,
            "attempt": attempt,
            **details,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        os.chmod(self.path, 0o600)
        return payload


@dataclass
class CodexEventState:
    """State accumulated while consuming one Codex JSONL stream."""

    session_id: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    raw_event_count: int = 0
    malformed_line_count: int = 0


def _item_paths(item: dict) -> list[str]:
    paths: list[str] = []
    for change in item.get("changes") or []:
        if isinstance(change, dict) and change.get("path"):
            paths.append(str(change["path"]))
    if item.get("path"):
        paths.append(str(item["path"]))
    return sorted(set(paths))


def normalize_codex_event(event: dict) -> tuple[str, dict]:
    """Map a current or future Codex event to a compact stable representation."""

    provider_type = str(event.get("type") or "unknown")
    details: dict = {"provider_event": provider_type}

    if provider_type == "thread.started":
        return "codex_thread_started", {
            **details,
            "thread_id": event.get("thread_id"),
        }
    if provider_type == "turn.started":
        return "turn_started", details
    if provider_type == "turn.completed":
        return "turn_completed", {
            **details,
            "usage": event.get("usage") or {},
        }
    if provider_type in {"turn.failed", "error"}:
        error = event.get("error")
        if isinstance(error, dict):
            error = error.get("message") or json.dumps(error, ensure_ascii=False)
        return "turn_failed", {
            **details,
            "error": safe_text(error or event.get("message")),
        }

    if provider_type in {"item.started", "item.updated", "item.completed"}:
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        item_type = str(item.get("type") or "unknown")
        phase = provider_type.rsplit(".", 1)[-1]
        details.update({"item_type": item_type, "phase": phase, "item_id": item.get("id")})

        if item_type == "command_execution":
            details.update(
                {
                    "command": safe_text(item.get("command")),
                    "status": item.get("status"),
                    "exit_code": item.get("exit_code"),
                    "output": safe_text(
                        item.get("aggregated_output") or item.get("output")
                    ),
                }
            )
            return f"command_{phase}", details
        if item_type == "file_change":
            details.update({"paths": _item_paths(item), "status": item.get("status")})
            return "file_changed" if phase == "completed" else f"file_change_{phase}", details
        if item_type in {"mcp_tool_call", "tool_call"}:
            details.update(
                {
                    "tool": item.get("server")
                    or item.get("tool")
                    or item.get("name"),
                    "status": item.get("status"),
                    "error": safe_text(item.get("error")),
                }
            )
            return f"tool_{phase}", details
        if item_type == "agent_message":
            details["message"] = safe_text(item.get("text") or item.get("message"))
            return "agent_message", details
        if item_type == "reasoning":
            # Codex JSONL exposes a user-facing progress summary, not hidden
            # chain-of-thought. Keep it bounded like every other text field.
            details["summary"] = safe_text(item.get("text") or item.get("summary"))
            return "agent_progress", details
        return "codex_item", details

    return "codex_event", details


def record_codex_jsonl(
    lines: Iterable[str],
    *,
    raw_log: Path,
    event_log: AgentEventLog,
    backend: str,
    attempt: int,
) -> CodexEventState:
    """Persist raw provider JSONL and append normalized events."""

    recorder = CodexJsonlRecorder(
        raw_log=raw_log,
        event_log=event_log,
        backend=backend,
        attempt=attempt,
    )
    for line in lines:
        recorder.record_line(line)
    return recorder.state


class CodexJsonlRecorder:
    """Record provider events one line at a time as Codex emits them."""

    def __init__(
        self,
        *,
        raw_log: Path,
        event_log: AgentEventLog,
        backend: str,
        attempt: int,
    ):
        self.raw_log = raw_log
        self.event_log = event_log
        self.backend = backend
        self.attempt = attempt
        self.state = CodexEventState()
        self.raw_log.parent.mkdir(parents=True, exist_ok=True)
        self.raw_log.write_text("", encoding="utf-8")
        os.chmod(self.raw_log, 0o600)

    def record_line(self, raw_line: str) -> None:
        line = raw_line.rstrip("\r\n")
        if not line:
            return
        with self.raw_log.open("a", encoding="utf-8") as raw_stream:
            raw_stream.write(line + "\n")
        try:
            provider_event = json.loads(line)
            if not isinstance(provider_event, dict):
                raise ValueError("event is not a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            self.state.malformed_line_count += 1
            self.event_log.append(
                "codex_stream_warning",
                backend=self.backend,
                attempt=self.attempt,
                error=safe_text(exc),
                line=safe_text(line),
            )
            return

        self.state.raw_event_count += 1
        event_name, details = normalize_codex_event(provider_event)
        self.event_log.append(
            event_name,
            backend=self.backend,
            attempt=self.attempt,
            **details,
        )
        if provider_event.get("type") == "thread.started":
            self.state.session_id = (
                provider_event.get("thread_id") or self.state.session_id
            )
        if provider_event.get("type") == "turn.completed":
            usage = provider_event.get("usage")
            if isinstance(usage, dict):
                self.state.usage = usage
