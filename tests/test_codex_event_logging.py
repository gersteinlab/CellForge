import json
import os
import stat
import sys

from cellforge.Code_Generation.codex_events import AgentEventLog, record_codex_jsonl
from cellforge.Code_Generation.contracts import AgentRunRequest
from cellforge.Code_Generation.runner import execute_agent


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_codex_jsonl_is_preserved_and_normalized(tmp_path):
    raw_log = tmp_path / "raw.jsonl"
    event_log = tmp_path / "events.jsonl"
    lines = [
        json.dumps({"type": "thread.started", "thread_id": "thread-123"}),
        json.dumps({"type": "turn.started"}),
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "cmd-1",
                    "type": "command_execution",
                    "command": "python result.py --token=secret",
                    "status": "completed",
                    "exit_code": 0,
                    "aggregated_output": "ok",
                },
            }
        ),
        "{malformed",
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 12, "output_tokens": 8},
            }
        ),
    ]

    state = record_codex_jsonl(
        lines,
        raw_log=raw_log,
        event_log=AgentEventLog(event_log),
        backend="codex",
        attempt=1,
    )

    events = _read_jsonl(event_log)
    assert state.session_id == "thread-123"
    assert state.usage == {"input_tokens": 12, "output_tokens": 8}
    assert state.raw_event_count == 4
    assert state.malformed_line_count == 1
    assert [event["sequence"] for event in events] == list(range(1, 6))
    assert {event["event"] for event in events} >= {
        "codex_thread_started",
        "command_completed",
        "codex_stream_warning",
        "turn_completed",
    }
    command = next(event for event in events if event["event"] == "command_completed")
    assert "secret" not in command["command"]
    assert raw_log.read_text(encoding="utf-8").splitlines() == lines
    assert stat.S_IMODE(raw_log.stat().st_mode) == 0o600
    assert stat.S_IMODE(event_log.stat().st_mode) == 0o600


def test_secret_redaction_covers_environment_json_and_bearer(tmp_path):
    event_log = tmp_path / "events.jsonl"
    lines = [
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": (
                        "OPENAI_API_KEY=sk-one "
                        '{"CODEX_API_KEY":"sk-two"} '
                        "Authorization: Bearer sk-three"
                    ),
                },
            }
        )
    ]

    record_codex_jsonl(
        lines,
        raw_log=tmp_path / "raw.jsonl",
        event_log=AgentEventLog(event_log),
        backend="codex",
        attempt=1,
    )

    normalized = event_log.read_text(encoding="utf-8")
    for secret in ("sk-one", "sk-two", "sk-three"):
        assert secret not in normalized


def test_execute_agent_records_codex_artifacts(tmp_path):
    fake_cli = tmp_path / "fake_codex.py"
    fake_cli.write_text(
        "import json\n"
        "import pathlib\n"
        "import sys\n"
        "sys.stdin.read()\n"
        "pathlib.Path('result.py').write_text("
        "\"import argparse\\nargparse.ArgumentParser().parse_args()\\n\", "
        "encoding='utf-8')\n"
        "print(json.dumps({'type': 'thread.started', 'thread_id': 'local-thread'}))\n"
        "print(json.dumps({'type': 'turn.started'}))\n"
        "print(json.dumps({'type': 'item.completed', 'item': {"
        "'id': 'file-1', 'type': 'file_change', 'status': 'completed', "
        "'changes': [{'path': 'result.py', 'kind': 'add'}]}}))\n"
        "print(json.dumps({'type': 'turn.completed', 'usage': {"
        "'input_tokens': 21, 'output_tokens': 13}}))\n"
        "print('diagnostic', file=sys.stderr)\n",
        encoding="utf-8",
    )

    def command_factory(_request, final_message):
        final_message.write_text("implemented", encoding="utf-8")
        return [sys.executable, str(fake_cli)], os.environ.copy()

    result = execute_agent(
        AgentRunRequest(research_plan={"task": "fixture"}, workspace=tmp_path / "work"),
        backend="codex",
        command_factory=command_factory,
    )

    assert result.status == "completed"
    assert result.session_id == "local-thread"
    assert result.event_count == 4
    assert result.usage == {"input_tokens": 21, "output_tokens": 13}
    assert result.raw_event_log and result.raw_event_log.exists()
    assert result.stderr_log and result.stderr_log.read_text(encoding="utf-8") == "diagnostic\n"
    assert result.changed_files == ["result.py"]
    events = _read_jsonl(result.event_log)
    assert events[0]["event"] == "agent_started"
    assert events[-1]["event"] == "agent_finished"
    assert any(event["event"] == "file_changed" for event in events)


def test_timeout_preserves_partial_codex_stream(tmp_path):
    fake_cli = tmp_path / "slow_codex.py"
    fake_cli.write_text(
        "import json\n"
        "import sys\n"
        "import time\n"
        "sys.stdin.read()\n"
        "print(json.dumps({'type': 'thread.started', 'thread_id': 'partial-thread'}), flush=True)\n"
        "print('partial diagnostic', file=sys.stderr, flush=True)\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )

    def command_factory(_request, _final_message):
        return [sys.executable, str(fake_cli)], os.environ.copy()

    result = execute_agent(
        AgentRunRequest(
            research_plan={},
            workspace=tmp_path / "work",
            timeout_seconds=0.2,
        ),
        backend="codex",
        command_factory=command_factory,
    )

    assert result.status == "timeout"
    assert result.session_id == "partial-thread"
    assert result.event_count == 1
    assert "thread.started" in result.raw_event_log.read_text(encoding="utf-8")
    assert "partial diagnostic" in result.stderr_log.read_text(encoding="utf-8")
