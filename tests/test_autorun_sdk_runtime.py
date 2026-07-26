import json
from types import SimpleNamespace

from cellforge.autorun import runner


def test_codex_sdk_uses_writable_runtime_directory(tmp_path, monkeypatch):
    code_file = tmp_path / "result.py"
    code_file.write_text("import os\n" + "# existing code\n" * 20, encoding="utf-8")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    calls = []

    monkeypatch.setattr(runner.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        runtime_dir = logs_dir / ".codex_sdk_runtime"
        if command[0] == "npm":
            (runtime_dir / "node_modules" / "@openai" / "codex-sdk").mkdir(
                parents=True
            )
        else:
            output_path = command[command.index("--out-file") + 1]
            updated = "import os\n" + "# generated code\n" * 30
            runner.Path(output_path).write_text(updated, encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    runner._run_codex_autoresearch(
        code_file=code_file,
        logs_dir=logs_dir,
        train_data="train.h5ad",
        val_data="validation.h5ad",
        rounds=1,
        model="test-model",
        prompt="Improve the model.",
    )

    runtime_dir = logs_dir / ".codex_sdk_runtime"
    assert (runtime_dir / "codex_agent_sdk.mjs").is_file()
    package = json.loads((runtime_dir / "package.json").read_text(encoding="utf-8"))
    assert package["dependencies"] == {"@openai/codex-sdk": "0.116.0"}
    assert calls[0][0][0] == "npm"
    assert calls[0][1]["cwd"] == runtime_dir
    assert calls[1][0][0] == "node"
    assert calls[1][1]["cwd"] == runtime_dir
    assert "# generated code" in code_file.read_text(encoding="utf-8")
