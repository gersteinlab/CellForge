import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_config_uses_public_dataset_path():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    assert config["dataset_path"].startswith("data/datasets")


def test_environment_template_has_no_filled_tokens():
    env_text = (ROOT / ".env.example").read_text(encoding="utf-8")
    forbidden = ["ghp_", "sk-", "1947d3", "db84c"]
    assert not any(token in env_text for token in forbidden)


def test_expected_entrypoints_exist():
    for relative_path in [
        "main.py",
        "cellforge/__init__.py",
        "cellforge/Task_Analysis/main.py",
        "cellforge/Method_Design/main.py",
        "cellforge/Code_Generation/__init__.py",
        "cellforge/Code_Generation/base.py",
        "cellforge/Code_Generation/codex_backend.py",
        "cellforge/Code_Generation/legacy/openhands_backend.py",
        "cellforge/legacy/rag_v1/rag.py",
    ]:
        assert (ROOT / relative_path).exists()


def test_duplicate_entrypoints_are_removed():
    assert not (ROOT / "start.py").exists()
    assert not (ROOT / "install.py").exists()
    assert "cellforge=main:cli_entrypoint" in (ROOT / "setup.py").read_text(
        encoding="utf-8"
    )
    assert 'python_requires=">=3.9,<3.13"' in (ROOT / "setup.py").read_text(
        encoding="utf-8"
    )


def test_cli_help_is_clean_without_env_file(tmp_path):
    process = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode == 0
    assert "usage:" in process.stdout
    assert ".env file not found" not in process.stdout
    assert "--codex-optimize-rounds" not in process.stdout
    assert process.stderr == ""


def test_legacy_rag_is_isolated_from_active_task_analysis():
    active = ROOT / "cellforge" / "Task_Analysis"
    for filename in ("rag.py", "search.py", "indexer.py", "dataparser.py", "utils.py"):
        assert not (active / filename).exists()
        assert (ROOT / "cellforge" / "legacy" / "rag_v1" / filename).exists()


def test_workspace_paths_follow_environment(tmp_path, monkeypatch):
    from cellforge.paths import config_path, data_path, workspace_root

    monkeypatch.setenv("CELLFORGE_WORKSPACE_DIR", str(tmp_path))

    assert workspace_root() == tmp_path.resolve()
    assert config_path() == tmp_path.resolve() / "config.json"
    assert data_path("plans", "adamson") == tmp_path.resolve() / "data" / "plans" / "adamson"


def test_doctor_failure_has_nonzero_exit_code(tmp_path):
    environment = os.environ.copy()
    environment["CELLFORGE_WORKSPACE_DIR"] = str(tmp_path)
    process = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--doctor"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode == 1


def test_init_creates_only_canonical_workspace_directories(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "--workspace",
            str(tmp_path),
            "--init",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode == 0
    for name in ("datasets", "analyses", "plans", "codes", "discussion"):
        assert (tmp_path / "data" / name).is_dir()
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "data" / "results").exists()


def test_collect_score_paths_accepts_numeric_scores(tmp_path):
    state_path = tmp_path / "dataset" / "worker" / "run_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "dataset_name": "adamson",
                "history": [{"iter": 1, "metrics": {"val_pearson": 0.5}}],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "reports" / "scores.csv"

    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "collect_score_paths.py"),
            "--root",
            str(tmp_path),
            "--out",
            str(output_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode == 0
    assert "0.5" in output_path.read_text(encoding="utf-8")


def test_codegen_backends_are_registered():
    import pytest

    from cellforge.Code_Generation import create_backend, list_backends
    from cellforge.Code_Generation.codex_backend import CodexCodeGenerator

    assert list_backends() == ["codex"]
    assert isinstance(create_backend("codex"), CodexCodeGenerator)
    with pytest.raises(ValueError, match="Unknown code generation backend"):
        create_backend("claudecode")


def test_coding_agents_do_not_inherit_general_pipeline_model(monkeypatch):
    from cellforge.Code_Generation.codex_backend import CodexCodeGenerator

    monkeypatch.setenv("MODEL_NAME", "general-pipeline-model")
    monkeypatch.delenv("CODEX_MODEL", raising=False)

    assert CodexCodeGenerator().model is None


def test_codex_command_uses_workspace_sandbox_and_non_git_workspace(tmp_path):
    from cellforge.Code_Generation.codex_backend import CodexCodeGenerator
    from cellforge.Code_Generation.contracts import AgentRunRequest

    backend = CodexCodeGenerator()
    request = AgentRunRequest(research_plan={}, workspace=tmp_path)
    command, _environment = backend._command(request, tmp_path / "final.txt")

    assert command[0:2] == [backend.codex_bin, "exec"]
    assert command[command.index("--sandbox") + 1] == "workspace-write"
    assert "--json" in command
    assert "--skip-git-repo-check" in command
    assert "--dangerously-bypass-approvals-and-sandbox" not in command


def test_codex_defaults_to_local_login_and_ignores_provider_keys(monkeypatch):
    from cellforge.Code_Generation.codex_backend import CodexCodeGenerator

    monkeypatch.delenv("CODEX_AUTH_MODE", raising=False)
    monkeypatch.setenv("CODEX_API_KEY", "codex-fixture")
    monkeypatch.setenv("OPENAI_API_KEY", "pipeline-fixture")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://router.example/v1")

    backend = CodexCodeGenerator()
    environment = backend._environment()

    assert backend.auth_mode == "local"
    assert "OPENAI_API_KEY" not in environment
    assert "OPENAI_BASE_URL" not in environment


def test_codex_api_fallback_requires_explicit_auth_mode(monkeypatch):
    from cellforge.Code_Generation.codex_backend import CodexCodeGenerator

    monkeypatch.setenv("CODEX_AUTH_MODE", "api")
    monkeypatch.setenv("CODEX_API_KEY", "codex-fixture")
    monkeypatch.setenv("OPENAI_API_KEY", "pipeline-fixture")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.example/v1")

    environment = CodexCodeGenerator()._environment()

    assert environment["OPENAI_API_KEY"] == "codex-fixture"
    assert environment["OPENAI_BASE_URL"] == "https://api.openai.example/v1"


def test_legacy_openhands_fails_during_backend_selection():
    import pytest

    from cellforge.Code_Generation import create_backend
    from cellforge.Code_Generation.legacy import LegacyBackendUnavailableError
    from cellforge.Code_Generation.legacy.openhands_backend import OpenHandsLegacyCodeGenerator

    for name in ("openhands", "openhands-legacy", "legacy-openhands"):
        with pytest.raises(LegacyBackendUnavailableError, match="no longer supported"):
            create_backend(name)
    with pytest.raises(LegacyBackendUnavailableError, match="Use 'codex'"):
        OpenHandsLegacyCodeGenerator()
