import json
import sys

import pytest

from cellforge.Code_Generation.verifier import (
    CodeGenerationVerifier,
    VerificationCommand,
    VerificationContract,
    verify_generated_code,
)


def test_verifier_accepts_valid_entrypoint_and_result(tmp_path):
    (tmp_path / "result.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.parse_args()\n",
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(
        json.dumps({"status": "ok", "mse": 0.1}),
        encoding="utf-8",
    )

    result = CodeGenerationVerifier().verify(
        tmp_path,
        VerificationContract(
            required_files=("result.py",),
            result_json="metrics.json",
            required_result_fields=("status", "mse"),
        ),
    )

    assert result.passed
    assert result.failures == []
    assert result.to_dict()["checks"]


def test_verifier_returns_structured_repair_feedback(tmp_path):
    (tmp_path / "result.py").write_text("def broken(:\n", encoding="utf-8")

    result = CodeGenerationVerifier().verify(tmp_path)
    feedback = result.repair_feedback(round_number=2, max_rounds=5)

    assert not result.passed
    assert feedback["kind"] == "code_generation_verification"
    assert feedback["round"] == 2
    assert feedback["remaining_rounds"] == 3
    assert any(failure["name"] == "python_compile" for failure in feedback["failures"])
    assert all("message" in failure for failure in feedback["failures"])


def test_verifier_supports_configurable_commands(tmp_path):
    (tmp_path / "result.py").write_text("print('generated')\n", encoding="utf-8")
    command = VerificationCommand(
        name="custom_smoke",
        argv=(sys.executable, "-c", "raise SystemExit(7)"),
    )

    result = CodeGenerationVerifier().verify(
        tmp_path,
        VerificationContract(commands=(command,)),
    )

    failure = next(check for check in result.failures if check.name == "custom_smoke")
    assert failure.exit_code == 7
    assert failure.command == list(command.argv)


def test_convenience_api_accepts_code_file_and_command_argv(tmp_path):
    code_file = tmp_path / "result.py"
    code_file.write_text("print('generated')\n", encoding="utf-8")

    result = verify_generated_code(
        code_file,
        acceptance_commands=((sys.executable, "-c", "print('smoke')"),),
    )

    assert result.passed
    assert result.report == result.to_dict()
    assert '"passed": true' in result.repair_feedback_text(1)


def test_verifier_reports_missing_files_and_json_fields(tmp_path):
    (tmp_path / "result.py").write_text(
        "import argparse\nargparse.ArgumentParser().parse_args()\n",
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text('{"status": "ok"}', encoding="utf-8")

    result = CodeGenerationVerifier().verify(
        tmp_path,
        VerificationContract(
            required_files=("requirements.txt",),
            result_json="metrics.json",
            required_result_fields=("status", "pcc"),
        ),
    )

    assert not result.passed
    names = {failure.name for failure in result.failures}
    assert "file_exists:requirements.txt" in names
    json_failure = next(failure for failure in result.failures if failure.name == "result_json")
    assert json_failure.details["missing_fields"] == ["pcc"]


def test_verifier_rejects_paths_outside_workspace(tmp_path):
    result = CodeGenerationVerifier().verify(
        tmp_path,
        VerificationContract(entrypoint="../result.py"),
    )

    assert not result.passed
    assert result.failures[0].name == "entrypoint_path"


@pytest.mark.parametrize("round_number", [0, 6])
def test_repair_feedback_enforces_five_round_limit(tmp_path, round_number):
    result = CodeGenerationVerifier().verify(tmp_path)

    with pytest.raises(ValueError):
        result.repair_feedback(round_number=round_number, max_rounds=5)
