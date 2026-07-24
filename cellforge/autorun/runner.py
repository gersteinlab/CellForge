from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import anndata as ad
import numpy as np
from ..paths import data_path, resolve_workspace_path, workspace_root

@dataclass
class JobRecord:
    task_id: str
    code_file: str
    status: str
    command: str
    logs_dir: str
    val_metrics_file: str = ""
    job_id: str = ""
    error: str = ""


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip()).strip("_").lower() or "unknown_dataset"


def _latest_plan_file(plans_dir: Path) -> Path:
    candidates = list(plans_dir.glob("research_plan_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No plan file found in {plans_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_dataset_file(dataset_path: str) -> Path:
    p = Path(dataset_path)
    if p.is_file():
        return p
    h5ad = sorted(p.glob("*.h5ad"))
    if h5ad:
        return h5ad[0]
    raise FileNotFoundError(f"No .h5ad file found under dataset path: {dataset_path}")


def _task_ids(plan: Dict[str, Any], max_tasks: Optional[int]) -> List[str]:
    ids: List[str] = []
    for item in plan.get("task_wise_plan", []) or []:
        tid = str(item.get("task_id", "")).strip()
        if tid and tid not in ids:
            ids.append(tid)
    if not ids:
        ids = ["T01"]
    if max_tasks and max_tasks > 0:
        ids = ids[:max_tasks]
    return ids


def _resolve_perturbation_key(adata: ad.AnnData) -> str:
    for key in ("perturbation", "perturbation_id", "gene", "target_gene", "condition"):
        if key in adata.obs.columns:
            return key
    raise KeyError(
        "No perturbation column found in adata.obs. Tried: perturbation, perturbation_id, gene, target_gene, condition"
    )


def _prepare_perturbation_split(
    dataset_file: Path,
    split_root: Path,
    ood_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    manifest_path = split_root / "split_manifest.json"
    train_path = split_root / "train.h5ad"
    val_path = split_root / "val.h5ad"
    if manifest_path.exists() and train_path.exists() and val_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    adata = ad.read_h5ad(dataset_file)
    pert_key = _resolve_perturbation_key(adata)
    uniq = sorted({str(x) for x in adata.obs[pert_key].astype(str).tolist()})
    if len(uniq) < 2:
        raise ValueError(f"Need at least 2 unique perturbations to split; got {len(uniq)}")

    import random

    rng = random.Random(seed)
    rng.shuffle(uniq)
    # Step 1: hold out OOD perturbations first.
    n_ood = max(1, int(round(len(uniq) * ood_ratio)))
    if n_ood >= len(uniq):
        n_ood = len(uniq) - 1
    ood_perts = set(uniq[:n_ood])
    in_dist_perts = set(uniq[n_ood:])

    obs_vals = adata.obs[pert_key].astype(str)
    ood_mask = obs_vals.isin(ood_perts).values
    in_dist_mask = obs_vals.isin(in_dist_perts).values

    in_dist_adata = adata[in_dist_mask].copy()
    n_ood_cells = int(np.asarray(ood_mask).sum())

    # Step 2: split in-distribution cells into train/val for progress monitoring.
    n_in = in_dist_adata.n_obs
    if n_in < 2:
        raise ValueError(f"Not enough in-distribution cells to split train/val: {n_in}")
    perm = np.arange(n_in)
    rng.shuffle(perm)
    n_val_cells = max(1, int(round(n_in * val_ratio)))
    if n_val_cells >= n_in:
        n_val_cells = n_in - 1
    val_idx = perm[:n_val_cells]
    train_idx = perm[n_val_cells:]

    train_adata = in_dist_adata[train_idx].copy()
    val_adata = in_dist_adata[val_idx].copy()

    split_root.mkdir(parents=True, exist_ok=True)
    train_adata.write_h5ad(train_path)
    val_adata.write_h5ad(val_path)

    meta = {
        "perturbation_key": pert_key,
        "n_unique_perturbations": len(uniq),
        "n_ood_perturbations": len(ood_perts),
        "n_in_dist_perturbations": len(in_dist_perts),
        "n_train_perturbations": len({str(x) for x in train_adata.obs[pert_key].astype(str).tolist()}),
        "n_val_perturbations": len({str(x) for x in val_adata.obs[pert_key].astype(str).tolist()}),
        "n_train_cells": int(train_adata.n_obs),
        "n_val_cells": int(val_adata.n_obs),
        "n_ood_cells": n_ood_cells,
        "train_h5ad": str(train_path),
        "val_h5ad": str(val_path),
        "ood_ratio": ood_ratio,
        "val_ratio": val_ratio,
        "seed": seed,
        "ood_perturbations": sorted(ood_perts),
    }
    manifest_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _select_data_arg_flag(code_file: Path) -> str:
    flags = _extract_cli_flags(code_file)
    for candidate in ("--adata-path", "--adata_path", "--data_path", "--input_path", "--adata", "--data"):
        if candidate in flags:
            return candidate
    return ""


def _extract_cli_flags(code_file: Path) -> set[str]:
    try:
        content = code_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return set()
    flags = set(re.findall(r"add_argument\(\s*['\"](--[A-Za-z0-9_-]+)", content))
    return flags


def _select_val_arg_flag(code_file: Path) -> str:
    flags = _extract_cli_flags(code_file)
    for flag in (
        "--val-adata-path",
        "--val_adata_path",
        "--val-data",
        "--val_data",
        "--val-path",
        "--val_path",
        "--val-h5ad",
    ):
        if flag in flags:
            return flag
    return ""


def _select_metrics_arg_flag(code_file: Path) -> str:
    flags = _extract_cli_flags(code_file)
    for flag in (
        "--output-metrics",
        "--output_metrics",
        "--metrics-path",
        "--metrics_path",
        "--metrics-out",
        "--metrics_out",
    ):
        if flag in flags:
            return flag
    return ""


def _select_output_arg_flag(code_file: Path) -> str:
    flags = _extract_cli_flags(code_file)
    for flag in ("--output", "--output_dir", "--output-file", "--output_file", "--result-path", "--result_path"):
        if flag in flags:
            return flag
    return ""


def _run_codex_autoresearch(
    *,
    code_file: Path,
    logs_dir: Path,
    train_data: str,
    val_data: str,
    rounds: int,
    model: str,
    prompt: str,
) -> None:
    rounds = max(1, rounds)
    if shutil.which("node") is None or shutil.which("npm") is None:
        return

    project_root = Path(__file__).resolve().parents[2]
    sdk_script = Path(__file__).resolve().parent / "codex_agent_sdk.mjs"
    node_pkg = project_root / "node_modules" / "@openai" / "codex-sdk"
    if not node_pkg.exists():
        subprocess.run(
            ["npm", "install", "--silent"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

    def extract_code(md: str) -> str:
        raw = md.strip()
        if raw.startswith("{"):
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    maybe = payload.get("finalResponse", "")
                    if isinstance(maybe, str) and maybe.strip():
                        raw = maybe.strip()
            except Exception:
                pass
        md = raw
        if "```python" in md:
            s = md.find("```python") + len("```python")
            e = md.find("```", s)
            return md[s:e].strip() if e != -1 else md[s:].strip()
        if "```" in md:
            s = md.find("```") + len("```")
            e = md.find("```", s)
            return md[s:e].strip() if e != -1 else md[s:].strip()
        return md.strip()

    def looks_like_python(code: str) -> bool:
        c = (code or "").strip()
        if not c:
            return False
        if c.startswith("{") and c.endswith("}"):
            return False
        markers = ("import ", "from ", "def ", "class ", "if __name__ == ")
        return any(m in c for m in markers)

    for i in range(1, rounds + 1):
        out_file = logs_dir / f"codex_round_{i}.md"
        current_code = code_file.read_text(encoding="utf-8", errors="ignore")
        full_prompt = f"""
{prompt}
You are editing one Python file: {code_file.name}
Train data path: {train_data}
Validation data path: {val_data}

Important:
- Do NOT run shell commands or tools.
- Do NOT return JSON metadata.
- Return only the full updated Python file (or one python fenced block).
- Keep CLI compatible and robust.

Current file content:
```python
{current_code}
```
"""
        prompt_file = logs_dir / f"codex_round_{i}.prompt.md"
        prompt_file.write_text(full_prompt, encoding="utf-8")
        proc = subprocess.run(
            [
                "node",
                str(sdk_script),
                "--cwd",
                str(code_file.parent),
                "--model",
                model,
                "--prompt-file",
                str(prompt_file),
                "--out-file",
                str(out_file),
            ],
            text=True,
            cwd=project_root,
            capture_output=True,
            timeout=1800,
            check=False,
        )
        (logs_dir / f"codex_round_{i}.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (logs_dir / f"codex_round_{i}.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode != 0 or not out_file.exists():
            continue
        updated = extract_code(out_file.read_text(encoding="utf-8", errors="ignore"))
        if updated and len(updated) > 200 and looks_like_python(updated):
            code_file.write_text(updated.rstrip() + "\n", encoding="utf-8")


def _render_sbatch(
    *,
    job_name: str,
    partition: str,
    time_limit: str,
    cpus: int,
    mem: str,
    gres: str,
    stdout_path: Path,
    stderr_path: Path,
    command: str,
    workdir: Path,
    conda_env: str,
    metrics_file: str = "",
    env: Optional[Dict[str, str]] = None,
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={time_limit}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --output={stdout_path}",
        f"#SBATCH --error={stderr_path}",
    ]
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
    lines.extend(
        [
            "",
            "set -eo pipefail",
            f"cd {shlex.quote(str(workdir))}",
            "set +u",
            "source ~/.bashrc || true",
            "set -u",
            f"conda activate {shlex.quote(conda_env)} || true",
        ]
    )
    if env:
        for key, val in env.items():
            safe_key = re.sub(r"[^A-Za-z0-9_]", "_", key)
            lines.append(f"export {safe_key}={shlex.quote(str(val))}")
    if metrics_file:
        lines.extend(
            [
                f"METRICS_FILE={shlex.quote(metrics_file)}",
                "python - <<'PY'",
                "import json, os, datetime",
                "p = os.environ.get('METRICS_FILE', '')",
                "if p:",
                "    os.makedirs(os.path.dirname(p), exist_ok=True)",
                "    payload = {",
                "        'status': 'started',",
                "        'started_at': datetime.datetime.now().isoformat(),",
                "        'train_h5ad': os.environ.get('CELLFORGE_TRAIN_ADATA_PATH', ''),",
                "        'val_h5ad': os.environ.get('CELLFORGE_VAL_ADATA_PATH', ''),",
                "    }",
                "    with open(p, 'w', encoding='utf-8') as f:",
                "        json.dump(payload, f, ensure_ascii=False, indent=2)",
                "PY",
                "set +e",
                f"{command}",
                "CMD_RC=$?",
                "export CMD_RC",
                "set -e",
                "python - <<'PY'",
                "import json, os, datetime",
                "p = os.environ.get('METRICS_FILE', '')",
                "if p:",
                "    payload = {}",
                "    if os.path.exists(p):",
                "        try:",
                "            with open(p, 'r', encoding='utf-8') as f:",
                "                payload = json.load(f)",
                "        except Exception:",
                "            payload = {}",
                "    payload.update({",
                "        'finished_at': datetime.datetime.now().isoformat(),",
                "        'exit_code': int(os.environ.get('CMD_RC', '1')),",
                "        'status': 'completed' if int(os.environ.get('CMD_RC', '1')) == 0 else 'failed'",
                "    })",
                "    with open(p, 'w', encoding='utf-8') as f:",
                "        json.dump(payload, f, ensure_ascii=False, indent=2)",
                "PY",
                "exit $CMD_RC",
            ]
        )
    else:
        lines.append(command)
    lines.append("")
    return "\n".join(lines)


def _submit_sbatch(script_path: Path, cwd: Path) -> tuple[bool, str, str]:
    proc = subprocess.run(
        ["sbatch", str(script_path)],
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return False, "", out.strip()
    m = re.search(r"Submitted batch job (\d+)", out)
    return True, (m.group(1) if m else ""), out.strip()


def run_autorun(
    *,
    dataset_path: str,
    plans_dir: Optional[str] = None,
    workers: int = 4,
    max_tasks: Optional[int] = None,
    executor: str = "slurm",
    partition: str = "scavenge_gpu",
    time_limit: str = "01:00:00",
    cpus_per_task: int = 4,
    mem: str = "32G",
    gres: str = "gpu:1",
    conda_env: str = "cellforge",
    split_ood_ratio: float = 0.2,
    split_val_ratio: float = 0.1,
    split_seed: int = 42,
    codex_optimize_rounds: int = 1,
    codex_model: str = "gpt-5-codex",
    codex_prompt: str = "",
    codegen_backend: str = "codex",
) -> Dict[str, Any]:
    project_root = workspace_root()
    dataset_file = _find_dataset_file(str(resolve_workspace_path(dataset_path)))
    dataset_name = _slug(dataset_file.parent.name if dataset_file.parent.name else dataset_file.stem)

    plan_root = (
        resolve_workspace_path(plans_dir)
        if plans_dir
        else data_path("plans", dataset_name)
    )
    plan_file = _latest_plan_file(plan_root)
    plan_obj = json.loads(plan_file.read_text(encoding="utf-8"))
    task_ids = _task_ids(plan_obj, max_tasks=max_tasks)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = data_path("runs", dataset_name, timestamp)
    run_root.mkdir(parents=True, exist_ok=True)
    split_tag = f"ood{int(round(split_ood_ratio * 100))}_val{int(round(split_val_ratio * 100))}_seed{split_seed}"
    split_root = dataset_file.parent / "splits" / split_tag
    split_meta = _prepare_perturbation_split(
        dataset_file=dataset_file,
        split_root=split_root,
        ood_ratio=split_ood_ratio,
        val_ratio=split_val_ratio,
        seed=split_seed,
    )

    from cellforge.Code_Generation import generate_code_from_plan_task

    jobs: List[JobRecord] = []
    for idx, task_id in enumerate(task_ids):
        worker_id = idx % max(workers, 1)
        task_dir = run_root / f"w{worker_id}" / task_id
        code_dir = task_dir / "code"
        logs_dir = task_dir / "logs"
        code_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        code_file = generate_code_from_plan_task(
            plan_obj,
            task_id=task_id,
            output_dir=str(code_dir),
            backend=codegen_backend,
        )
        if not code_file:
            jobs.append(
                JobRecord(
                    task_id=task_id,
                    code_file="",
                    status="codegen_failed",
                    command="",
                    logs_dir=str(logs_dir),
                    error="Failed to generate code for this task",
                )
            )
            continue

        if codex_optimize_rounds > 0:
            if codegen_backend.strip().lower() != "codex":
                raise ValueError(
                    "--codex-optimize-rounds cannot rewrite output from a "
                    f"different backend ({codegen_backend!r}). Use that backend's "
                    "repair loop instead."
                )
            _run_codex_autoresearch(
                code_file=Path(code_file),
                logs_dir=logs_dir,
                train_data=str(split_meta["train_h5ad"]),
                val_data=str(split_meta["val_h5ad"]),
                rounds=codex_optimize_rounds,
                model=codex_model,
                prompt=codex_prompt or "Improve the training script for better validation metric with robust runtime behavior.",
            )

        data_flag = _select_data_arg_flag(Path(code_file))
        val_flag = _select_val_arg_flag(Path(code_file))
        metrics_flag = _select_metrics_arg_flag(Path(code_file))
        output_flag = _select_output_arg_flag(Path(code_file))
        metrics_path = logs_dir / "val_metrics.json"
        cmd_parts = ["python", shlex.quote(code_file)]
        if data_flag:
            cmd_parts.extend([data_flag, shlex.quote(split_meta["train_h5ad"])])
        else:
            # Positional-args style CLI: python result.py <data_path> [perturbation_key]
            cmd_parts.append(shlex.quote(split_meta["train_h5ad"]))
        if val_flag:
            cmd_parts.extend([val_flag, shlex.quote(split_meta["val_h5ad"])])
        if metrics_flag:
            cmd_parts.extend([metrics_flag, shlex.quote(str(metrics_path))])
        elif output_flag:
            cmd_parts.extend([output_flag, shlex.quote(str(metrics_path))])
        cmd = " ".join(cmd_parts)
        job_env = {
            "CELLFORGE_WORKSPACE_DIR": str(project_root),
            "CELLFORGE_TRAIN_ADATA_PATH": str(split_meta["train_h5ad"]),
            "CELLFORGE_VAL_ADATA_PATH": str(split_meta["val_h5ad"]),
            "CELLFORGE_VAL_METRICS_PATH": str(metrics_path),
            "CELLFORGE_RUN_ROOT": str(run_root),
        }
        if executor == "local":
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=project_root,
                text=True,
                capture_output=True,
                env={**os.environ, **job_env},
            )
            (logs_dir / "local.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
            (logs_dir / "local.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
            jobs.append(
                JobRecord(
                    task_id=task_id,
                    code_file=str(code_file),
                    status="completed" if proc.returncode == 0 else "failed",
                    command=cmd,
                    logs_dir=str(logs_dir),
                    val_metrics_file=str(metrics_path),
                    error="" if proc.returncode == 0 else f"exit code {proc.returncode}",
                )
            )
            continue

        script_path = logs_dir / "job.sbatch"
        stdout_path = logs_dir / "slurm-%j.out"
        stderr_path = logs_dir / "slurm-%j.err"
        script_path.write_text(
            _render_sbatch(
                job_name=f"cf_{dataset_name}_{task_id}",
                partition=partition,
                time_limit=time_limit,
                cpus=cpus_per_task,
                mem=mem,
                gres=gres,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                command=cmd,
                workdir=project_root,
                conda_env=conda_env,
                metrics_file=str(metrics_path),
                env=job_env,
            ),
            encoding="utf-8",
        )
        ok, job_id, submit_output = _submit_sbatch(script_path, cwd=project_root)
        jobs.append(
            JobRecord(
                task_id=task_id,
                code_file=str(code_file),
                status="submitted" if ok else "submit_failed",
                command=cmd,
                logs_dir=str(logs_dir),
                val_metrics_file=str(metrics_path),
                job_id=job_id,
                error="" if ok else submit_output,
            )
        )

    summary = {
        "dataset_file": str(dataset_file),
        "dataset_name": dataset_name,
        "plan_file": str(plan_file),
        "run_root": str(run_root),
        "split_root": str(split_root),
        "executor": executor,
        "partition": partition if executor == "slurm" else "",
        "split": split_meta,
        "jobs": [asdict(j) for j in jobs],
    }
    (run_root / "autorun_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
