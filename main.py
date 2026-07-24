#!/usr/bin/env python3
"""
cellforge Main Entry Point
End-to-End Intelligent Multi-Agent System for Automated Single-Cell Data Analysis and Method Design
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import Dict, Any
from cellforge.paths import config_path as resolve_config_path
from cellforge.paths import data_path, resolve_workspace_path, workspace_root
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Load environment variables from the runtime workspace without producing
# output before argparse handles --help or reports an invalid argument.
env_file = workspace_root() / ".env"
try:
    if env_file.exists() and load_dotenv is not None:
        load_dotenv(env_file)
except Exception:
    pass

# Default task description - EDIT THIS VARIABLE TO CUSTOMIZE YOUR TASK
DEFAULT_TASK_DESCRIPTION = """Your task is to develop a predictive model that accurately estimates gene expression profiles of individual K562 cells following CRISPR interference (CRISPRi), using the dataset from Norman et al. (2019, Science).

Task Definition:
- Input: Baseline gene expression profile of an unperturbed K562 cell and the identity of the target gene(s) for perturbation
- Output: Predicted gene expression profile after perturbation

Evaluation Scenarios:
1. Unseen Perturbations: Predict effects of gene perturbations not present during training
2. Unseen Cell Contexts: Predict responses in cells with gene expression profiles not observed during training

Evaluation Metrics:
- Mean Squared Error (MSE): Measures the average squared difference between predicted and observed gene expression.
- Pearson Correlation Coefficient (PCC): Quantifies linear correlation between predicted and observed profiles.
- R² (Coefficient of Determination): Represents the proportion of variance in the observed gene expression that can be explained by the predicted values.
- MSE for Differentially Expressed (DE) Genes (MSE_DE): Same as MSE but computed specifically for genes identified as differentially expressed.
- PCC for Differentially Expressed (DE) Genes (PCC_DE): Same as PCC but computed specifically for genes identified as differentially expressed.
- R² for Differentially Expressed (DE) Genes (R2_DE): Same as R² but computed specifically for genes identified as differentially expressed."""


def _dataset_slug(dataset_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (dataset_name or "").strip()).strip("_").lower()
    return slug or "unknown_dataset"


def _dataset_name_from_task(task_analysis: Dict[str, Any]) -> str:
    dataset = task_analysis.get("dataset", {}) if isinstance(task_analysis, dict) else {}
    if isinstance(dataset, dict):
        return dataset.get("name", "unknown_dataset")
    return "unknown_dataset"


def _phase_dataset_dir(phase_root: str, dataset_name: str) -> Path:
    return data_path(phase_root, _dataset_slug(dataset_name))


def _dataset_name_from_dataset_path(dataset_path: str) -> str:
    p = Path(dataset_path)
    # If path points to a file, use file stem; if directory, use directory name.
    if p.suffix:
        return p.stem or "unknown_dataset"
    return p.name or "unknown_dataset"

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration file"""
    target = resolve_config_path(config_path)
    if target.exists():
        with target.open('r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        # Default configuration
        default_config = {
            "task_description": DEFAULT_TASK_DESCRIPTION,
            "dataset_path": "data/datasets/",
            "output_dir": "data/",
            "llm_config": {
                "provider": "openai",  # openai, anthropic, local
                "model": os.getenv("MODEL_NAME", "gpt-4"),
                "api_key": "loaded_from_env"  # API keys are loaded from .env file
            },
            "workflow_phases": ["task_analysis", "method_design", "code_generation"],
            "code_generation": {
                "backend": os.getenv("CODEGEN_BACKEND", "codex")
            },
            "qdrant_config": {
                "host": os.getenv("QDRANT_URL", "localhost"),
                "port": int(os.getenv("QDRANT_PORT", "6333"))
            }
        }

        # Save default configuration
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)

        print(f"✅ Default configuration file created: {target}")
        print("⚠️  Please configure your API keys in .env file")
        print("💡 To customize your task, edit the --task or --task-file CLI option")
        return default_config

    # Update task description from the variable if config exists
    config["task_description"] = DEFAULT_TASK_DESCRIPTION
    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration file completeness"""
    required_fields = ["task_description", "dataset_path", "llm_config"]

    for field in required_fields:
        if field not in config:
            print(f"❌ Configuration file missing required field: {field}")
            return False

    # Check if at least one LLM API key is configured in .env file
    llm_api_keys = [
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("DEEPSEEK_API_KEY"),
        os.getenv("LLAMA_API_KEY"),
        os.getenv("QWEN_API_KEY")
    ]

    configured_llm_keys = [key for key in llm_api_keys if key and key != "your_openai_api_key_here"]

    if not configured_llm_keys:
        print("⚠️  No LLM API keys found in .env file")
        print("💡 Please copy .env.example to .env and configure at least one LLM API key")
        return False

    print(f"✅ {len(configured_llm_keys)} LLM API key(s) configured")

    return True

def run_task_analysis(config: Dict[str, Any]) -> bool:
    """Run Task Analysis phase"""
    try:
        print("\n" + "="*60)
        print("PHASE 1: TASK ANALYSIS")
        print("="*60)

        from cellforge.Task_Analysis.main import run_task_analysis
        from cellforge.retrieval import LiteratureRetriever

        dataset_name = _dataset_name_from_dataset_path(config.get("dataset_path", "unknown_dataset"))
        # Prepare dataset info
        dataset_info = {
            "dataset_path": config["dataset_path"],
            "dataset_name": dataset_name,
            "data_type": "scRNA-seq",
            "cell_line": "K562",
            "perturbation_type": "CRISPRi"
        }

        # Phase-1 output contract: ./data/analyses/<dataset>/
        analyses_dir = _phase_dataset_dir("analyses", dataset_info["dataset_name"])
        analyses_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TASK_ANALYSIS_OUTPUT_DIR"] = str(analyses_dir)
        print(f"📂 Phase 1 output: {analyses_dir}")

        # Run task analysis
        retriever = LiteratureRetriever.from_env(
            trace_dir=analyses_dir / "retrieval"
        )
        result = run_task_analysis(
            config["task_description"],
            dataset_info,
            retriever=retriever,
        )

        if result:
            print("✅ Task analysis completed")
            return True
        else:
            print("❌ Task analysis failed")
            return False

    except Exception as e:
        print(f"❌ Error in task analysis: {str(e)}")
        return False

def run_method_design(config: Dict[str, Any]) -> bool:
    """Run Method Design phase"""
    try:
        print("\n" + "="*60)
        print("PHASE 2: METHOD DESIGN")
        print("="*60)

        # Import method design modules
        from cellforge.Method_Design import generate_research_plan
        from cellforge.Method_Design.main import load_task_analysis
        from cellforge.retrieval import LiteratureRetriever

        # Load task analysis results from contract path: ./data/analyses/<dataset>/
        task_analysis_root = data_path("analyses")
        if not task_analysis_root.exists():
            print("❌ Task analysis results not found. Please run task analysis first.")
            return False

        # Find latest task analysis report recursively by dataset folder
        task_reports = list(task_analysis_root.rglob("task_analysis_*.json"))
        if not task_reports:
            print("❌ No task analysis reports found. Please run task analysis first.")
            return False

        latest_report = max(task_reports, key=lambda x: x.stat().st_mtime)

        # Load and normalize task analysis schema for Method Design
        task_analysis = load_task_analysis(file_path=str(latest_report), latest=False)

        # Phase-2 output contract: ./data/plans/<dataset>/
        dataset_name = _dataset_name_from_task(task_analysis)
        output_dir_path = _phase_dataset_dir("plans", dataset_name)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_dir = str(output_dir_path)
        print(f"📂 Dataset-scoped output: {output_dir}")

        print("🔧 Generating research plan...")
        retriever = LiteratureRetriever.from_env(
            trace_dir=output_dir_path / "retrieval"
        )
        plan = generate_research_plan(
            task_analysis=task_analysis,
            rag_retriever=retriever,
            task_type=task_analysis.get("task_type", "gene_knockout"),
            output_dir=output_dir,
            auto_generate_code=False
        )

        if plan:
            print("✅ Method design completed")

            # Show generated files
            if 'generated_files' in plan:
                files_info = plan['generated_files']
                base_filename = files_info['base_filename']
                print(f"📁 Generated files:")
                print(f"  - {output_dir}/{base_filename}.md (Research plan)")
                print(f"  - {output_dir}/{base_filename}.json (Detailed data)")
                print(f"  - {output_dir}/{base_filename}.mmd (Architecture diagram)")
                print(f"  - {output_dir}/{base_filename}_consensus.png (Consensus progress)")

                # Show code generation result
                if 'code_generation' in plan:
                    code_info = plan['code_generation']
                    if code_info['status'] == 'success':
                        print(f"  - {output_dir}/result.py (Generated code)")
                        print(f"✅ Code generation completed successfully")
                    elif code_info['status'] == 'failed':
                        print(f"❌ Code generation failed: {code_info.get('error', 'Unknown error')}")
                    elif code_info['status'] == 'error':
                        print(f"❌ Code generation error: {code_info.get('error', 'Unknown error')}")

            return True
        else:
            print("❌ Method design failed")
            return False

    except Exception as e:
        print(f"❌ Error in method design: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_code_generation(config: Dict[str, Any]) -> bool:
    """Run Code Generation phase"""
    try:
        print("\n" + "="*60)
        print("PHASE 3: CODE GENERATION")
        print("="*60)

        # Phase-3 reads plan from ./data/plans/<dataset>/ and writes code to ./data/codes/<dataset>/
        plans_root = data_path("plans")
        codes_root = data_path("codes")
        if not plans_root.exists():
            print("❌ Plans directory not found. Please run method design first.")
            return False

        dataset_dir_hint = os.getenv("CODEGEN_DATASET_DIR", "").strip()
        dataset_hint = os.getenv("CODEGEN_DATASET", "").strip()
        if dataset_dir_hint:
            plans_dir = resolve_workspace_path(dataset_dir_hint)
            print(f"📂 Using dataset folder from CODEGEN_DATASET_DIR: {plans_dir}")
        elif dataset_hint:
            plans_dir = plans_root / _dataset_slug(dataset_hint)
            print(f"📂 Using dataset folder from CODEGEN_DATASET: {plans_dir}")
        else:
            # Pick folder that contains the latest research plan recursively.
            all_plan_files = list(plans_root.rglob("research_plan_*.json"))
            if not all_plan_files:
                print("❌ No research plans found. Please run method design first.")
                return False
            latest_any_plan = max(all_plan_files, key=lambda x: x.stat().st_mtime)
            plans_dir = latest_any_plan.parent
            print(f"📂 Auto-selected plan folder: {plans_dir}")

        if not plans_dir.exists():
            print(f"❌ Dataset plan folder not found: {plans_dir}")
            return False

        code_output_dir = codes_root / plans_dir.name
        code_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Phase 3 output: {code_output_dir}")

        codegen_config = config.get("code_generation", {}) if isinstance(config.get("code_generation", {}), dict) else {}
        codegen_backend = os.getenv("CODEGEN_BACKEND", "").strip() or codegen_config.get("backend", "codex")
        task_id = os.getenv("CODEGEN_TASK_ID", "").strip()
        code_filename = f"result_{task_id}.py" if task_id else "result.py"
        code_file = code_output_dir / code_filename
        if code_file.exists():
            from cellforge.Code_Generation.verifier import verify_generated_code

            existing_verification = verify_generated_code(code_file)
            if existing_verification.passed:
                print("✅ Existing generated code passed verification")
                print(f"📁 Generated code: {code_file}")
                return True
            print("⚠️  Existing generated code failed verification; regenerating")

        # Check for research plan files in the selected dataset folder.
        plan_files = list(plans_dir.glob("research_plan_*.json"))
        if not plan_files:
            print(f"❌ No research plans found in {plans_dir}. Please run method design first.")
            return False

        latest_plan = max(plan_files, key=lambda x: x.stat().st_mtime)
        print(f"📋 Found research plan: {latest_plan}")

        # Import code generation module
        try:
            from cellforge.Code_Generation import generate_code_from_plan, generate_code_from_plan_task
        except ImportError as e:
            print(f"❌ Code generation module not available: {e}")
            print("💡 Code generation requires the supported `codex` backend")
            return False

        # Generate code from plan
        print(f"🔧 Generating code from research plan with backend: {codegen_backend}")
        research_plan_obj = json.load(open(latest_plan, 'r', encoding='utf-8'))
        if task_id:
            print(f"🎯 Task-wise code generation enabled: CODEGEN_TASK_ID={task_id}")
            code_file_path = generate_code_from_plan_task(
                research_plan=research_plan_obj,
                task_id=task_id,
                output_dir=str(code_output_dir),
                backend=codegen_backend,
            )
        else:
            code_file_path = generate_code_from_plan(
                research_plan=research_plan_obj,
                output_dir=str(code_output_dir),
                backend=codegen_backend,
            )

        if code_file_path and Path(code_file_path).exists():
            print("✅ Code generation completed")
            print(f"📁 Generated code: {code_file_path}")
            return True
        else:
            print("❌ Code generation failed")
            return False

    except Exception as e:
        print(f"❌ Error in code generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_autorun_phase(config: Dict[str, Any], args: argparse.Namespace) -> bool:
    """Run task-wise split + execute (local/slurm) phase."""
    try:
        print("\n" + "=" * 60)
        print("PHASE 4: AUTORUN")
        print("=" * 60)

        from cellforge.autorun import run_autorun

        summary = run_autorun(
            dataset_path=config["dataset_path"],
            plans_dir=args.autorun_plan_dir,
            workers=args.workers,
            max_tasks=args.max_tasks,
            executor=args.executor,
            partition=args.partition,
            time_limit=args.slurm_time,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            gres=args.gres,
            conda_env=args.conda_env,
            split_ood_ratio=args.split_ood_ratio,
            split_val_ratio=args.split_val_ratio,
            split_seed=args.split_seed,
            codex_optimize_rounds=args.codex_optimize_rounds,
            codex_model=args.codex_model,
            codex_prompt=(
                resolve_workspace_path(args.codex_prompt_file).read_text(encoding="utf-8")
                if args.codex_prompt_file
                else ""
            ),
            codegen_backend=args.codegen_backend,
        )
        print("✅ Autorun finished")
        print(f"📂 Run root: {summary['run_root']}")
        submitted = [j for j in summary["jobs"] if j["status"] == "submitted"]
        failed = [j for j in summary["jobs"] if j["status"] in {"submit_failed", "failed", "codegen_failed"}]
        print(f"📊 Jobs: total={len(summary['jobs'])}, submitted={len(submitted)}, failed={len(failed)}")
        return len(failed) == 0
    except Exception as e:
        print(f"❌ Error in autorun: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_complete_workflow(config: Dict[str, Any]) -> bool:
    """Run complete end-to-end workflow"""
    print("🚀 Starting cellforge End-to-End Workflow")
    print("="*80)

    # Validate configuration
    if not validate_config(config):
        print("❌ Configuration validation failed, please check .env file")
        return False

    success = True

    # Run each phase
    for phase in config["workflow_phases"]:
        if phase == "task_analysis":
            success &= run_task_analysis(config)
        elif phase == "method_design":
            success &= run_method_design(config)
        elif phase == "code_generation":
            success &= run_code_generation(config)

    if success:
        print("\n" + "="*80)
        print("🎉 All phases completed!")
        print("="*80)
        print(f"Results saved under: {data_path()}")
    else:
        print("\n" + "="*80)
        print("❌ Workflow execution failed")
        print("="*80)

    return success

def create_sample_dataset():
    """Create sample dataset directory structure"""
    print("📁 Creating sample dataset directory structure...")

    directories = [
        data_path("datasets"),
        data_path("analyses"),
        data_path("plans"),
        data_path("codes"),
        data_path("discussion"),
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")

    # Create sample README
    readme_content = """# Dataset Directory

Please place your single-cell datasets in the appropriate directories:

- `scRNA-seq/`: Single-cell RNA-seq data (.h5ad files)
- `scATAC-seq/`: Single-cell ATAC-seq data (.h5ad files)
- `perturbation/`: Drug perturbation data (.h5ad files)

## Data Format Requirements

Recommended AnnData format (.h5ad):
- Gene expression matrix stored in `adata.X`
- Cell metadata stored in `adata.obs`
- Gene metadata stored in `adata.var`
- Required annotations: cell type, condition, batch (if applicable)

## Example Datasets

You can download datasets from [scPerturb](https://projects.sanderlab.org/scperturb/):
- Norman et al. (2019) K562 CRISPRi data
- Adamson et al. (2016) Drug perturbation data
"""

    with data_path("datasets", "README.md").open('w', encoding='utf-8') as f:
        f.write(readme_content)

    print("✅ Sample dataset directory structure created")


def run_doctor(config_name: str = "config.json") -> bool:
    """Validate a workspace without mutating it."""
    print("🩺 CellForge workspace doctor")
    print(f"Workspace: {workspace_root()}")

    checks = []
    target_config = resolve_config_path(config_name)
    checks.append(("configuration", target_config.exists(), str(target_config)))
    checks.append(("datasets directory", data_path("datasets").is_dir(), str(data_path("datasets"))))

    literature_root = Path(
        os.getenv("CELLFORGE_LITERATURE_DIR", str(data_path("literature")))
    ).expanduser()
    checks.append(("literature directory", literature_root.is_dir(), str(literature_root)))

    llm_keys = (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "LLAMA_API_KEY",
        "QWEN_API_KEY",
    )
    llm_configured = any(os.getenv(name) for name in llm_keys) or bool(
        os.getenv("CUSTOM_API_KEY") and os.getenv("CUSTOM_API_URL")
    )
    checks.append(("LLM provider", llm_configured, "environment variables"))

    required_ok = True
    for name, ok, detail in checks:
        marker = "✅" if ok else "⚠️"
        print(f"{marker} {name}: {detail}")
        if name in {"configuration", "datasets directory"} and not ok:
            required_ok = False

    if sys.version_info < (3, 9):
        print(f"❌ Python {sys.version.split()[0]} is unsupported; use Python 3.9+")
        required_ok = False
    else:
        print(f"✅ Python: {sys.version.split()[0]}")

    return required_ok


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="cellforge - Intelligent Single-Cell Analysis System")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument(
        "--workspace",
        help="Runtime workspace root (or set CELLFORGE_WORKSPACE_DIR)",
    )
    parser.add_argument("--init", action="store_true", help="Initialize project structure")
    parser.add_argument("--doctor", action="store_true", help="Validate workspace configuration")
    parser.add_argument("--phase", choices=["task_analysis", "method_design", "code_generation", "autorun"],
                       help="Run specific phase")
    parser.add_argument("--dataset-path", help="Dataset path, e.g. ./data/datasets/<dataset> or .h5ad file")
    parser.add_argument("--task", help="Task description text (overrides config/default)")
    parser.add_argument("--task-file", help="Path to a text file containing task description")
    parser.add_argument("--workers", type=int, default=4, help="Autorun workers for task-wise split")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of task-wise jobs")
    parser.add_argument("--executor", choices=["local", "slurm"], default="slurm", help="Autorun execution backend")
    parser.add_argument("--partition", default="scavenge_gpu", help="Slurm partition for autorun")
    parser.add_argument("--slurm-time", default="01:00:00", help="Slurm time limit for each job")
    parser.add_argument("--cpus-per-task", type=int, default=4, help="Slurm cpus per task")
    parser.add_argument("--mem", default="32G", help="Slurm memory")
    parser.add_argument("--gres", default="gpu:1", help="Slurm gres, e.g. gpu:1")
    parser.add_argument("--conda-env", default="cellforge", help="Conda env activated in slurm job")
    parser.add_argument("--autorun-plan-dir", default=None, help="Optional plan folder override for autorun")
    parser.add_argument("--split-ood-ratio", type=float, default=0.2, help="Holdout ratio for OOD perturbations")
    parser.add_argument("--split-val-ratio", type=float, default=0.1, help="Validation cell ratio within in-distribution data")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for perturbation split")
    parser.add_argument(
        "--codex-optimize-rounds",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--codex-model", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--codex-prompt-file",
        default="cellforge/autorun/autoresearch_prompt.md",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--codegen-backend",
        default=os.getenv("CODEGEN_BACKEND", "codex"),
        choices=["codex"],
        help="Code-generation backend (currently Codex only)",
    )

    args = parser.parse_args()

    if args.workspace:
        os.environ["CELLFORGE_WORKSPACE_DIR"] = str(Path(args.workspace).expanduser().resolve())
        workspace_env = workspace_root() / ".env"
        if workspace_env.exists() and load_dotenv is not None:
            load_dotenv(workspace_env, override=False)

    if args.doctor:
        return run_doctor(args.config)

    if args.init:
        print("🚀 Initializing cellforge project...")
        create_sample_dataset()
        load_config(args.config)  # Create default configuration
        print("\n✅ Project initialization completed!")
        print("📝 Please copy .env.example to .env and configure your API keys")
        print("💡 To customize your task, edit the --task or --task-file CLI option")
        return True

    # Load configuration
    config = load_config(args.config)

    # User-friendly CLI overrides
    if args.dataset_path:
        config["dataset_path"] = args.dataset_path
    config.setdefault("code_generation", {})
    config["code_generation"]["backend"] = args.codegen_backend
    if args.task_file:
        task_file = resolve_workspace_path(args.task_file)
        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")
        config["task_description"] = task_file.read_text(encoding="utf-8").strip()
    elif args.task:
        config["task_description"] = args.task.strip()

    print(f"🧪 Dataset path: {config.get('dataset_path')}")
    print(f"📝 Task chars: {len(config.get('task_description', ''))}")

    if args.phase:
        # Run specific phase
        if args.phase == "task_analysis":
            return run_task_analysis(config)
        elif args.phase == "method_design":
            return run_method_design(config)
        elif args.phase == "code_generation":
            return run_code_generation(config)
        elif args.phase == "autorun":
            return run_autorun_phase(config, args)
    else:
        # Run complete workflow
        return run_complete_workflow(config)


def cli_entrypoint() -> int:
    """Console-script adapter with conventional process exit codes."""
    return 0 if main() else 1


if __name__ == "__main__":
    raise SystemExit(cli_entrypoint())
