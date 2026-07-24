"""Task Analysis schema normalization and loading for Method Design."""

import json
import re
from typing import Dict, Any
from pathlib import Path
from ..paths import data_path, resolve_workspace_path

REQUIRED_METHOD_FIELDS = [
    "task_type", "dataset", "perturbations", "cell_types",
    "objectives", "constraints", "evaluation_metrics"
]

def _dataset_slug(dataset_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (dataset_name or "").strip()).strip("_").lower()
    return slug or "unknown_dataset"

def _normalize_task_analysis_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Task Analysis outputs into Method Design expected schema."""
    if not isinstance(data, dict):
        return data

    if all(k in data for k in REQUIRED_METHOD_FIELDS):
        return data

    task_description = data.get("task_description", "")
    dataset_info = data.get("dataset_info", {}) if isinstance(data.get("dataset_info", {}), dict) else {}
    analysis_results = data.get("analysis_results", {}) if isinstance(data.get("analysis_results", {}), dict) else {}
    final_recs = analysis_results.get("final_recommendations", {}) if isinstance(analysis_results.get("final_recommendations", {}), dict) else {}

    text_blob = f"{task_description}\n{json.dumps(dataset_info, ensure_ascii=False)}\n{json.dumps(final_recs, ensure_ascii=False)}".lower()
    task_type = "gene_knockout" if ("crispr" in text_blob or "knockout" in text_blob or "perturb" in text_blob) else "drug_perturbation"

    dataset_name = dataset_info.get("dataset_name") or dataset_info.get("name") or "TaskAnalysisDataset"
    dataset_type = dataset_info.get("data_type") or dataset_info.get("type") or "single_cell_RNA_seq"
    dataset_desc = dataset_info.get("description") or f"Normalized from Task Analysis report ({dataset_name})"

    cell_types = []
    if dataset_info.get("cell_line"):
        cell_types = [dataset_info["cell_line"]]
    elif "k562" in text_blob:
        cell_types = ["K562"]
    else:
        cell_types = ["K562"]

    perturbations = [{
        "type": "gene_knockout" if "crispr" in text_blob else "perturbation",
        "targets": ["CRISPRi_targets"] if "crispr" in text_blob else ["unknown_targets"],
        "description": "Extracted from Task Analysis context"
    }]

    metrics = []
    for m in ["MSE", "PCC", "R2", "MSE_DE", "PCC_DE", "R2_DE", "Pearson correlation", "Mean Squared Error"]:
        if m.lower() in text_blob and m not in metrics:
            metrics.append(m)
    if not metrics:
        metrics = ["MSE", "Pearson correlation"]

    objectives = [
        "Predict gene expression responses after perturbation",
        "Generalize to unseen perturbations and unseen cell contexts"
    ]
    constraints = [
        "Biological interpretability",
        "Generalization under data sparsity",
        "Computational efficiency"
    ]

    normalized = {
        "task_type": task_type,
        "dataset": {
            "name": dataset_name,
            "type": dataset_type,
            "description": dataset_desc
        },
        "perturbations": perturbations,
        "cell_types": cell_types,
        "objectives": objectives,
        "constraints": constraints,
        "evaluation_metrics": metrics,
        "_normalized_from_task_analysis_report": True
    }
    return normalized

def load_task_analysis(file_path: str = None, plan_id: str = None, latest: bool = True) -> Dict[str, Any]:
    """
    Load task analysis from file with enhanced path resolution

    Args:
        file_path: Direct file path (absolute or relative)
        plan_id: Specific plan ID to find
        latest: If True and no plan_id/file_path, return latest plan

    Returns:
        Task analysis data
    """
    try:
        # If no file_path is provided, use the latest workspace analysis report.
        if not file_path:
            # First try to find task analysis from contract directory
            analyses_dir = data_path("analyses")

            if latest:
                # Look for task analysis files in analyses directory
                task_analysis_files = []

                # Look for analysis_report.json (task analysis output)
                analysis_report = analyses_dir / "analysis_report.json"
                if analysis_report.exists():
                    task_analysis_files.append(analysis_report)

                # Look for other potential task analysis files
                for pattern in ["*analysis*.json", "*task*.json", "*report*.json"]:
                    if analyses_dir.exists():
                        task_analysis_files.extend(analyses_dir.rglob(pattern))

                if task_analysis_files:
                    # Get the most recent file
                    latest_file = max(task_analysis_files, key=lambda x: x.stat().st_mtime)
                    file_path = str(latest_file)
                    print(f"Found latest task analysis file: {latest_file.name}")
                else:
                    raise FileNotFoundError(
                        f"No task analysis reports found under {analyses_dir}"
                    )
            else:
                raise FileNotFoundError("No file path or plan ID provided")

        # Resolve path (handle relative paths)
        target_path = Path(file_path)
        if not target_path.is_absolute():
            target_path = resolve_workspace_path(target_path)

        # Load the file
        with open(target_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert task analysis format to expected format
        if isinstance(data, dict):
            # Check if it's an analysis report (task analysis output)
            if 'task_requirements' in data and 'dataset_characteristics' in data:
                # Convert analysis report to task analysis format
                task_analysis = {
                    "task_type": "drug_perturbation",  # Default based on L1000 data
                    "dataset": {
                        "name": "L1000_Connectivity_Map",
                        "type": "bulk_RNA_seq",
                        "description": data.get('dataset_characteristics', {}).get('source_protocol', 'L1000 Connectivity Map')
                    },
                    "perturbations": [
                        {
                            "type": "drug_perturbation",
                            "targets": ["Small molecule compounds"],
                            "description": "Drug perturbation analysis using L1000 data"
                        }
                    ],
                    "cell_types": data.get('dataset_characteristics', {}).get('composition', {}).get('cell_lines', ['A549', 'MCF7', 'PC3']),
                    "objectives": [
                        data.get('task_requirements', {}).get('core_task_definition', 'Predict gene expression responses to perturbations')
                    ],
                    "constraints": [
                        "Limited training data",
                        "Need for biological interpretability",
                        "Computational efficiency requirements"
                    ],
                    "evaluation_metrics": data.get('task_requirements', {}).get('evaluation_criteria', {}).get('primary_metrics', ['MSE', 'Pearson correlation'])
                }
                normalized = _normalize_task_analysis_schema(task_analysis)
                print(f"Successfully converted analysis report to task analysis from: {target_path}")
                return normalized
            elif 'research_plan' in data:
                # Extract task analysis from the enhanced plan structure
                task_analysis = data['research_plan']
                normalized = _normalize_task_analysis_schema(task_analysis)
                print(f"Successfully loaded task analysis from: {target_path}")
                return normalized
            else:
                # If it's a direct task analysis file (old format)
                normalized = _normalize_task_analysis_schema(data)
                print(f"Successfully loaded task analysis from: {target_path}")
                return normalized
        else:
            print(f"Successfully loaded task analysis from: {target_path}")
            return _normalize_task_analysis_schema(data)

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Task analysis file not found: {file_path} - {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file: {file_path} - {e}")
    except Exception as e:
        raise Exception(f"Error loading task analysis: {e}")
