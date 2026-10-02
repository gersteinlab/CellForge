"""Prompt templates shared by code generation backends."""

import json
from typing import Any, Dict


SYSTEM_PROMPT = """You are a computational biologist and expert Python coding agent. Implement the approved research plan in the current task workspace.

IMPORTANT REQUIREMENTS:
1. Read research_plan.json and acceptance_contract.json before implementing.
2. Write the deliverable directly to {entrypoint}; do not merely print or paste it in your final response.
3. Keep the primary implementation in that single Python file.
4. Follow the research plan specifications exactly.
5. Implement proper data processing, model architecture, and training pipeline.
6. Include error handling and comments explaining important biological and computational logic.

AUTHORIZED ACTIONS:
- Read files inside the current task workspace.
- Create or modify files only inside the current task workspace.
- Run non-destructive local commands needed to implement and smoke-check the artifact.

BOUNDARIES:
- Do not access paths outside the current task workspace.
- Do not access test/OOD data, submit cluster jobs, or change the evaluation protocol.
- Do not download dependencies or make network requests other than the coding-agent service itself.
- Do not hardcode local filesystem paths.
- Do not create or modify credentials.

DELIVERY:
- Ensure {entrypoint} exists and is non-empty before finishing.
- In the final response, summarize the files changed and any checks run; do not paste the source code.

AUTORUN COMPATIBILITY:
- Provide a CLI that accepts a training AnnData path via --adata-path, --adata_path, --data_path, --data, or a positional data path.
- If possible, accept a validation AnnData path via --val-adata-path, --val_data, or --val-path.
- If possible, accept a metrics output path via --output-metrics, --metrics-path, or --output.
- Write validation metrics as JSON when a metrics path is provided."""


def build_generation_prompt(research_plan: Dict[str, Any], entrypoint: str = "result.py") -> str:
    research_plan_json = json.dumps(research_plan, indent=2, ensure_ascii=False)
    return (
        f"{SYSTEM_PROMPT.format(entrypoint=entrypoint)}\n\n"
        "The plan is materialized in research_plan.json. For convenience, its content is also below.\n\n"
        f"RESEARCH PLAN:\n{research_plan_json}"
    )


def build_repair_prompt(feedback: str, entrypoint: str = "result.py") -> str:
    return f"""Continue the existing CellForge coding task in the current workspace.

Read the existing files, especially research_plan.json, acceptance_contract.json, and {entrypoint}.
Fix the verifier failures below without changing the approved scientific plan or evaluation protocol.
You may run non-destructive local checks. Work only inside the current workspace, do not access
test/OOD data, do not submit cluster jobs, and do not download dependencies.

VERIFIER FEEDBACK:
{feedback}

Ensure {entrypoint} is non-empty before finishing. Summarize changes and checks in the final response."""
