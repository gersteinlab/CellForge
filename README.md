# CellForge

CellForge is a multi-agent framework for open-ended design of computational methods for single-cell omics. It decomposes a biological modeling task, searches and organizes relevant prior knowledge, coordinates expert-style method design, and can generate executable analysis code from the resulting research plan.

Its main capabilities are:

- literature-grounded task analysis using local papers, PubMed, Crossref, and Semantic Scholar;
- collaborative method design with domain-specialized agents;
- research-plan generation with traceable supporting evidence;
- verified code generation through Codex;
- optional local or SLURM execution for task-wise experiments.

## Workflow

CellForge is organized around four stages:

1. `task_analysis`: parse the user task and dataset context, then write structured reports to `data/analyses/<dataset>/`.
2. `method_design`: run multi-agent method discussion and write research plans to `data/plans/<dataset>/`.
3. `code_generation`: generate implementation code from a selected research plan into `data/codes/<dataset>/`.
4. `autorun`: optionally split, optimize, and execute task-wise jobs locally or on SLURM.

The first three stages run by default when invoking `cellforge`. The `autorun` stage is opt-in.

## Installation

Create an isolated Python environment:

```bash
conda create -n cellforge python=3.9
conda activate cellforge
```

Install the package:

```bash
git clone git@github.com:GabbyKoki/cellforge-new.git
cd cellforge-new
pip install -r requirements.txt
pip install -e .
```

Install and authenticate the Codex CLI:

```bash
npm install -g @openai/codex
codex login
```

## Configuration

Create a local environment file:

```bash
cp .env.example .env
```

Fill in at least one LLM provider key in `.env` for the full pipeline. Search integrations are optional.

Common settings:

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
MODEL_NAME=gpt-4o-mini
CODEGEN_BACKEND=codex
CODEX_CLI_BIN=codex
CODEX_AUTH_MODE=local
QDRANT_ENABLED=false
PUBMED_API_KEY=
PUBMED_EMAIL=
SEMANTIC_SCHOLAR_API_KEY=
CROSSREF_EMAIL=
CELLFORGE_LITERATURE_DIR=./data/literature
CELLFORGE_ONLINE_RETRIEVAL=true
CELLFORGE_WORKSPACE_DIR=.
```

The default workflow configuration is in `config.json`. Local datasets can be
placed under `data/datasets/` or supplied with `--dataset-path`. Code
generation currently uses the `codex` backend.

The unified literature retriever is shared by Task Analysis and Method Design.
It searches an external PDF corpus, PubMed, Crossref, and Semantic Scholar,
deduplicates records by DOI/PMID/Semantic Scholar ID, and writes a JSONL
provenance trace for each query. Set `CELLFORGE_LITERATURE_DIR` to the
directory containing the local literature collection.

Test literature-provider connectivity:

```bash
python scripts/test_literature_apis.py
```

## Prepare a Workspace

Initialize the runtime directories:

```bash
cellforge --init
```

Place a single-cell dataset in `data/datasets/<dataset_name>/`, or keep it
elsewhere and pass its `.h5ad` path with `--dataset-path`. Then check the local
configuration:

```bash
cellforge --doctor
```

Run the automated tests if you are developing CellForge:

```bash
python -m pytest tests
```

## Usage

Run Task Analysis:

```bash
cellforge --phase task_analysis \
  --dataset-path data/datasets/<dataset_name_or_file> \
  --task "Describe the single-cell modeling task here."
```

Use the resulting Task Analysis report to design a research plan:

```bash
cellforge --phase method_design
```

Generate code from the latest research plan:

```bash
cellforge --phase code_generation --codegen-backend codex
```

Code generation uses the Codex CLI. By default, `CODEX_AUTH_MODE=local` uses the
account authenticated by `codex login` and removes provider API keys from the
Codex subprocess. If local login is unavailable, opt into the API fallback
explicitly with `CODEX_AUTH_MODE=api` and `CODEX_API_KEY=...`. General
Task Analysis/Method Design credentials are never silently reused for code
generation.

Codex runs as a coding agent rather than a text-only generator. CellForge
creates a task-scoped workspace, materializes the research plan and
acceptance contract, asks the selected agent to edit `result.py` in place, and
then independently checks the file, Python syntax, and CLI. Failed checks are
returned to the same workspace for a bounded repair loop (five attempts by
default):

```bash
CODEGEN_MAX_REPAIR_ROUNDS=5 \
cellforge --phase code_generation --codegen-backend codex
```

Agent traces and verifier reports are stored under
`<output_dir>/.cellforge_workspaces/`. Each workspace keeps the original Codex
JSONL stream, a provider-neutral `logs/agent_events.jsonl` timeline, stderr,
the final agent message, and one verification report per attempt. A generated
`result.py` or `result_<task_id>.py` is published only after deterministic
verification passes.

Codex events are recorded as they arrive. Raw events, stderr, and final-message
files may contain model-generated command/output content, so CellForge stores
the logs with owner-only permissions; do not publish a workspace without
reviewing it.

Codex additionally enforces its `workspace-write` OS sandbox.

Run the full three-stage workflow:

```bash
cellforge \
  --dataset-path data/datasets/<dataset_name_or_file> \
  --task "Describe the single-cell modeling task here."
```

Run autorun locally:

```bash
cellforge --phase autorun \
  --dataset-path data/datasets/<dataset_name_or_file> \
  --codegen-backend codex \
  --executor local \
  --workers 2
```

Run autorun on SLURM:

```bash
cellforge --phase autorun \
  --dataset-path data/datasets/<dataset_name_or_file> \
  --codegen-backend codex \
  --executor slurm \
  --partition <partition> \
  --gres gpu:1 \
  --mem 32G
```

## Project Structure

```text
cellforge/
  Task_Analysis/      Task parsing, dataset analysis, retrieval, and planning context
  Method_Design/      Expert discussion and research-plan generation
  Code_Generation/    Verified Codex plan-to-code generation
  retrieval/          Active local/PubMed/Crossref/Semantic Scholar retrieval
  autorun/            Optional task-wise execution and SLURM helpers
scripts/              Utility scripts
tests/                Automated tests
main.py               Command-line workflow entry point
config.json           Default workflow configuration
.env.example          Environment configuration template
```

## Citation

If you use CellForge, cite the accompanying manuscript. A `CITATION.cff` file is included for software citation metadata.

## License

This project is released under the MIT License. See `LICENSE`.
