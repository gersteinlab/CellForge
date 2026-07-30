# Quickstart

From nothing to a generated model in about an hour of wall clock, five minutes of which
is you.

- [Install](#install)
- [Configure](#configure)
- [Prepare a workspace](#prepare-a-workspace)
- [Get data](#get-data)
- [Run the pipeline](#run-the-pipeline)
- [Reading the output](#reading-the-output)
- [Running on Slurm](#running-on-slurm)
- [Keeping it cheap](#keeping-it-cheap)

---

## Install

CellForge supports Python 3.9 through 3.12.

```bash
git clone https://github.com/gersteinlab/CellForge.git
cd CellForge

conda create -n cellforge python=3.11 -y
conda activate cellforge

pip install -e .
```

The full install pulls the scientific stack — torch, scanpy, anndata, scvi-tools,
transformers, qdrant-client. Expect several minutes and a few GB.

### Code-generation backend

Code generation runs through the Codex CLI:

```bash
npm install -g @openai/codex
codex login
```

`codex login` authenticates the agent independently of your provider API keys, which is
the default and recommended posture. If you cannot log in interactively, opt into the
API fallback explicitly:

```bash
CODEX_AUTH_MODE=api
CODEX_API_KEY=...
```

CellForge will not silently reuse your Task Analysis or Method Design credentials for
code generation.

---

## Configure

```bash
cp .env.example .env
```

The only strictly required setting is one LLM provider key:

```bash
OPENAI_API_KEY=sk-...
# or ANTHROPIC_API_KEY / DEEPSEEK_API_KEY / QWEN_API_KEY / LLAMA_API_KEY
MODEL_NAME=gpt-4o-mini
```

Using a self-hosted or third-party OpenAI-compatible endpoint (vLLM, Ollama, TGI,
OpenRouter):

```bash
CUSTOM_API_URL=http://localhost:8000/v1
CUSTOM_API_KEY=whatever-your-server-expects
```

Everything else is optional. The settings worth knowing about:

| Variable | Default | What it controls |
|---|---|---|
| `METHOD_DESIGN_MAX_ROUNDS` | `6` | Debate rounds before the design is forced to close |
| `METHOD_DESIGN_MAX_EXPERTS` | `4` | How many domain experts participate |
| `METHOD_DESIGN_MAX_TOKENS_PER_CALL` | `350` | Per-turn token cap during debate |
| `TASK_ANALYSIS_MAX_REFINEMENT_ROUNDS` | `3` | Refinement passes over the analysis report |
| `CODEGEN_MAX_REPAIR_ROUNDS` | `5` | Verification-failure repair attempts |
| `CELLFORGE_LITERATURE_DIR` | `./data/literature` | Local PDF corpus |
| `CELLFORGE_ONLINE_RETRIEVAL` | `true` | Whether to query PubMed / Crossref / Semantic Scholar |
| `QDRANT_ENABLED` | `false` | Vector search over the local corpus |
| `CELLFORGE_WORKSPACE_DIR` | `.` | Where runtime directories are created |

Literature providers are optional but improve grounding. Check connectivity:

```bash
python scripts/test_literature_apis.py
```

---

## Prepare a workspace

```bash
cellforge --init
```

Creates `data/datasets/`, `data/analyses/`, `data/plans/`, `data/codes/`, and
`data/literature/` under `CELLFORGE_WORKSPACE_DIR`.

Keep runtime output off your home directory — on a cluster especially:

```bash
cellforge --workspace /scratch/$USER/cellforge --init
```

Then verify:

```bash
cellforge --doctor
```

```
🩺 CellForge workspace doctor
Workspace: /scratch/you/cellforge
✅ configuration: /scratch/you/cellforge/config.json
✅ datasets directory: /scratch/you/cellforge/data/datasets
✅ literature directory: /scratch/you/cellforge/data/literature
✅ LLM provider: environment variables
✅ Python: 3.11.6
```

`--doctor` never mutates anything, so it is safe to run any time.

---

## Get data

```bash
python scripts/download_datasets.py --list
python scripts/download_datasets.py adamson --out data/datasets/
```

Or bring your own `.h5ad`. CellForge expects an AnnData object with a perturbation
label in `.obs`; see [DATASETS.md](DATASETS.md#bring-your-own-data) for the expected
fields.

---

## Run the pipeline

### All at once

```bash
cellforge \
  --dataset-path data/datasets/adamson.h5ad \
  --task "Predict post-perturbation gene expression in K562 cells after CRISPRi \
knockdown. Evaluate on unseen perturbations and unseen cell contexts. Report MSE, \
PCC, and R2, both overall and restricted to differentially expressed genes."
```

Long task descriptions are easier to keep in a file:

```bash
cellforge --dataset-path data/datasets/adamson.h5ad --task-file examples/adamson_crispri.txt
```

### Stage by stage

Recommended the first time, so you can read each artifact before paying for the next
stage.

```bash
# ① understand the task and the data, ground it in literature
cellforge --phase task_analysis \
  --dataset-path data/datasets/adamson.h5ad \
  --task-file examples/adamson_crispri.txt

# ② let the expert agents debate an architecture
cellforge --phase method_design

# ③ turn the winning plan into verified, runnable code
cellforge --phase code_generation --codegen-backend codex
```

### Train what it wrote

The execution stage is opt-in and never runs implicitly.

```bash
cellforge --phase autorun \
  --dataset-path data/datasets/adamson.h5ad \
  --executor local \
  --workers 2
```

---

## Reading the output

```
data/
├── analyses/<dataset>/
│   ├── analysis_report.*        # task decomposition, data characterisation, baselines
│   └── retrieval/*.jsonl        # every literature query and what it returned
├── plans/<dataset>/
│   └── research_plan.*          # the architecture the agents converged on, with rationale
└── codes/<dataset>/
    ├── result.py                # published only after verification passes
    └── .cellforge_workspaces/   # agent traces, verifier reports, stderr (owner-only)
```

**Read the research plan before running the code.** That is the checkpoint that makes
the rest of this safe, and it is where you will catch a leaky split or a metric that
does not answer your question.

Inside a workspace under `.cellforge_workspaces/` you get the raw agent event stream, a
provider-neutral `logs/agent_events.jsonl` timeline, stderr, the final agent message,
and one verification report per repair attempt. These files can contain
model-generated commands and output — review before sharing.

---

## Running on Slurm

```bash
cellforge --phase autorun \
  --dataset-path data/datasets/adamson.h5ad \
  --codegen-backend codex \
  --executor slurm \
  --partition gpu \
  --gres gpu:1 \
  --mem 32G \
  --slurm-time 08:00:00 \
  --cpus-per-task 4 \
  --conda-env cellforge
```

Autorun splits the work task-wise and submits one job per task.

| Flag | Default | Notes |
|---|---|---|
| `--executor` | `slurm` | Pass `local` to run in-process |
| `--partition` | `scavenge_gpu` | Set this; the default is site-specific |
| `--gres` | `gpu:1` | |
| `--mem` | `32G` | 200k-peak scATAC needs more |
| `--slurm-time` | `01:00:00` | Raise it — full runs take 4–8h |
| `--cpus-per-task` | `4` | |
| `--conda-env` | `cellforge` | Activated inside the job script |
| `--workers` | `4` | Concurrent task-wise jobs |
| `--max-tasks` | unlimited | Cap the number of jobs |
| `--split-ood-ratio` | `0.2` | Held-out perturbation fraction |
| `--split-val-ratio` | `0.1` | Validation cells within the in-distribution split |
| `--split-seed` | `42` | |

The default `--slurm-time 01:00:00` will kill a real training run. Set it deliberately.

---

## Keeping it cheap

A full run costs roughly 40–80k prompt and 200–400k completion tokens. To smoke-test
the plumbing for a few cents:

```bash
MODEL_NAME=gpt-4o-mini \
METHOD_DESIGN_MAX_ROUNDS=2 \
METHOD_DESIGN_MAX_EXPERTS=2 \
METHOD_DESIGN_MAX_TOKENS_PER_CALL=200 \
CELLFORGE_ONLINE_RETRIEVAL=false \
cellforge --phase task_analysis --dataset-path data/datasets/adamson.h5ad \
          --task "Smoke test."
```

Expect a shallow, unconvincing plan — that is the point. Turn the limits back up once
the pipeline runs end to end.

More in the [FAQ](FAQ.md).
