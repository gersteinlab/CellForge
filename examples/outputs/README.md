# Example outputs

What CellForge actually hands you, so you can judge it before spending a token.

The pipeline is not cheap — a complex task runs roughly 80k prompt / 400k
completion tokens, plus 4–8 GPU-hours if you train the model it writes. It is
reasonable to want to see the deliverables first.

> [!IMPORTANT]
> **The documents here are reference artifacts, not transcripts of a live run.**
> They are hand-written to match, section for section, the schemas the code
> actually serialises. The `result.py` in the bundle is real, runnable, verified
> code, and its `metrics.json` and `verification.json` are the genuine output of
> really executing it — on **synthetic** data.
>
> Every file is accounted for individually in
> [`adamson_crispri/PROVENANCE.md`](adamson_crispri/PROVENANCE.md). Read that
> before citing anything here. For real benchmark numbers, see
> [`docs/RESULTS.md`](../../docs/RESULTS.md).

## The bundle

[`adamson_crispri/`](adamson_crispri/) — CRISPRi knockdown response in K562
cells, the entry-point task of the six benchmarks. Input task description:
[`examples/adamson_crispri.txt`](../adamson_crispri.txt).

```
adamson_crispri/
├── PROVENANCE.md                     ← read this first
├── analyses/
│   └── task_analysis_report.md       Stage 1 — Task Analysis
├── plans/
│   ├── research_plan.md              Stage 2 — Method Design  👤 checkpoint
│   ├── research_plan.json
│   └── research_plan.mmd
└── workspace/
    ├── result.py                     Stage 3 — Code Generation  👤 checkpoint
    ├── metrics.json                  real output of really running it
    └── verification.json             real output of the deterministic verifier
```

The two 👤 markers are the human checkpoints. A real run stops at each one and
waits for you: after the research plan, and again before any GPU job starts.

## What each stage produces

**Task Analysis** → [`task_analysis_report.md`](adamson_crispri/analyses/task_analysis_report.md)

Five fixed sections: dataset analysis, problem investigation, baseline
assessment, the refinement round-trip between the agents, and final
recommendations. The section layout is not free-form — it comes from
`TaskAnalysisReport.to_markdown()`.

The part worth reading is §2, *Key Challenges*. It is where the analysis either
identifies the actual crux of the task or does not. Here the crux is that
holding out whole perturbations makes a free per-perturbation embedding table
useless, because held-out rows never receive gradient. If a plan misses that,
the model it specifies cannot work, and this is the document where you catch it.

**Method Design** → [`research_plan.md`](adamson_crispri/plans/research_plan.md)

Architecture, preprocessing, training strategy, evaluation protocol, and the
acceptance criteria the generated code must satisfy. Emitted as markdown, JSON,
and a mermaid diagram. **This is the first human checkpoint** — nothing is
executed until you approve it, and it is much cheaper to reject a plan than to
debug the code written from it.

**Code Generation** → [`result.py`](adamson_crispri/workspace/result.py)

A single self-contained training script with a command-line interface. Before it
is handed to you it must pass a **deterministic, non-LLM verifier** — the model
does not get to grade its own work. The contract:

| Check | What it means |
| --- | --- |
| `file_exists` | the entrypoint is where the plan said it would be |
| `python_compile` | it parses |
| `cli_help` | `--help` exits 0 |
| `acceptance_*` | plan-specific commands exit 0 (here, `--selftest`) |
| `result_json` | `metrics.json` parses and has the required fields |

Failures are fed back to the same agent session as structured JSON for a bounded
repair loop of at most five attempts. The real result of running that verifier
against this workspace is committed as
[`verification.json`](adamson_crispri/workspace/verification.json) — all five
checks pass.

## Try it

`result.py` imports torch/numpy/anndata lazily, so the cheap paths need nothing
but the standard library:

```bash
cd examples/outputs/adamson_crispri/workspace
python result.py --help
python result.py --selftest
```

To reproduce `metrics.json` (~60 s on CPU, needs torch and numpy):

```bash
OMP_NUM_THREADS=4 python result.py --synthetic --epochs 200 --seed 0 --out /tmp/cfrun
```

To run it on the real dataset, fetch it first with
[`scripts/download_datasets.py`](../../scripts/download_datasets.py):

```bash
python scripts/download_datasets.py adamson --out data/datasets
python result.py --data data/datasets/AdamsonWeissman2016_GSM2406681_10X010.h5ad \
    --split unseen_perturbation --epochs 200 --out /tmp/adamson
```

## Contributing a real bundle

A bundle from an actual run is strictly more useful than this one, and we would
like to replace this with one. If you have run the pipeline with credentials and
a real dataset:

```bash
python scripts/export_example_run.py --dataset adamson_crispri --out examples/outputs/adamson_crispri_real
```

That collects the artifacts, strips API keys, absolute paths, and hostnames, and
writes a `PROVENANCE.md` stub for you to complete. Then open a PR. See
[`CONTRIBUTING.md`](../../CONTRIBUTING.md).
