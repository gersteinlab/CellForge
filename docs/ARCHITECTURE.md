# Architecture

CellForge turns *"here is a dataset and a question"* into *"here is a trained model"*
by running three stages, each of which produces an artifact a human can read.

The paper calls the stages **Formulation**, **Ideation**, and **Execution**. The code
calls them `Task_Analysis`, `Method_Design`, and `Code_Generation` (+ `autorun`). Same
things:

| Paper | Package | Artifact produced |
|---|---|---|
| Formulation | [`cellforge/Task_Analysis/`](../cellforge/Task_Analysis/) | `data/analyses/<dataset>/` — analysis report + retrieval provenance |
| Ideation | [`cellforge/Method_Design/`](../cellforge/Method_Design/) | `data/plans/<dataset>/` — research plan |
| Execution | [`cellforge/Code_Generation/`](../cellforge/Code_Generation/) | `data/codes/<dataset>/result.py` |
| Execution | [`cellforge/autorun/`](../cellforge/autorun/) | trained model + metrics |

---

## ① Task Analysis (Formulation)

Four agents work over the dataset and the task description:

| Agent | Module | Job |
|---|---|---|
| **Dataset Analyst** | [`dataset_analyst.py`](../cellforge/Task_Analysis/dataset_analyst.py) | Characterise the assay: modality, sparsity, cell and feature counts, perturbation structure, batch covariates |
| **Problem Investigator** | [`problem_investigator.py`](../cellforge/Task_Analysis/problem_investigator.py) | Turn prose into a formal learning problem: inputs, outputs, evaluation scenarios, metrics |
| **Baseline Assessor** | [`baseline_assessor.py`](../cellforge/Task_Analysis/baseline_assessor.py) | Identify which existing methods apply and what they achieve |
| **Refinement Agent** | [`refinement_agent.py`](../cellforge/Task_Analysis/refinement_agent.py) | Iterate the report until it is internally consistent (`TASK_ANALYSIS_MAX_REFINEMENT_ROUNDS`) |

Coordination between them lives in
[`collaboration.py`](../cellforge/Task_Analysis/collaboration.py); the shared record
types are in [`data_structures.py`](../cellforge/Task_Analysis/data_structures.py).

### Retrieval

Grounding happens here and is reused downstream. The unified retriever in
[`cellforge/retrieval/`](../cellforge/retrieval/) searches:

- **a local PDF corpus** (`CELLFORGE_LITERATURE_DIR`) — the paper seeds this with a
  fixed corpus of 46 perturbation-modelling papers;
- **PubMed**, **Crossref**, and **Semantic Scholar** when `CELLFORGE_ONLINE_RETRIEVAL=true`.

Records are deduplicated by DOI / PMID / Semantic Scholar ID, and **every query writes
a JSONL provenance trace**. When the research plan cites something, you can find out
where it came from.

With `QDRANT_ENABLED=true`, the local corpus is embedded with Sentence-BERT and indexed
in [Qdrant](https://qdrant.tech/) for cosine-similarity search. The search strategy
alternates breadth-first (survey the space) and depth-first (chase one thread) passes,
rather than issuing one flat query.

Ablating retrieval degrades the final model. Literature is not decoration here — it is
what stops the design agents from reinventing 2019.

---

## ② Method Design (Ideation)

**This is the part that distinguishes CellForge from prompting a model to "design an
architecture".** It is a structured disagreement, modelled on conference peer review.

### The roster

Four domain experts, defined in [`experts.py`](../cellforge/Method_Design/experts.py):

| Expert | Cares about |
|---|---|
| **Data modeling** | Representation, normalisation, how perturbations are encoded |
| **Single-cell biology** | Whether the model respects what is known about the biology |
| **Deep learning** | Architecture, inductive bias, capacity, what will actually train |
| **Training** | Optimisation, schedules, regularisation, compute budget |

Plus a **central Critic**, which plays area chair: it reads every proposal and every
review, and it is *external to the authors* — a design point, not an accident. In the
paper's evaluation, human expert judgements correlated with an external judge panel at
r = 0.87 but with the system's own internal confidence at only r = 0.53. Self-assessment
is not reliable enough to close the loop on.

### The loop

Implemented in [`expert_discussion.py`](../cellforge/Method_Design/expert_discussion.py)
and [`graph_discussion.py`](../cellforge/Method_Design/graph_discussion.py):

```
  round t
  ┌──────────────────────────────────────────────────────┐
  │  1. each expert i drafts or revises a proposal        │
  │  2. each expert reviews every other proposal   → r_peer│
  │  3. the Critic reviews every proposal          → r_crit│
  │  4. coordination scores update                        │
  │  5. converged?  ── no ──▶ experts revise, t := t+1     │
  │        │ yes                                          │
  └────────┼──────────────────────────────────────────────┘
           ▼
     research plan
```

### The coordination score

Each expert *i* carries a score at round *t*:

$$c_t^{(i)} = 0.3\,c_{t-1}^{(i)} \;+\; 0.4\,r_{\text{crit},t}^{(i)} \;+\; \frac{0.3}{k-1}\sum_{j \neq i} r_{\text{peer},t}^{(i,j)}$$

with *k* experts. The weights say something specific about how the system is meant to
behave:

- **0.3 on history** — momentum, so a single harsh round does not throw the process;
- **0.4 on the Critic** — the external reviewer carries the most weight of any single voice;
- **0.3 spread across peers** — no individual peer can dominate, but collectively they
  outweigh nobody.

**Convergence requires two conditions simultaneously:**

1. every expert clears $c^{(i)} \ge 0.8$ — the plan is broadly good; **and**
2. $\max_{i,j} |c^{(i)} - c^{(j)}| < 0.03$ — the experts *agree that* it is good.

The second condition is what rules out "three experts are delighted and the biologist
is horrified". Debate runs at most $T_{\max} = 20$ rounds; if it has not converged by
then, the best plan so far is returned and the failure to converge is recorded.

[`refinement.py`](../cellforge/Method_Design/refinement.py) handles the revision step
between rounds.

### Cost control

Debate is the expensive stage. Three environment variables bound it:
`METHOD_DESIGN_MAX_ROUNDS` (default 6), `METHOD_DESIGN_MAX_EXPERTS` (4), and
`METHOD_DESIGN_MAX_TOKENS_PER_CALL` (350). The paper's full configuration is more
generous than the shipped defaults; the defaults are tuned so a first run does not
surprise you with a bill.

### 👤 Checkpoint

The research plan is a human approval gate. Read it. This is where a leaky split, a
metric that does not answer your question, or a biologically incoherent conditioning
scheme is cheapest to catch.

---

## ③ Code Generation (Execution)

The plan becomes a running program, in a loop designed on the assumption that the
agent will get it wrong at least once.

```
research plan
     │
     ▼
[orchestrator]  create task-scoped workspace, materialise plan + acceptance contract
     │
     ▼
[coding agent]  edit result.py in place  ◀───────────┐
     │                                               │
     ▼                                               │
[verifier]      file exists?                         │ ≤ 5 attempts
                Python syntax parses?                │ (CODEGEN_MAX_REPAIR_ROUNDS)
                CLI surface present?                 │
                acceptance contract satisfied?       │
     │                                               │
     ├── fail ──▶ report returned to same workspace ─┘
     │
     └── pass ──▶ publish result.py
```

| Module | Role |
|---|---|
| [`orchestrator.py`](../cellforge/Code_Generation/orchestrator.py) | Drives the generate → verify → repair loop |
| [`contracts.py`](../cellforge/Code_Generation/contracts.py) | The acceptance contract the generated code must satisfy |
| [`verifier.py`](../cellforge/Code_Generation/verifier.py) | Deterministic checks — no LLM in this path |
| [`codex_backend.py`](../cellforge/Code_Generation/codex_backend.py) | Codex CLI/SDK driver |
| [`codex_events.py`](../cellforge/Code_Generation/codex_events.py) | Streams agent events to a provider-neutral timeline |
| [`registry.py`](../cellforge/Code_Generation/registry.py) | Backend registry — add yours here |
| [`base.py`](../cellforge/Code_Generation/base.py) | The interface a backend implements |

Two properties matter:

- **The verifier contains no LLM.** If the checks were themselves a model call, a
  confident wrong agent could talk its way past them.
- **The repair loop is bounded.** Five attempts, then stop. An unbounded loop against a
  metered API is a way to lose money slowly.

`result.py` is published *only* after verification passes. Everything else stays in
`.cellforge_workspaces/`, written owner-only, because agent traces can echo secrets.

> The paper's experiments used [OpenHands](https://github.com/All-Hands-AI/OpenHands)
> as the coding agent. That backend is preserved at
> [`Code_Generation/legacy/openhands_backend.py`](../cellforge/Code_Generation/legacy/openhands_backend.py);
> the shipped default is Codex.

### 👤 Checkpoint

The training script is the second approval gate. Nothing is submitted to a GPU until
you have looked at it.

---

## ④ Autorun (opt-in)

[`cellforge/autorun/runner.py`](../cellforge/autorun/runner.py) splits the experiment
task-wise and dispatches to a local worker pool or Slurm `sbatch`. Splits are controlled
by `--split-ood-ratio` (held-out perturbations), `--split-val-ratio`, and
`--split-seed`.

This stage never runs implicitly. You ask for it with `--phase autorun`.

---

## What the ablations say

Each of these, removed, makes the final model worse:

- **the Critic** — proposals drift and nothing arbitrates between equally confident experts;
- **the peer-review loop** — experts optimise their own proposal in isolation;
- **retrieval** — designs regress toward generic architectures rather than
  perturbation-aware ones;
- **the coordination score** — without a convergence criterion the debate either stops
  too early or runs to the round cap.

The interesting consequence is negative: the two novel components in the results
(the trajectory-aware encoder in CPA-Traj, the diffusion denoiser in ChromDDPM) have no
counterpart in the seed corpus. They are not retrieval hits. They came out of the
debate, which is the argument for keeping the expensive part expensive.

## Where things live

```
cellforge/
├── Task_Analysis/       ① dataset characterisation, problem formalisation, baselines
├── Method_Design/       ② expert debate, peer review, Critic, refinement
├── Code_Generation/     ③ plan → verified code, backends, contracts, verifier
├── autorun/             ④ task-wise split, local and Slurm execution
├── retrieval/           unified local + PubMed + Crossref + Semantic Scholar retrieval
├── legacy/rag_v1/       superseded RAG implementation, kept for reference
├── llm.py               provider-agnostic LLM interface
└── paths.py             workspace path resolution
```

Details of what came out the other end: [MODELS.md](MODELS.md) and [RESULTS.md](RESULTS.md).
