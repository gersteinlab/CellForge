# FAQ

- [What it is](#what-it-is)
- [Cost](#cost)
- [Failure modes](#failure-modes)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

---

## What it is

### Is this AutoML for single-cell?

No, and the distinction matters. AutoML searches a predefined space — architectures,
hyperparameters, pipeline components you enumerated in advance. CellForge writes a
research plan and then writes the code, so the space is whatever the agents can
articulate. Two components in the paper's results (the trajectory-aware encoder in
CPA-Traj, the diffusion denoiser in ChromDDPM) are not in any menu and not in the seed
literature corpus.

The flip side: an AutoML search is reproducible and CellForge is not, in the strict
sense. See [below](#will-i-reproduce-the-papers-models).

### Will I reproduce the paper's models?

No, and you should not try. CellForge designs a *new* architecture on each run. LLM
sampling, retrieval results that change as the literature grows, and the trajectory of
the debate all vary between runs.

What should reproduce is the **quality band** — a competitive perturbation model — not
CPA-X specifically. If you get something materially worse than the paper's numbers on
the same dataset, that is a bug report worth filing; please attach the research plan.

### Do you ship pretrained weights?

No. CellForge ships the process that produces models, not the models. The six
architectures in [MODELS.md](MODELS.md) are documented outputs, not a model zoo.

### Which LLM should I use?

Any of the supported providers works. A frontier model produces noticeably better
research plans than a small one — the Method Design stage is genuinely hard reasoning,
not formatting. Use `gpt-4o-mini` or similar for smoke tests, then switch up.

For self-hosted or sensitive data, point `CUSTOM_API_URL` at any OpenAI-compatible
endpoint (vLLM, Ollama, TGI, OpenRouter).

### Can I use it on my own data?

Yes — see [DATASETS.md](DATASETS.md#bring-your-own-data). Any AnnData `.h5ad` with a
perturbation label in `.obs`.

### Does it work on non-perturbation tasks?

The framework is not perturbation-specific, but the agent prompts, the seed literature
corpus, and the entire evaluation are. Expect degraded plans outside perturbation
modelling. If you try it on a different problem class, an issue reporting how it went
would be genuinely useful.

---

## Cost

### What does one run cost?

| | Simple task | Complex task |
|---|---|---|
| Prompt tokens | ~40k | ~80k |
| Completion tokens | ~200k | ~400k |
| GPU wall clock | ~4h | ~8h |

Completion dominates roughly 5:1. Convert to dollars using your provider's rates —
the ratio is what matters for choosing a model.

### How do I make it cheaper?

```bash
MODEL_NAME=gpt-4o-mini \
METHOD_DESIGN_MAX_ROUNDS=2 \
METHOD_DESIGN_MAX_EXPERTS=2 \
METHOD_DESIGN_MAX_TOKENS_PER_CALL=200 \
CELLFORGE_ONLINE_RETRIEVAL=false \
cellforge --phase task_analysis --dataset-path data/datasets/adamson.h5ad --task "..."
```

Also: run stage by stage rather than end to end. Reading the analysis report before
paying for Method Design catches a misread task early, and Method Design is the
expensive stage.

### Where does the money actually go?

Method Design. Four experts × peer reviews × Critic × up to 20 rounds is a quadratic-ish
number of calls in the expert count, which is why `METHOD_DESIGN_MAX_EXPERTS` is the
single most effective cost lever.

### Do I need a GPU?

Only for the opt-in `autorun` stage. Task Analysis, Method Design, and Code Generation
are all API calls and run fine on a laptop — they produce a training script you can run
wherever you like.

---

## Failure modes

Measured across runs:

| Failure | Share | What it actually is |
|---|---|---|
| Computation execution error | 41% | Shape mismatch, OOM, CUDA error in generated code |
| Invalid type / unsupported operation | 23% | Operation not supported for a dtype or sparse layout |
| Error-recovery failure | 16% | The repair loop tried and could not fix it |
| Model misconfiguration | 6% | Bad hyperparameters, incompatible layer sizes |
| Data access | 5% | Path, permission, or format problem |
| Other system-level | 5% | Environment, dependency, cluster |
| **Hallucinated structures** | **4%** | An invented layer, API, or method |

**The surprise is the last row.** The failure people expect from an LLM writing model
code is the rarest one. Two thirds of failures are ordinary engineering bugs, and the
verifier plus bounded repair loop resolves many before you ever see them. The 16% in
"error-recovery failure" is where that loop exhausted its five attempts.

### It failed. Now what?

1. Read the verification report in `<output_dir>/.cellforge_workspaces/<task>/`. There
   is one per repair attempt and it says exactly which check failed.
2. Raise the budget: `CODEGEN_MAX_REPAIR_ROUNDS=8`.
3. Fix `result.py` by hand. It is a normal Python file and often the problem is two lines.
4. If the *plan* was wrong rather than the code, that is a
   [bad-plan report](https://github.com/gersteinlab/CellForge/issues/new?template=bad_plan.yml)
   and we want it.

---

## Troubleshooting

### `cellforge --doctor` reports no LLM provider

The `.env` file is read from the workspace root. If you passed `--workspace`, the `.env`
must be *there*, not in the repo directory.

### Code generation fails immediately

Check the Codex CLI:

```bash
codex --version
codex login
```

With the default `CODEX_AUTH_MODE=local`, CellForge deliberately strips provider API
keys from the Codex subprocess — your `OPENAI_API_KEY` will not be picked up, and that
is intentional. To use a key explicitly, set `CODEX_AUTH_MODE=api` and `CODEX_API_KEY`.

### Literature retrieval returns nothing

```bash
python scripts/test_literature_apis.py
```

PubMed rate-limits unauthenticated clients hard. Set `PUBMED_API_KEY` and
`PUBMED_EMAIL`. If you are offline, set `CELLFORGE_ONLINE_RETRIEVAL=false` and rely on
a local PDF corpus in `CELLFORGE_LITERATURE_DIR`.

### Method Design never converges

Convergence requires *every* expert above 0.8 **and** the widest pairwise gap below
0.03. Hitting the round cap without converging is a legitimate outcome — the best plan
so far is returned and the non-convergence recorded. If it happens every time, the task
description is probably underspecified. Say what the input is, what the output is, how
it will be evaluated, and what counts as success.

### Slurm jobs die immediately

The default `--slurm-time 01:00:00` will kill a real training run. Set it to 8 hours or
more. Also check that `--conda-env` names an environment that exists on the compute
node, not just the login node.

### Out of memory on scATAC

200k peaks is a wide feature space. Raise `--mem` to 64G or higher and lower
`--workers`.

---

## Limitations

Stated plainly:

- **Six datasets.** Broad in modality and perturbation type, but six. Adding a seventh
  is the highest-value [contribution](../CONTRIBUTING.md) available.
- **Sparse modalities are weak.** 0.535 DEG recall on the cytokine time course, 0.420
  protein recall on Papalexi CITE-seq.
- **Execution, not ideation, is the bottleneck.** 64% of failures are execution-level.
- **Prompt injection through retrieved literature is unmitigated.** A hostile PDF in
  your corpus can influence a research plan. This is one of the reasons the plan is a
  human checkpoint. See [SECURITY.md](../SECURITY.md).
- **Self-assessment is unreliable.** The system's internal confidence correlates with
  human expert judgement at r = 0.53, against r = 0.87 for an external judge panel.
  Do not treat a high internal score as validation — read the plan.
- **Not reproducible in the strict sense.** By design; see
  [above](#will-i-reproduce-the-papers-models).
- **Not a replacement for you.** It is a very fast first draft by a well-read
  collaborator who has never run a wet-lab experiment.

---

Still stuck? [Open a discussion](https://github.com/gersteinlab/CellForge/discussions)
or [file an issue](https://github.com/gersteinlab/CellForge/issues/new/choose).
