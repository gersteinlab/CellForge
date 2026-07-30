# Provenance

Every file in this directory, and exactly where it came from. Read this before
you cite, quote, or benchmark against anything here.

## The short version

**No part of this bundle is the output of a live CellForge run.** CellForge's
Task Analysis and Method Design stages require an LLM provider key, and its Code
Generation stage requires the Codex CLI. The bundle was assembled in an
environment that had neither, and the Adamson dataset was not downloaded.

What the bundle *does* give you honestly:

- the exact **shape and depth** of the documents the pipeline produces, matched
  section-for-section against the code that serialises them;
- a **training script that genuinely runs**, verified by the repository's own
  deterministic verifier;
- **real metrics** from really executing that script — on synthetic data.

## Per-file provenance

| File | What it is | Real run output? |
| --- | --- | --- |
| [`analyses/task_analysis_report.md`](analyses/task_analysis_report.md) | Hand-written to match `TaskAnalysisReport.to_markdown()` in [`cellforge/Task_Analysis/data_structures.py`](../../../cellforge/Task_Analysis/data_structures.py), section for section. Content grounded in the Adamson case in [arXiv:2508.02276](https://arxiv.org/abs/2508.02276). | **No** — reference artifact |
| [`plans/research_plan.md`](plans/research_plan.md) | Hand-written to match the structure emitted by [`cellforge/Method_Design/refinement.py`](../../../cellforge/Method_Design/refinement.py). | **No** — reference artifact |
| [`plans/research_plan.json`](plans/research_plan.json) | Same, using the five top-level keys the code actually produces: `model_architecture`, `data_processing`, `training_strategy`, `evaluation_metrics`, `implementation_details`. | **No** — reference artifact |
| [`plans/research_plan.mmd`](plans/research_plan.mmd) | Mermaid diagram, hand-written. | **No** — reference artifact |
| [`workspace/result.py`](workspace/result.py) | Hand-written against the plan. **Executes correctly**; passes the repository's deterministic verifier; `--selftest` covers the Benjamini–Hochberg and label-parsing logic. | **No** — but it is real, working code |
| [`workspace/metrics.json`](workspace/metrics.json) | **Genuine output** of really running `result.py --synthetic --epochs 200 --seed 0`. | **Yes** — real execution, **synthetic data** |
| [`workspace/verification.json`](workspace/verification.json) | **Genuine output** of `CodeGenerationVerifier` from [`cellforge/Code_Generation/verifier.py`](../../../cellforge/Code_Generation/verifier.py) run against this workspace. Absolute paths scrubbed to `$WORKSPACE` / `$REPO`. | **Yes** — real execution |

## About the numbers in `metrics.json`

They came from a real execution, and they are **not** biologically meaningful.

```
mse    0.0198     mse_de   0.4181
pcc    0.6183     pcc_de   0.2176
r2     0.3788     r2_de   -2.5228
```

over 8 held-out perturbations, on a **6,000 cell × 400 gene synthetic matrix** —
not the ~111,000 × ~33,000 Adamson dataset.

Do not compare these to the paper. The paper reports **PCC 0.9883** on real
Adamson data; see [`docs/RESULTS.md`](../../../docs/RESULTS.md). The two numbers
measure different things on different data and the gap between them says nothing
about either.

Two of these deserve a word, because both look like bugs and neither is:

- **`r2_de` is negative.** The DE gene set is selected for having moved, so it
  has low residual variance, and R² punishes that hard. Negative R2_DE is a
  normal outcome on this metric, which is why the plan says so in advance.
- **`pcc` is only 0.62 where the paper reports 0.9883.** Genome-wide PCC on real
  data is inflated by the unperturbed baseline profile, which is shared between
  prediction and truth and dominates the correlation. The synthetic matrix has a
  far flatter baseline, so it removes most of that free signal. This is a
  property of the synthetic generator, not a measurement of the architecture.

### The synthetic generator is not neutral

`make_synthetic()` builds a world in which a gene's co-expression behaviour and
its knockdown effect both derive from the same hidden factor vector. That is
precisely the assumption CPA-X encodes, so the generator is *sympathetic to the
architecture being demonstrated*.

This is deliberate and it is a limitation. An earlier revision drew the two from
independent random matrices; under that generator the unseen-perturbation split
is unlearnable in principle and every model, correct or not, scores at chance.
A smoke test has to be passable to be useful. But it means these numbers
demonstrate that **the pipeline runs**, not that **the architecture works**.

## Reproducing the real artifacts

```bash
OMP_NUM_THREADS=4 python examples/outputs/adamson_crispri/workspace/result.py \
    --synthetic --epochs 200 --seed 0 --out /tmp/cfrun
```

Roughly 60 s on CPU. Recorded environment: Python 3.10.12, torch 2.13.0+cpu,
numpy 1.26.4. Repository state: `5c0d20f`.

Deterministic under `--seed`; a different torch build may shift the last digits.

To verify the script the way the pipeline does:

```bash
python - <<'PY'
import sys; sys.path.insert(0, '.')
from cellforge.Code_Generation.verifier import verify_generated_code
r = verify_generated_code(
    'examples/outputs/adamson_crispri/workspace/result.py',
    result_json='metrics.json',
    required_result_fields=('model', 'dataset', 'split', 'metrics'),
)
print('passed:', r.passed)
for c in r.checks:
    print(' ', c.name, c.passed)
PY
```

## Replacing this with a real run

This bundle should be replaced by genuine output the moment anyone runs the
pipeline with credentials and the real dataset.
[`scripts/export_example_run.py`](../../../scripts/export_example_run.py) does
the packaging and the secret-scrubbing:

```bash
python scripts/export_example_run.py \
    --dataset adamson_crispri \
    --out examples/outputs/adamson_crispri_real
```

If you do this, please open a PR — a real bundle is strictly more useful than
this one, and this file is the first thing that should be deleted when one
exists.
