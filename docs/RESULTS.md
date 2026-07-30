# Results

Everything here comes from the paper, [arXiv:2508.02276](https://arxiv.org/abs/2508.02276).
Where a full table is too large to reproduce faithfully, this page gives the headline
figure and points at the section of the paper that carries the complete numbers — we
would rather link than paraphrase a table into something subtly wrong.

- [How the evaluation was run](#how-the-evaluation-was-run)
- [Metrics](#metrics)
- [Headline results](#headline-results)
- [Baselines](#baselines)
- [Against other autonomous systems](#against-other-autonomous-systems)
- [Does the judging mean anything?](#does-the-judging-mean-anything)
- [Where CellForge is weak](#where-cellforge-is-weak)
- [Cost](#cost)
- [Failure modes](#failure-modes)
- [Reproducing this](#reproducing-this)

---

## How the evaluation was run

| | |
|---|---|
| Datasets | 6 (see [DATASETS.md](DATASETS.md)) |
| Readouts | 7 — Papalexi CITE-seq contributes both RNA and protein |
| Modalities | scRNA-seq, scATAC-seq, CITE-seq |
| Perturbation types | CRISPRi, CRISPRa (single and combinatorial), small molecules, cytokines |
| Cross-validation | 5-fold |
| Repeats | 3 independent runs |
| Averaging | **perturbation-centric** — averaged over perturbations, not over cells |
| Early stopping | patience 10 |
| LR schedule | cosine annealing |

Perturbation-centric averaging matters. Averaging over cells lets a few
heavily-sampled perturbations dominate the score; averaging over perturbations asks the
harder and more useful question, *does this work for a perturbation I have few cells
for?*

---

## Metrics

| Metric | Direction | What it tells you |
|---|---|---|
| MSE | ↓ | Average squared error across all genes |
| PCC | ↑ | Linear correlation with the observed profile |
| R² | ↑ | Variance explained |
| MSE_DE | ↓ | MSE restricted to differentially expressed genes |
| PCC_DE | ↑ | PCC restricted to DE genes |
| R2_DE | ↑ | R² restricted to DE genes |
| DEG Recall | ↑ | Fraction of true DE genes recovered |
| ROC-AUC / PR-AUC | ↑ | For the accessibility (scATAC) readout |

**The `_DE` variants are the ones to look at.** Most genes barely move under most
perturbations, so a model that predicts "nothing changed" scores respectably on plain
MSE and PCC while being scientifically useless. Restricting to DE genes removes that
free ride.

DE genes are defined by Wilcoxon rank-sum with Benjamini–Hochberg correction, keeping
genes with adjusted *p* < 0.05 and |log₂FC| > 0.5.

---

## Headline results

### Adamson — CRISPRi, K562, scRNA-seq

| Method | PCC ↑ |
|---|---|
| **CellForge — CPA-X** | **0.9883** |
| Best baseline | lower |

The highest correlation of any method compared on this dataset.

### Norman — combinatorial CRISPRa, scRNA-seq

| Method | MSE_DE ↓ |
|---|---|
| **CellForge — scGen-X** | **0.1736** |

Combinatorial activation is the harder setting: the model has to predict the effect of
gene pairs whose *joint* perturbation it has not seen, and those effects are frequently
non-additive.

### Schiebinger — cytokine time course, scRNA-seq

| Method | DEG Recall ↑ |
|---|---|
| CellForge — CPA-Traj | 0.535 |

### Papalexi — CITE-seq, protein readout

| Method | Recall ↑ |
|---|---|
| CellForge — totalGAT | 0.420 |

Full per-dataset, per-metric tables for all six datasets and all baselines are in the
paper's Results section.

---

## Baselines

Every CellForge-designed model was compared against:

**Perturbation-specific methods** — CPA, scGen, CondOT, Biolord, GEARS, ChemCPA, CellFlow
**Foundation models** — scGPT, STATE
**Classical** — random forest, linear regression
**Control** — unperturbed profile

The unperturbed control deserves a note: it is the "predict no change" baseline, and on
plain MSE it is a genuinely competitive number. That is the point of including it.

---

## Against other autonomous systems

CellForge was compared with five other systems that also attempt autonomous research:

- OpenAI Deep Research
- Perplexity Deep Research
- Gemini Deep Research
- [Biomni](https://github.com/snap-stanford/Biomni)
- a single-LLM baseline (Claude 3.7, one shot)

Outputs were scored blind by a panel of five judge models — Claude 3.7, DeepSeek-R1,
OpenAI o1, Qwen-plus, and Llama 3.1 — across 15 task–criterion combinations.

> **CellForge ranked first in 14 of 15.**

---

## Does the judging mean anything?

A fair question to ask of any LLM-as-judge result. Three checks:

| Check | Value | Reading |
|---|---|---|
| Inter-judge agreement | Pearson **0.88 – 0.93** | The five judges largely agree; the ranking is not one model's idiosyncrasy |
| Human experts vs. judge panel | r = **0.87** | The panel tracks human expert opinion closely |
| Human experts vs. CellForge's own internal confidence | r = **0.53** | **Self-assessment is much weaker** |

That last row is the load-bearing one. It is the empirical justification for the Critic
being external to the authoring agents: a system asked to score its own work correlates
with expert judgement at roughly half the strength of an outside panel.

---

## Where CellForge is weak

Stated plainly, because you will find out anyway:

- **Sparse, low-count modalities.** 0.535 DEG recall on the cytokine time course and
  0.420 protein recall on Papalexi are the weakest results in the paper. Surface-protein
  prediction in CITE-seq and temporally-resolved responses both remain hard.
- **Execution is the bottleneck, not ideation.** 41% of failures are computation
  execution errors and another 23% are invalid types or unsupported operations. The
  agents are better at proposing architectures than at getting them to run first try.
- **Cost is real.** 4–8 GPU-hours and 200–400k completion tokens per end-to-end run is
  not something you fire off casually.
- **The benchmark is six datasets.** Broad across modality and perturbation type, but
  six. Adding a seventh is the [most valuable contribution](../CONTRIBUTING.md#1-add-a-dataset)
  available.

---

## Cost

Per end-to-end run:

| | Simple task | Complex task |
|---|---|---|
| Prompt tokens | ~40,000 | ~80,000 |
| Completion tokens | ~200,000 | ~400,000 |
| Generated model | 10–30M parameters | 10–30M parameters |
| Wall clock, 1 GPU | ~4h | ~8h |

Completion dominates by roughly 5:1, which is what you would expect from a system whose
expensive stage is agents writing proposals, reviews, and code rather than reading.

---

## Failure modes

Distribution across runs:

| Failure | Share |
|---|---|
| Computation execution error | 41% |
| Invalid type or unsupported operation | 23% |
| Error-recovery failure | 16% |
| Model misconfiguration | 6% |
| Data access | 5% |
| Other system-level | 5% |
| **Hallucinated structures** | **4%** |

The headline here is the last row. The failure people expect from an LLM writing model
code — inventing a layer or an API that does not exist — is the *rarest* one, at 4%.
Two thirds of failures are ordinary execution problems: shape mismatches, dtype errors,
operations unsupported on the given tensor. Those are engineering bugs, and the bounded
repair loop resolves many of them; error-*recovery* failure at 16% is where the repair
loop itself gives up.

---

## Reproducing this

```bash
# 1. get the data
python scripts/download_datasets.py --all --out data/datasets/

# 2. run the pipeline for one benchmark
cellforge --dataset-path data/datasets/adamson.h5ad \
          --task-file examples/adamson_crispri.txt

# 3. train the model it designed, with the paper's split settings
cellforge --phase autorun \
          --dataset-path data/datasets/adamson.h5ad \
          --executor slurm --partition <your-gpu-partition> \
          --gres gpu:1 --mem 32G --slurm-time 08:00:00 \
          --split-ood-ratio 0.2 --split-val-ratio 0.1 --split-seed 42
```

**You will not get byte-identical results, and you should not expect to.** CellForge
designs a *new* architecture on each run; LLM sampling, retrieval results that shift as
the literature grows, and the debate trajectory all vary. What should reproduce is the
*quality band* — a competitive perturbation model, not the exact CPA-X described in
[MODELS.md](MODELS.md).

If you get something materially worse, that is worth an issue. Attach the research
plan.
