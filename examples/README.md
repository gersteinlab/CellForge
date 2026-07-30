# Examples

Task descriptions for the six benchmark datasets, ready to hand to CellForge:

```bash
python scripts/download_datasets.py adamson --out data/datasets/

cellforge --dataset-path data/datasets/AdamsonWeissman2016_GSM2406681_10X010.h5ad \
          --task-file examples/adamson_crispri.txt
```

| File | Dataset | Task |
|---|---|---|
| [`adamson_crispri.txt`](adamson_crispri.txt) | Adamson 2016 | Single-gene CRISPRi response in K562 |
| [`norman_combinatorial.txt`](norman_combinatorial.txt) | Norman 2019 | Combinatorial CRISPRa, including unseen gene pairs |
| [`srivatsan_drug.txt`](srivatsan_drug.txt) | Srivatsan 2020 | Dose-dependent small-molecule response |
| [`schiebinger_timecourse.txt`](schiebinger_timecourse.txt) | Schiebinger 2019 | Cytokine response over a time course |
| [`papalexi_citeseq.txt`](papalexi_citeseq.txt) | Papalexi 2021 | Joint RNA + surface-protein prediction |
| [`liscovitch_atac.txt`](liscovitch_atac.txt) | Liscovitch-Brauer 2021 | Chromatin accessibility after CRISPR |

---

## What makes a good task description

The task description is the only thing standing between you and a plan that answers a
different question. A description that works names five things:

1. **The input** — what the model is given at prediction time. Be exact. "Baseline
   expression of an unperturbed cell plus the identity of the target gene" is a
   specification; "the data" is not.
2. **The output** — what it predicts, in what space (raw counts, log-normalised,
   fold-change).
3. **The evaluation scenario** — unseen perturbations? unseen cell contexts? both? This
   determines the split, and getting it wrong is the most common way to end up with a
   leaky result that looks great.
4. **The metrics** — including whether they are computed genome-wide or restricted to
   differentially expressed genes. These give very different numbers.
5. **Anything non-standard about your data** — where the perturbation label lives, how
   controls are marked, which covariates matter.

The five files above all follow that shape. Copy one and edit it.

### A worked contrast

Too vague — the agents will guess, and their guess will be generic:

> Build a model for the Norman dataset.

Specified — the agents can design against it:

> Predict post-perturbation expression in K562 cells after combinatorial CRISPRa.
> Input: the unperturbed profile and the identity of one or two activated genes.
> Output: log-normalised expression across all genes. Evaluate on gene pairs held out
> entirely from training, where individual genes may have been seen alone. Report MSE,
> PCC, and R², each genome-wide and restricted to differentially expressed genes
> (Wilcoxon, BH-adjusted p < 0.05, |log2FC| > 0.5). Non-additive interaction between
> the two activated genes is the point of the task.

---

## Using your own data

See [DATASETS.md](../docs/DATASETS.md#bring-your-own-data). If your `.h5ad` uses
non-standard field names, say so in the task description — the Dataset Analyst reads it:

> Perturbations are in `obs['guide_target']`; control cells are labelled `safe_harbor`.
> Dose in µM is in `obs['concentration']`. Ignore `obs['batch_v1']`, it is superseded by
> `obs['batch']`.
