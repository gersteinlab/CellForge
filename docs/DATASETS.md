# Datasets

CellForge is benchmarked on six perturbation datasets spanning three modalities and
four perturbation types.

- [The benchmark](#the-benchmark)
- [Getting the data](#getting-the-data)
- [Preprocessing](#preprocessing)
- [Splits](#splits)
- [Differential expression](#differential-expression)
- [Bring your own data](#bring-your-own-data)
- [Adding a dataset to the benchmark](#adding-a-dataset-to-the-benchmark)

---

## The benchmark

| Dataset | Perturbation | Modality | Cells | Features | Accession |
|---|---|---|---|---|---|
| **Adamson 2016** | CRISPRi (single gene) | scRNA-seq | 111k | 33k genes | [GSE90546](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546) |
| **Norman 2019** | CRISPRa (single + combinatorial) | scRNA-seq | 84k | 17k genes | [GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344) |
| **Srivatsan 2020** | small molecules, multiple doses | scRNA-seq | 81k | 18k genes | [GSE139944](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139944) |
| **Schiebinger 2019** | cytokine stimulation, time course | scRNA-seq | 65k | 17k genes | via scPerturb |
| **Papalexi 2021** | CRISPR | CITE-seq | 171k | 18k genes + 200 proteins | via scPerturb |
| **Liscovitch-Brauer 2021** | CRISPR | scATAC-seq | 58k | 200k peaks | via scPerturb |

Six datasets, **seven readouts** — Papalexi contributes an RNA readout and a protein
readout, which are scored separately.

Coverage is deliberate:

| | genetic KD | genetic OE | combinatorial | chemical | cytokine | temporal |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Adamson | ✅ | | | | | |
| Norman | | ✅ | ✅ | | | |
| Srivatsan | | | | ✅ | | |
| Schiebinger | | | | | ✅ | ✅ |
| Papalexi | ✅ | | | | | |
| Liscovitch-Brauer | ✅ | | | | | |

| | scRNA-seq | scATAC-seq | CITE-seq |
|---|:---:|:---:|:---:|
| | 4 datasets | 1 | 1 |

---

## Getting the data

Preprocessed `.h5ad` files for all six are distributed by
[**scPerturb**](https://projects.sanderlab.org/scperturb/), archived at
DOI [10.5281/zenodo.13350497](https://doi.org/10.5281/zenodo.13350497). Use these
rather than reprocessing from GEO — it is what the paper used, and it removes a large
source of irreproducibility.

```bash
python scripts/download_datasets.py --list                # what is available
python scripts/download_datasets.py adamson               # one
python scripts/download_datasets.py norman papalexi       # several
python scripts/download_datasets.py --all --out data/datasets/
```

The script verifies checksums and skips files already present. Raw counts from GEO are
linked in the table above if you want to reprocess from scratch.

**Do not commit these files.** They range from hundreds of MB to several GB and
`data/` is gitignored.

---

## Preprocessing

The scPerturb releases are already normalised and harmonised. Beyond that:

| Step | Setting |
|---|---|
| Count normalisation | library-size normalise, log1p |
| Gene filtering | as distributed by scPerturb |
| Perturbation labels | harmonised to scPerturb's `perturbation` field in `.obs` |
| Control cells | scPerturb's non-targeting / vehicle annotation |

For scATAC-seq the feature space is a 200k-peak binary-ish accessibility matrix, which
is why that readout is scored with ROC-AUC and PR-AUC rather than MSE and PCC.

---

## Splits

Two evaluation scenarios, and they test different things:

### Unseen perturbations (out-of-distribution)

A fraction of perturbations is held out entirely — the model never sees a single cell
carrying them. This is the scientifically interesting question: *can you predict the
effect of an experiment nobody has run?*

Controlled by `--split-ood-ratio` (default `0.2`).

### Unseen cell contexts

All perturbations are seen in training, but the held-out cells have baseline expression
profiles that were not. This tests generalisation across cell state rather than across
perturbation.

Controlled by `--split-val-ratio` (default `0.1`, applied within the in-distribution
split).

### Cross-validation

5-fold, 3 independent runs, `--split-seed 42` in the paper. Results are averaged
**per perturbation, not per cell** — see [RESULTS.md](RESULTS.md#how-the-evaluation-was-run)
for why that choice changes the numbers.

---

## Differential expression

DE genes are called with:

- **Wilcoxon rank-sum test**, perturbed versus control
- **Benjamini–Hochberg** correction for multiple testing
- kept if **adjusted *p* < 0.05** and **|log₂FC| > 0.5**

Every `_DE` metric (MSE_DE, PCC_DE, R2_DE) and DEG Recall is computed over that set.

This definition is doing real work. Most genes do not respond to most perturbations, so
a model predicting "no change" scores well on genome-wide MSE. Restricting to genes that
actually moved is what makes the metric discriminative.

---

## Bring your own data

CellForge accepts any AnnData `.h5ad`:

```bash
cellforge --dataset-path /path/to/your.h5ad --task "..."
```

The Dataset Analyst inspects the object directly, so there is no rigid schema, but it
works best when:

| Field | Expectation |
|---|---|
| `adata.X` | Normalised expression (log1p of library-size-normalised counts) |
| `adata.obs['perturbation']` | Perturbation label per cell |
| control cells | Labelled `control`, `non-targeting`, or `NT` |
| `adata.obs` covariates | Batch, cell type, dose, time — anything the model should condition on or adjust for |
| `adata.var_names` | Gene symbols or a consistent stable ID |

If your labels use different names, say so in the task description. The analyst reads
it:

> "Perturbations are in `obs['guide_target']`; control cells are labelled `safe_harbor`.
> Dose in µM is in `obs['concentration']`."

---

## Adding a dataset to the benchmark

The single most valuable contribution to this project. A complete one has:

1. **An entry in `scripts/download_datasets.py`** — stable URL, SHA-256, file size.
2. **A row in the table above** — assay, perturbation type, cell and feature counts,
   accession.
3. **A documented split** — which perturbations are held out and on what principle.
   "Random 20%" is fine if you say so.
4. **At least one baseline number** so the comparison table is not empty.
5. **A `.h5ad` that is publicly downloadable without registration.** If it needs a data
   access agreement, it cannot go in the benchmark, though CellForge will still run on it
   locally.

Open a [feature request](https://github.com/gersteinlab/CellForge/issues/new?template=feature_request.yml)
first if you want to check whether a dataset is a good fit before doing the work.
