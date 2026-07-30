> [!IMPORTANT]
> **This is a reference artifact, not the transcript of a live run.**
>
> It was written by hand to match the structure that
> [`cellforge/Method_Design/refinement.py`](../../../../cellforge/Method_Design/refinement.py)
> emits, and its content is grounded in the Adamson case reported in
> [arXiv:2508.02276](https://arxiv.org/abs/2508.02276). A real run writes this
> file as `research_plan_YYYYmmdd_HHMMSS.md` alongside a `.json` and a `.mmd`.
>
> This is the document a real run puts in front of you at the **first human
> checkpoint**. Nothing is executed until you approve it. See
> [`../PROVENANCE.md`](../PROVENANCE.md).

# Research Plan — Adamson CRISPRi perturbation response

**Task type**: `gene_knockout`
**Target model**: CPA-X
**Machine-readable form**: [`research_plan.json`](research_plan.json)
**Diagram**: [`research_plan.mmd`](research_plan.mmd)

---

## Model Architecture

**CPA-X** — a compositional autoencoder with adversarial disentanglement.

One design decision determines whether this task is solvable at all. The evaluation
holds out **whole perturbations**, so at test time the model is asked about target genes
it has never seen perturbed. If the perturbation is represented as a free row in an
embedding table, those rows never receive gradient during training and still hold their
random initialisation at test time — the model is predicting noise, and no amount of
training fixes it.

CPA-X instead derives the perturbation representation from the **target gene's own
embedding**, and ties that embedding to the decoder's output projection. Every gene
therefore receives gradient from every reconstruction, whether or not it was ever
perturbed. A held-out target gene arrives at test time with a trained representation.

| Component | Specification |
| --- | --- |
| Encoder | MLP `n_genes → 512 → 512 → 128`, LayerNorm + ReLU + dropout 0.1 |
| Gene embedding | `n_genes × 128`, shared between perturbation conditioning and decoding |
| Perturbation encoder | MLP `128 → 512 → 128` applied to the target gene's embedding; summed over targets so combinatorial perturbations use the same code path; zero vector for controls |
| Composition | Additive in latent space, `z_basal + z_perturbation` |
| Decoder | MLP trunk `128 → 512 → 512 → 128`, then `h @ gene_embedding.T + gene_bias` |
| Adversary | MLP `128 → 512 → n_classes`, trained to recover the perturbation from the basal latent |

Scale is on the order of 10⁷ parameters — small enough to train on a single GPU.

## Data Processing

1. **Quality control** — drop cells with fewer than 200 genes detected or a
   mitochondrial fraction above 0.2; drop genes expressed in fewer than 3 cells.
2. **Normalisation** — library-size normalise to 10,000 counts, then `log1p`. Skipped if
   the matrix is already on a log scale.
3. **Feature selection** — retain the 2,000 most variable genes, **forcing perturbation
   target genes into the retained set** regardless of variance rank. Without this, a
   target gene that happens not to be highly variable is filtered out, and every
   perturbation aimed at it becomes unpredictable for reasons unrelated to the model.
4. **Label parsing** — condition labels follow the `TARGET+ctrl` convention, with
   `ctrl` / `control` / `nt` / `non-targeting` treated as control tokens.

## Training Strategy

The objective is reconstruction loss minus a weighted adversarial term:

```
loss = MSE(decode(encode(x) + p), x) − adv_weight · CE(adversary(encode(x)), perturbation)
```

The adversary is optimised separately and in opposition, so it needs its own optimiser.
Each step first lets the adversary improve at reading the perturbation off the basal
latent, then updates the encoder and decoder to reconstruct while defeating it.

| Setting | Value |
| --- | --- |
| Optimiser | Adam, lr 1e-3, weight decay 1e-6 |
| Adversarial weight | 0.5 |
| Batch size | 256 |
| Max epochs | 200 |
| Early stopping | patience 20 on a 10% validation slice of the training cells |
| Seeds | 0, 1, 2 |
| Expected runtime | 4–8 hours on one GPU for the full dataset |

## Evaluation Protocol

| Scope | Metrics |
| --- | --- |
| Genome-wide | MSE, PCC, R² |
| DE-restricted | MSE_DE, PCC_DE, R²_DE |

**The DE-restricted metrics are primary.** Genome-wide figures are dominated by the
unperturbed baseline profile, so a model that predicts "no change" scores well on them.

Differentially expressed genes are defined per perturbation by a Wilcoxon rank-sum test
against control cells with Benjamini–Hochberg correction, keeping genes with adjusted
p < 0.05 and |log2 fold change| > 0.5.

Scoring is **perturbation-centric**: each perturbation is scored once and the results are
averaged over perturbations, so that heavily sampled perturbations do not dominate. Five
folds, three seeds each. Two scenarios: `unseen_perturbation` and `unseen_context`.

Required comparisons: unperturbed control mean, linear regression, CPA, scGen, GEARS.

> One expected behaviour worth stating in advance, so it is not mistaken for a bug:
> **R²_DE can legitimately be negative.** The DE subset is selected for having moved, so
> it has low residual variance, and R² punishes that severely.

## Implementation

Deliverable: a single self-contained training script, [`../workspace/result.py`](../workspace/result.py).

**Acceptance criteria** — checked by the deterministic verifier, with no LLM involved:

- `python result.py --help` exits 0 **without third-party dependencies installed**
- `python result.py --selftest` exits 0
- `metrics.json` parses as a JSON object and contains `model`, `dataset`, `split`,
  `metrics`, `per_perturbation`, `de_definition`, `averaging`

To satisfy the first criterion, torch/numpy/anndata are imported lazily inside the
functions that need them.

**Prediction estimator.** The default is `delta`:

```
prediction = observed control mean + mean( decode(z + p) − decode(z) )
```

Decoding the same basal latent with and without the perturbation isolates the shift the
perturbation causes. Adding that shift to the *observed* control mean cancels systematic
reconstruction bias in `decode(encode(·))`, which would otherwise contaminate every gene
and swamp the comparatively small differential signal. A `direct` estimator, which
decodes the profile outright, is available for comparison.
