> [!IMPORTANT]
> **This is a reference artifact, not the transcript of a live run.**
>
> It was written by hand to match, section for section, the markdown that
> `TaskAnalysisReport.to_markdown()` in
> [`cellforge/Task_Analysis/data_structures.py`](../../../../cellforge/Task_Analysis/data_structures.py)
> emits, and its content is grounded in the Adamson case reported in
> [arXiv:2508.02276](https://arxiv.org/abs/2508.02276). It is here so you can
> see the shape, depth, and register of a Task Analysis report before spending
> tokens on one.
>
> A real run's wording, ordering, and confidence scores will differ — the agents
> are stochastic. See [`../PROVENANCE.md`](../PROVENANCE.md) for the provenance
> of every file in this directory.

# Task Analysis Report

Generated on: (reference artifact — a real run stamps the wall-clock time here)

## 1. Dataset Analysis

### Experimental Design & Scale

**Source**: Adamson et al. 2016, *Cell* 167(7):1867-1882. GEO accession GSE90546.

**Assay**: Perturb-seq — pooled CRISPRi with direct-capture sgRNA readout, so the
perturbation identity of each cell is recovered from the same library as its
transcriptome rather than inferred.

**Cell line**: K562, a chronic myelogenous leukaemia line. Effectively clonal, which
removes donor and tissue heterogeneity as a confounder and makes this the least noisy
entry point of the six benchmark datasets.

**Scale**: approximately 111,000 cells across roughly 33,000 measured genes.

**Perturbation class**: transcriptional knockdown via dCas9-KRAB. Note that this is
knockdown, not knockout — residual target expression is expected and the effect size is
graded rather than binary.

### Data Characteristics

**Modality**: single-cell RNA-seq, UMI counts.

**Sparsity**: high. The overwhelming majority of gene-by-cell entries are zero, so
per-cell profiles are individually unreliable and most quantities of interest are only
estimable at the level of a perturbation group.

**Count distribution**: over-dispersed relative to Poisson. Library size varies
substantially between cells and must be normalised before profiles are comparable.

**Class balance**: the number of cells per perturbation is uneven. Any metric averaged
over cells will silently become a metric about the most heavily sampled perturbations.

**Controls**: non-targeting sgRNA cells are present and are the reference against which
differential expression is defined.

### Preprocessing Considerations

**Normalisation**: library-size normalise to a fixed count depth, then log1p. The
downstream metrics (MSE, PCC, R2) all assume an approximately homoscedastic scale, which
raw counts do not provide.

**Feature selection**: retaining highly variable genes reduces the feature space by an
order of magnitude at little cost in signal. One caveat that is easy to get wrong: a
model conditioning on the target gene's own representation must not have that gene
filtered out of the matrix, or the perturbations whose targets were dropped become
unpredictable for reasons that have nothing to do with the model.

**Quality control**: filter cells on minimum genes detected and maximum mitochondrial
fraction; filter genes on minimum cells expressing.

**Perturbation labels**: parse the condition field into target gene symbols. The
benchmark convention writes a single perturbation as `TARGET+ctrl`, so a naive string
comparison against the gene list will fail to match.

### Quality Assessment

**Strengths**: direct sgRNA capture removes assignment ambiguity; a clonal line removes
inter-individual variation; the dataset is large enough that per-perturbation means are
well estimated.

**Limitations**: single cell line, so nothing about cross-tissue generalisation can be
concluded here. Knockdown efficiency varies by target and is not directly observed,
which puts an unknown ceiling on achievable accuracy for some perturbations.

**Suitability**: high. This is the appropriate first task for a new architecture, and a
model that cannot do well here will not do well on the harder datasets.

## 2. Problem Investigation

### Formal Definition

**Given**: a baseline expression profile `x_c` for an unperturbed cell `c`, and a target
gene `g` knocked down by CRISPRi.

**Predict**: the post-perturbation profile `x_hat` over all measured genes.

**Learn**: a map `f(x_c, g) -> x_hat` from observed (profile, perturbation) pairs.

**Generalisation demanded**: the map must be evaluated on target genes `g` that never
appeared in training, which is a different and much harder requirement than
interpolating within a seen perturbation set.

### Key Challenges

**Unseen-perturbation extrapolation**: this is the crux of the task. A model that
represents each perturbation as a free row in a lookup table cannot generalise at all —
a held-out row keeps its random initialisation. The perturbation representation must be
a function of features of the target gene that are themselves learnable from the
training data.

**Cell-state confounding**: a cell's basal state and the perturbation applied to it are
entangled in the observed profile. Composing a perturbation onto a cell context never
seen paired with it requires that the two be separated first.

**Signal-to-noise**: the perturbation effect is small relative to both technical noise
and basal biological variation. Genome-wide metrics are dominated by the unperturbed
baseline, so a model that predicts "no change" scores deceptively well.

**Metric selection**: consequently, the differentially-expressed-gene-restricted metrics
carry most of the discriminative information, and the genome-wide figures should be read
as sanity checks rather than as the result.

### Research Questions

- What representation of a perturbation supports extrapolation to unseen target genes?
- Does explicit disentanglement of basal state from perturbation effect measurably
  improve unseen-perturbation performance over an unconstrained encoder?
- How much of the achievable accuracy comes from the model versus from simply predicting
  the control mean?
- Does the ranking of models change between the genome-wide and DE-restricted metrics?

### Analysis Methods

**Differential expression**: Wilcoxon rank-sum against control cells with
Benjamini-Hochberg correction; retain genes with adjusted p < 0.05 and
|log2 fold change| > 0.5.

**Splitting**: hold out whole perturbations for the unseen-perturbation scenario; hold
out cells for the unseen-context scenario. Five folds.

**Aggregation**: perturbation-centric — score each perturbation once, then average over
perturbations. This is not the same as averaging over cells and gives materially
different numbers.

**Repetition**: three runs per fold with different seeds; report mean.

## 3. Baseline Assessment

### Baseline Models Analysis

**Unperturbed control**: predict the control mean, ignoring the perturbation. The
essential sanity baseline — anything that fails to beat it has learned nothing about
perturbation.

**Linear regression / random forest**: capture main effects without composition. Cheap
and surprisingly hard to beat genome-wide.

**CPA**: compositional autoencoder with adversarial removal of the perturbation from the
latent. The direct methodological ancestor of what is proposed here.

**scGen**: latent vector arithmetic over a VAE. Strong when the perturbation is seen in
training, structurally weak for unseen perturbations.

**GEARS**: uses a gene-gene relational graph to propagate perturbation effects to unseen
targets. The reference point for the extrapolation question specifically.

**scGPT**: pretrained transformer, fine-tuned. Tests whether large-scale pretraining
substitutes for task-specific structure.

### Evaluation Framework

**Genome-wide**: MSE, PCC, R2.

**DE-restricted**: MSE_DE, PCC_DE, R2_DE over the differentially expressed gene set.

**Scenarios**: unseen perturbations, unseen cell contexts.

**Protocol**: five folds, three seeds, perturbation-centric averaging.

### Performance Analysis

**Expected ordering**: methods with a structured perturbation representation should lead
on the unseen-perturbation split, while the gap narrows on unseen contexts where a
lookup embedding remains valid.

**Metric behaviour to anticipate**: genome-wide PCC will be high for every method
including trivial ones, because the baseline profile dominates the correlation. R2
restricted to DE genes can and often does go negative, since that subset is selected for
having moved and has low residual variance. Negative R2_DE is informative, not a bug.

### Improvement Suggestions

- Derive the perturbation representation from the target gene's own learned embedding
  rather than a free lookup table.
- Tie that gene embedding to the decoder's output weights so every gene receives
  gradient from every reconstruction, including genes never perturbed in training.
- Apply adversarial pressure to the basal latent so that composition onto novel contexts
  is meaningful.
- Report the control-mean baseline alongside every result.

## 4. Refinement Process

### Round 1

#### Dataset Analysis

**Concerns**:
- Highly-variable-gene filtering as originally specified could remove genes that are
  themselves perturbation targets.

**Resolutions**:
- Force target genes into the retained feature set regardless of variance rank.

#### Problem Investigation

**Concerns**:
- The original formulation did not distinguish the unseen-perturbation requirement from
  ordinary held-out evaluation, which understates the difficulty.

**Resolutions**:
- State extrapolation to unseen target genes as the primary generalisation demand and
  make it the headline scenario.

#### Baseline Assessment

**Concerns**:
- Genome-wide metrics alone would rank a no-change predictor competitively.

**Resolutions**:
- Elevate the DE-restricted metrics to primary and retain the control-mean baseline as a
  mandatory comparison.

## 5. Final Recommendations

### Data Processing Pipeline

**Steps**:
- Quality-control filtering on cells and genes.
- Library-size normalisation to a fixed depth followed by log1p.
- Highly-variable-gene selection, with perturbation target genes forced in.
- Condition-label parsing into target gene indices, handling the `TARGET+ctrl` form.

### Model Architecture

**Family**: compositional autoencoder with adversarial disentanglement (CPA-X).

**Encoder**: expression profile to a basal latent vector.

**Perturbation representation**: a learned function of the target gene's embedding,
where that embedding is shared with the decoder output layer.

**Composition**: additive in latent space.

**Decoder**: composed latent back to expression, through the tied gene embedding.

**Adversary**: a classifier trained to recover the perturbation from the basal latent,
with the encoder trained against it.

### Training Strategy

**Objective**: reconstruction loss minus a weighted adversarial term.

**Optimisation**: two optimisers — one for the adversary, one for everything else.

**Regularisation**: dropout, weight decay, early stopping on a held-out slice of the
training cells.

### Evaluation Protocol

**Scenarios**: unseen perturbations and unseen cell contexts.

**Metrics**: MSE, PCC, R2 genome-wide and restricted to DE genes.

**Aggregation**: perturbation-centric, five folds, three seeds.

**Mandatory comparison**: unperturbed-control baseline.

### Implementation Roadmap

**Deliverable**: a single self-contained training script exposing a command-line
interface, writing a machine-readable metrics file.

**Requirements**: deterministic under a fixed seed; runnable on one GPU; able to run
without a downloaded dataset for smoke-testing.
