# Model cards

Six datasets went in. Six architectures came out. None of them were chosen from a
menu — each was specified by the Method Design agents during debate, then implemented
by the code-generation agent and trained end to end.

Descriptions are from the paper, [arXiv:2508.02276](https://arxiv.org/abs/2508.02276);
the complete generated training scripts are in its supplementary material.

> **These are outputs, not products.** CellForge does not ship weights, and re-running
> it on the same dataset will produce a *different* architecture. What is stable is the
> process and the quality band, not the artifact. Treat each card below as a worked
> example of what the system produces.

| Model | Dataset | Modality | Core idea |
|---|---|---|---|
| [CPA-X](#cpa-x) | Adamson | scRNA-seq | Disentangled perturbation embedding + adversarial covariate removal |
| [scGen-X](#scgen-x) | Norman | scRNA-seq | Latent arithmetic over gene *pairs* with a non-additive interaction term |
| [ChemCellFlow](#chemcellflow) | Srivatsan | scRNA-seq | Sinkhorn conditional OT + 6-layer normalizing flow |
| [CPA-Traj](#cpa-traj) | Schiebinger | scRNA-seq | Trajectory-aware VAE, time as a continuous covariate |
| [totalGAT](#totalgat) | Papalexi | CITE-seq | Graph attention + RNA↔protein cross-attention, dual heads |
| [ChromDDPM](#chromddpm) | Liscovitch-Brauer | scATAC-seq | Denoising diffusion over 200k-peak accessibility |

All six land in the 10–30M parameter range and train in 4–8 hours on a single GPU.

---

## CPA-X

**Adamson 2016 · CRISPRi in K562 · scRNA-seq · 111k cells / 33k genes**

An extension of the compositional perturbation autoencoder. Cells are encoded into a
latent space that is explicitly factorised: perturbation effect on one axis, cell-state
and technical covariates on another. An adversarial objective on the covariate branch
pushes batch and cell-context information *out* of the perturbation embedding, so
composing a perturbation onto an unseen cell context stays meaningful.

**Result: PCC 0.9883 — the best of every method compared on this dataset.**

Why the agents converged here: Adamson is single-gene CRISPRi with dense coverage per
perturbation. The hard part is not the perturbation model, it is not letting cell-state
variation leak into it. The design spends its capacity on disentanglement.

---

## scGen-X

**Norman 2019 · combinatorial CRISPRa · scRNA-seq · 84k cells / 17k genes**

scGen predicts a perturbation as a vector offset in latent space. That works for single
perturbations and breaks for combinations, because the effect of activating genes A and
B together is frequently *not* the sum of activating each alone.

scGen-X keeps the latent-arithmetic backbone and adds an explicit **interaction term**
for gene pairs, so the model can represent synergy and antagonism rather than being
forced into additivity.

**Result: MSE_DE 0.1736.**

This is the clearest case in the paper of the agents diagnosing a specific failure mode
of a known method and patching exactly that.

---

## ChemCellFlow

**Srivatsan 2020 · small-molecule perturbation · scRNA-seq · 81k cells / 18k genes**

Two pieces:

1. **Sinkhorn conditional optimal transport** — drug response is treated as transporting
   the unperturbed cell distribution to the perturbed one, conditioned on compound
   identity and dose. Entropic (Sinkhorn) regularisation makes the OT problem tractable
   at single-cell scale.
2. **A 6-layer normalizing flow** — an invertible density model on top, so the output is
   a full conditional distribution rather than a point estimate.

Drug response is dose-dependent and heterogeneous across cells; a conditional
distribution is the honest output type, and the flow provides it.

---

## CPA-Traj

**Schiebinger 2019 · cytokine stimulation time course · scRNA-seq · 65k cells / 17k genes**

A trajectory-aware VAE. The key decision: **time is conditioned on as a continuous
covariate, not a discrete class label.** Treating timepoints as categories throws away
the ordering and the intervals, and cannot interpolate to an unmeasured time. Continuous
conditioning keeps both.

**Result: DEG recall 0.535** — one of the weaker numbers in the paper, and reported as
such. Time-resolved response prediction is not solved.

> The trajectory-aware encoder **has no counterpart in the seed literature corpus.** It
> was not retrieved. It came out of the debate, and it is one of the two pieces of
> evidence in the paper that the Ideation stage does more than recombine what it read.

---

## totalGAT

**Papalexi 2021 · CRISPR · CITE-seq · 171k cells / 18k genes + 200 proteins**

The only genuinely multi-modal task: predict both the transcriptome and 200 surface
proteins from the same perturbation.

- **Graph attention** over a gene interaction network, so the perturbation effect
  propagates along known biology rather than through a dense layer.
- **Cross-attention between the RNA and protein branches**, letting each condition on
  the other — protein abundance is not a deterministic function of its transcript, and
  the model is allowed to learn that relationship.
- **Separate decoder heads per modality**, because RNA counts and protein counts have
  different noise models and different dynamic ranges. One shared head would force a
  bad compromise.

**Result: protein recall 0.420.** Surface-protein prediction is the hardest readout in
the benchmark.

---

## ChromDDPM

**Liscovitch-Brauer 2021 · CRISPR · scATAC-seq · 58k cells / 200k peaks**

A denoising diffusion probabilistic model over the chromatin accessibility profile,
conditioned on the perturbation. Evaluated with ROC-AUC and PR-AUC rather than
regression metrics, because accessibility is near-binary and extremely sparse.

The 200k-peak feature space is an order of magnitude wider than the transcriptomic
tasks. Iterative denoising handles that better than a single-shot decoder, which tends
to collapse toward the marginal accessibility profile.

> Like the CPA-Traj encoder, **the diffusion denoiser has no counterpart in the seed
> corpus.** Diffusion for perturbation-conditioned accessibility was not retrieved; it
> was proposed.

---

## Reading these critically

- **Sample size is one.** Each card is a single run on a single dataset. They are
  existence proofs that the system produces coherent, competitive designs — not
  evidence that these are the *best* designs for these datasets.
- **The names are the agents' own.** "CPA-X" and "scGen-X" are what the debate called
  them, and they signal honest lineage: these are extensions of published methods, not
  clean-sheet inventions. The two clean-sheet components are called out explicitly above.
- **Competitive is not always winning.** The paper reports competitive-or-better across
  the board with a clear win on Adamson. It does not claim state of the art on all six.

Full numbers: [RESULTS.md](RESULTS.md). How the designs are produced:
[ARCHITECTURE.md](ARCHITECTURE.md).
