#!/usr/bin/env python3
"""CPA-X: perturbation response prediction for Adamson et al. (2016) CRISPRi.

This is the training script that CellForge's Code Generation stage is expected to
produce for the Adamson task, written against the research plan in
``../plans/research_plan.md``.

Design, in one paragraph. A cell's expression profile is split into a *basal*
state and a *perturbation* effect. An encoder maps the observed profile to a
basal latent vector; an adversarial discriminator is trained to predict the
perturbation label from that latent, and the encoder is trained to defeat it, so
the basal latent is pushed toward being perturbation-free. The perturbation is
then re-introduced as a learned embedding added in latent space, and a decoder
maps the composed latent back to expression. Predicting an *unseen* perturbation
therefore reduces to composing a known basal state with a perturbation embedding
the model has never decoded before, which is what the unseen-perturbation split
actually asks for.

Dependencies are imported lazily so that ``--help`` and ``--selftest`` work in a
bare interpreter with nothing but the standard library. Only the paths that
actually train require torch/numpy.

Usage
-----
    python result.py --synthetic --epochs 20 --out ./run
    python result.py --data adamson.h5ad --split unseen_perturbation --out ./run

Outputs ``metrics.json`` in ``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "main",
    "build_parser",
    "benjamini_hochberg",
    "rank_sum_pvalues",
    "parse_perturbation",
]

DE_ADJ_P = 0.05
DE_LOG2FC = 0.5


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="result.py",
        description="CPA-X: predict single-cell transcriptional response to CRISPRi perturbation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--data", type=Path, default=None,
                      help="Path to the Adamson .h5ad file. Omit together with --synthetic.")
    data.add_argument("--synthetic", action="store_true",
                      help="Generate a small synthetic dataset instead of reading --data. "
                           "For smoke-testing the pipeline; the resulting metrics are not "
                           "comparable to published numbers.")
    data.add_argument("--perturbation-key", default="condition",
                      help="obs column holding the perturbation label.")
    data.add_argument("--control-label", default="ctrl",
                      help="Value in --perturbation-key marking unperturbed control cells.")
    data.add_argument("--max-cells", type=int, default=0,
                      help="Subsample to at most this many cells (0 = use all).")
    data.add_argument("--n-top-genes", type=int, default=2000,
                      help="Number of highly variable genes to retain (0 = use all).")

    split = parser.add_argument_group("split")
    split.add_argument("--split", choices=["unseen_perturbation", "unseen_context"],
                       default="unseen_perturbation",
                       help="Evaluation scenario. unseen_perturbation holds out whole "
                            "perturbations; unseen_context holds out cells.")
    split.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds.")
    split.add_argument("--fold", type=int, default=0, help="Which fold to evaluate (0-indexed).")

    model = parser.add_argument_group("model")
    model.add_argument("--latent-dim", type=int, default=128)
    model.add_argument("--hidden-dim", type=int, default=512)
    model.add_argument("--n-layers", type=int, default=3)
    model.add_argument("--dropout", type=float, default=0.1)
    model.add_argument("--adv-weight", type=float, default=0.5,
                       help="Weight on the adversarial disentanglement loss.")
    model.add_argument("--estimator", choices=["delta", "direct"], default="delta",
                       help="How to turn the decoder into a prediction. 'delta' adds the "
                            "model's predicted shift to the observed control mean and "
                            "cancels reconstruction bias; 'direct' decodes the profile "
                            "outright.")

    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-6)
    train.add_argument("--patience", type=int, default=20,
                       help="Early-stopping patience in epochs (0 disables).")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    out = parser.add_argument_group("output")
    out.add_argument("--out", type=Path, default=Path("./run"),
                     help="Directory for metrics.json and checkpoints.")
    out.add_argument("--save-checkpoint", action="store_true")
    out.add_argument("--quiet", action="store_true")

    parser.add_argument("--selftest", action="store_true",
                        help="Run the dependency-free unit checks and exit.")
    return parser


# --------------------------------------------------------------------------- #
# Statistics — implemented directly so the script does not depend on scipy
# --------------------------------------------------------------------------- #

def benjamini_hochberg(pvalues: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg adjusted p-values, order preserved.

    Pure Python so it can be exercised by --selftest without numpy.
    """
    n = len(pvalues)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = n - rank + 1  # 1-based rank of this p-value in ascending order
        value = min(prev, pvalues[idx] * n / i)
        adjusted[idx] = value
        prev = value
    return adjusted


def rank_sum_pvalues(group_a, group_b):
    """Two-sided Wilcoxon rank-sum p-value per column, normal approximation.

    ``group_a`` and ``group_b`` are 2-D arrays with matching column counts.
    Ties receive average ranks and the variance is tie-corrected, which matters
    here because single-cell matrices are mostly zeros.
    """
    import numpy as np

    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    n1, n2 = a.shape[0], b.shape[0]
    if n1 == 0 or n2 == 0:
        return np.ones(a.shape[1], dtype=np.float64)

    combined = np.vstack([a, b])
    n = n1 + n2

    # Average ranks per column.
    order = np.argsort(combined, axis=0, kind="mergesort")
    ranks = np.empty_like(combined, dtype=np.float64)
    col_idx = np.arange(combined.shape[1])
    ranks[order, col_idx] = np.arange(1, n + 1)[:, None]

    sorted_vals = np.take_along_axis(combined, order, axis=0)
    tie_correction = np.zeros(combined.shape[1], dtype=np.float64)
    for j in range(combined.shape[1]):
        col = sorted_vals[:, j]
        start = 0
        while start < n:
            stop = start + 1
            while stop < n and col[stop] == col[start]:
                stop += 1
            size = stop - start
            if size > 1:
                mean_rank = (start + stop + 1) / 2.0
                ranks[order[start:stop, j], j] = mean_rank
                tie_correction[j] += size ** 3 - size
            start = stop

    rank_sum_a = ranks[:n1, :].sum(axis=0)
    expected = n1 * (n + 1) / 2.0
    variance = n1 * n2 * ((n + 1) - tie_correction / (n * (n - 1))) / 12.0
    variance = np.maximum(variance, 1e-12)

    z = (rank_sum_a - expected) / np.sqrt(variance)
    # Two-sided normal tail via the error function.
    from math import erfc, sqrt
    return np.array([erfc(abs(v) / sqrt(2.0)) for v in z], dtype=np.float64)


def differential_genes(perturbed, control) -> "Any":
    """Boolean mask of differentially expressed genes.

    Wilcoxon rank-sum against control with Benjamini-Hochberg correction,
    keeping genes with adjusted p < 0.05 and |log2 fold change| > 0.5. This is
    the definition used throughout the CellForge benchmark.
    """
    import numpy as np

    pvals = rank_sum_pvalues(perturbed, control)
    adj = np.asarray(benjamini_hochberg(list(pvals)))
    # Inputs are log1p-normalised, so the mean difference is already a log ratio;
    # convert from natural log to log2.
    log2fc = (perturbed.mean(axis=0) - control.mean(axis=0)) / np.log(2.0)
    return (adj < DE_ADJ_P) & (np.abs(log2fc) > DE_LOG2FC)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _pearson(x, y) -> float:
    import numpy as np

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2:
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.sqrt((xc ** 2).sum()) * np.sqrt((yc ** 2).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((xc * yc).sum() / denom)


def _r2(true, pred) -> float:
    import numpy as np

    true = np.asarray(true, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def evaluate_perturbation(true_profile, pred_profile, de_mask) -> Dict[str, float]:
    """Metrics for a single perturbation, on the mean post-perturbation profile."""
    import numpy as np

    scores = {
        "mse": float(np.mean((true_profile - pred_profile) ** 2)),
        "pcc": _pearson(true_profile, pred_profile),
        "r2": _r2(true_profile, pred_profile),
    }
    if de_mask is not None and int(de_mask.sum()) > 1:
        t, p = true_profile[de_mask], pred_profile[de_mask]
        scores.update({
            "mse_de": float(np.mean((t - p) ** 2)),
            "pcc_de": _pearson(t, p),
            "r2_de": _r2(t, p),
            "n_de_genes": int(de_mask.sum()),
        })
    return scores


def aggregate(per_perturbation: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Perturbation-centric averaging: mean over perturbations, not over cells.

    Averaging over cells would let perturbations with many cells dominate; the
    benchmark averages each perturbation once.
    """
    import numpy as np

    keys: List[str] = []
    for scores in per_perturbation.values():
        for key in scores:
            if key not in keys and key != "n_de_genes":
                keys.append(key)

    summary: Dict[str, float] = {}
    for key in keys:
        values = [s[key] for s in per_perturbation.values()
                  if key in s and not np.isnan(s[key])]
        if values:
            summary[key] = float(np.mean(values))
    summary["n_perturbations"] = len(per_perturbation)
    return summary


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def make_synthetic(n_cells: int = 6000, n_genes: int = 400, n_perts: int = 40,
                   latent_k: int = 10, seed: int = 0):
    """Small synthetic dataset with a genuine, *generalisable* perturbation structure.

    Every gene g carries a hidden factor vector ``F[g]``. Two things derive from
    it, and the fact that they share it is the whole point:

    1. Co-expression. A cell's basal profile is ``baseline + s @ F.T`` for a
       random cell-level score vector ``s``, so genes with similar factors
       covary across cells.
    2. Perturbation response. Knocking down gene g shifts every gene in
       proportion to its factor-space similarity to g, ``-alpha * (F @ F[g])``,
       with an extra hit to g's own transcript because that is what CRISPRi
       does directly.

    Because both derive from ``F``, a model that learns gene representations
    from co-expression can extrapolate to a target gene it never saw perturbed.
    That is the assumption the CPA-X architecture encodes, so this synthetic set
    is a fair smoke test of the *pipeline* — it is not evidence that the
    assumption holds in real tissue, and the resulting numbers say nothing about
    real biological performance.

    An earlier revision of this generator drew the response from a matrix
    independent of the co-expression structure. That makes the
    unseen-perturbation split unlearnable in principle, and every model scores
    at chance. If you adapt this generator, keep the two coupled.

    Returns ``(X, labels, gene_names, control_label)``.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    gene_names = [f"GENE{i:04d}" for i in range(n_genes)]

    factors = rng.normal(0.0, 1.0, size=(n_genes, latent_k)).astype(np.float32)
    factors /= np.linalg.norm(factors, axis=1, keepdims=True) + 1e-8

    target_ids = rng.choice(n_genes, size=n_perts, replace=False)
    alpha = 2.0
    effects = {}
    for gene_id in target_ids:
        effect = -alpha * (factors @ factors[gene_id])
        effect[gene_id] -= 2.0  # direct knockdown of the target transcript
        effects[gene_names[gene_id]] = effect.astype(np.float32)

    pert_names = list(effects)
    labels = np.array(
        ["ctrl"] * (n_cells // 4)
        + [pert_names[i % n_perts] for i in range(n_cells - n_cells // 4)]
    )
    rng.shuffle(labels)

    baseline = rng.normal(4.0, 0.4, size=(1, n_genes)).astype(np.float32)
    scores = rng.normal(0.0, 1.0, size=(n_cells, latent_k)).astype(np.float32)
    X = baseline + scores @ factors.T
    X += rng.normal(0.0, 0.3, size=(n_cells, n_genes)).astype(np.float32)
    for name, effect in effects.items():
        X[labels == name] += effect

    X = np.log1p(np.maximum(X, 0.0)).astype(np.float32)
    return X, labels, gene_names, "ctrl"


def load_h5ad(path: Path, pert_key: str, n_top_genes: int, max_cells: int, seed: int):
    """Load and normalise a .h5ad file into a dense matrix plus labels."""
    import anndata as ad
    import numpy as np

    adata = ad.read_h5ad(path)
    if pert_key not in adata.obs:
        raise SystemExit(
            f"Column '{pert_key}' not found in obs. Available: {list(adata.obs.columns)}"
        )

    if max_cells and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        keep = rng.choice(adata.n_obs, size=max_cells, replace=False)
        adata = adata[np.sort(keep)].copy()

    X = adata.X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)

    # Library-size normalise to 1e4 then log1p, unless it already looks logged.
    if X.max() > 50:
        totals = X.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        X = np.log1p(X / totals * 1e4).astype(np.float32)

    gene_names = [str(g) for g in adata.var_names]
    labels = adata.obs[pert_key].astype(str).to_numpy()

    if n_top_genes and X.shape[1] > n_top_genes:
        variances = X.var(axis=0)
        keep = np.sort(np.argsort(variances)[::-1][:n_top_genes])
        # Never drop a gene that is itself a perturbation target: the model
        # conditions on the target's embedding, so dropping it would silently
        # make those perturbations unpredictable.
        targeted = {
            token.strip().upper()
            for label in set(labels)
            for token in str(label).replace(",", "+").split("+")
        }
        upper_names = [g.upper() for g in gene_names]
        forced = [i for i, g in enumerate(upper_names) if g in targeted]
        keep = np.union1d(keep, np.array(forced, dtype=int)) if forced else keep
        X = X[:, keep]
        gene_names = [gene_names[i] for i in keep]

    return X, labels, gene_names


def make_split(labels, control_label: str, mode: str, folds: int, fold: int, seed: int):
    """Return boolean train/test masks for the requested scenario."""
    import numpy as np

    rng = np.random.default_rng(seed)
    perts = sorted({p for p in labels if p != control_label})
    if not perts:
        raise SystemExit("No perturbed cells found — check --perturbation-key/--control-label.")
    folds = max(1, min(folds, len(perts)))
    fold = fold % folds

    if mode == "unseen_perturbation":
        shuffled = list(perts)
        rng.shuffle(shuffled)
        held_out = set(shuffled[fold::folds])
        test = np.isin(labels, list(held_out))
    else:  # unseen_context — hold out cells, every perturbation stays visible
        assignment = rng.integers(0, folds, size=len(labels))
        test = (assignment == fold) & (labels != control_label)
        held_out = set(perts)

    train = ~test
    return train, test, sorted(held_out)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def build_model(n_genes: int, n_classes: int, args):
    """CPA-X.

    The one design decision that matters here: the perturbation is *not* a free
    per-perturbation lookup embedding. It is a learned function of the target
    gene's own embedding, and that gene embedding is tied to the decoder's
    output weights. Every gene therefore receives gradient from every
    reconstruction, whether or not it was ever perturbed in training. A held-out
    target gene consequently has a trained representation at test time, which is
    what makes the unseen-perturbation split solvable. With a plain
    ``nn.Embedding(n_perturbations, d)`` the held-out rows would still hold
    their random initialisation and the model would be predicting noise.
    """
    import torch
    import torch.nn as nn

    def mlp(sizes, dropout):
        layers: List[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.extend([nn.LayerNorm(sizes[i + 1]), nn.ReLU(), nn.Dropout(dropout)])
        return nn.Sequential(*layers)

    hidden = [args.hidden_dim] * max(1, args.n_layers - 1)
    d = args.latent_dim

    class CPAX(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = mlp([n_genes, *hidden, d], args.dropout)
            self.decoder_trunk = mlp([d, *hidden, d], args.dropout)

            # Shared gene embedding: conditions the perturbation *and* decodes.
            self.gene_embedding = nn.Parameter(torch.randn(n_genes, d) * 0.02)
            self.gene_bias = nn.Parameter(torch.zeros(n_genes))

            self.pert_encoder = mlp([d, args.hidden_dim, d], args.dropout)
            self.adversary = mlp([d, args.hidden_dim, n_classes], args.dropout)

        def basal(self, x):
            return self.encoder(x)

        def perturbation_latent(self, pert_gene_ids, pert_gene_mask):
            """Sum the encoded target-gene embeddings; zero for controls.

            ``pert_gene_ids`` is (batch, max_targets) and ``pert_gene_mask``
            marks which slots are real, so single and combinatorial
            perturbations share one code path.
            """
            embeddings = self.gene_embedding[pert_gene_ids]        # (B, T, d)
            encoded = self.pert_encoder(embeddings)                # (B, T, d)
            return (encoded * pert_gene_mask.unsqueeze(-1)).sum(dim=1)

        def decode(self, latent):
            h = self.decoder_trunk(latent)
            return h @ self.gene_embedding.t() + self.gene_bias

        def forward(self, x, pert_gene_ids, pert_gene_mask):
            z = self.encoder(x)
            p = self.perturbation_latent(pert_gene_ids, pert_gene_mask)
            return self.decode(z + p), z

    return CPAX()


def parse_perturbation(label: str, control_label: str, gene_to_idx: Dict[str, int]) -> List[int]:
    """Map a condition label to the gene indices it targets.

    Handles the ``"AARS+ctrl"`` / ``"CBL+CNN1"`` convention used by the
    benchmark datasets, so single and combinatorial perturbations both work.
    Unrecognised target names yield no indices, i.e. an unperturbed prediction.
    """
    if label == control_label:
        return []
    targets = []
    for token in str(label).replace(",", "+").split("+"):
        token = token.strip()
        if not token or token.lower() in {"ctrl", "control", "nt", "non-targeting"}:
            continue
        idx = gene_to_idx.get(token, gene_to_idx.get(token.upper()))
        if idx is not None:
            targets.append(idx)
    return targets


def resolve_device(choice: str) -> str:
    import torch

    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.", file=sys.stderr)
        return "cpu"
    return choice


def encode_perturbations(labels, control_label, gene_to_idx, max_targets: int = 2):
    """Vectorise condition labels into padded target-gene id/mask arrays."""
    import numpy as np

    n = len(labels)
    ids = np.zeros((n, max_targets), dtype=np.int64)
    mask = np.zeros((n, max_targets), dtype=np.float32)
    for i, label in enumerate(labels):
        targets = parse_perturbation(label, control_label, gene_to_idx)[:max_targets]
        for slot, gene_id in enumerate(targets):
            ids[i, slot] = gene_id
            mask[i, slot] = 1.0
    return ids, mask


def train_model(X, labels, gene_names, train_mask, control_label, args, log):
    import numpy as np
    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gene_to_idx = {name: i for i, name in enumerate(gene_names)}
    class_names = [control_label] + sorted({p for p in labels[train_mask] if p != control_label})
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    device = resolve_device(args.device)
    model = build_model(X.shape[1], len(class_names), args).to(device)

    # The adversary is optimised against the encoder, so it gets its own optimiser.
    main_params = [p for n, p in model.named_parameters() if not n.startswith("adversary")]
    opt = torch.optim.Adam(main_params, lr=args.lr, weight_decay=args.weight_decay)
    opt_adv = torch.optim.Adam(model.adversary.parameters(), lr=args.lr)

    idx_all = np.where(train_mask)[0]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx_all)
    n_val = max(1, int(0.1 * len(idx_all)))          # 10% held out for early stopping
    val_idx, fit_idx = idx_all[:n_val], idx_all[n_val:]

    pert_ids, pert_mask = encode_perturbations(labels, control_label, gene_to_idx)
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    ids_t = torch.as_tensor(pert_ids, device=device)
    mask_t = torch.as_tensor(pert_mask, device=device)
    y_t = torch.as_tensor(
        np.array([class_to_idx.get(p, 0) for p in labels]), dtype=torch.long, device=device
    )

    recon_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    best_val, best_state, stale = float("inf"), None, 0
    fit_t = torch.as_tensor(fit_idx, dtype=torch.long, device=device)
    val_t = torch.as_tensor(val_idx, dtype=torch.long, device=device)

    for epoch in range(args.epochs):
        model.train()
        batch_source = fit_t[torch.randperm(len(fit_t), device=device)]
        total = 0.0

        for start in range(0, len(batch_source), args.batch_size):
            batch = batch_source[start:start + args.batch_size]
            xb, yb = X_t[batch], y_t[batch]
            ib, mb = ids_t[batch], mask_t[batch]

            # 1. Adversary learns to read the perturbation off the basal latent.
            with torch.no_grad():
                z_detached = model.basal(xb)
            opt_adv.zero_grad(set_to_none=True)
            ce_loss(model.adversary(z_detached), yb).backward()
            opt_adv.step()

            # 2. Encoder/decoder reconstruct while defeating the adversary.
            opt.zero_grad(set_to_none=True)
            pred, z = model(xb, ib, mb)
            loss = recon_loss(pred, xb) - args.adv_weight * ce_loss(model.adversary(z), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(batch)

        model.eval()
        with torch.no_grad():
            pred, _ = model(X_t[val_t], ids_t[val_t], mask_t[val_t])
            val = float(recon_loss(pred, X_t[val_t]).item())

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            log(f"epoch {epoch:4d}  train {total / max(1, len(fit_idx)):.4f}  val {val:.4f}")

        if val < best_val - 1e-5:
            best_val, stale = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                log(f"early stop at epoch {epoch} (best val {best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, gene_to_idx, device


def predict_and_score(model, X, labels, train_mask, test_mask, gene_to_idx,
                      control_label, device, estimator: str = "delta") -> Dict[str, Dict[str, float]]:
    """Predict each held-out perturbation from control cells and score it.

    The prediction for a perturbation is the mean profile obtained by applying
    that perturbation to every *training* control cell. Comparing it against the
    mean observed profile of the held-out perturbed cells matches the
    perturbation-centric protocol the benchmark uses.
    """
    import numpy as np
    import torch

    model.eval()
    control_idx = np.where((labels == control_label) & train_mask)[0]
    if control_idx.size == 0:
        raise SystemExit("No control cells in the training split; cannot predict.")

    X_ctrl = torch.as_tensor(X[control_idx], dtype=torch.float32, device=device)
    control_matrix = X[control_idx]

    results: Dict[str, Dict[str, float]] = {}
    for pert in sorted({p for p in labels[test_mask] if p != control_label}):
        true_cells = X[test_mask & (labels == pert)]
        if true_cells.shape[0] < 3:
            continue

        ids, mask = encode_perturbations(
            np.array([pert] * X_ctrl.shape[0]), control_label, gene_to_idx
        )
        if mask.sum() == 0:
            # Target gene is not in the measured gene set: nothing to condition on.
            continue

        ids_t = torch.as_tensor(ids, device=device)
        mask_t = torch.as_tensor(mask, device=device)
        with torch.no_grad():
            perturbed, _ = model(X_ctrl, ids_t, mask_t)
            if estimator == "direct":
                pred_profile = perturbed.mean(dim=0).cpu().numpy()
            else:
                # Delta estimator. Decoding the same basal latent with and
                # without the perturbation isolates the shift the perturbation
                # causes; adding it to the *observed* control mean cancels any
                # systematic reconstruction bias in decode(encode(.)), which
                # would otherwise contaminate every gene equally and swamp the
                # comparatively small differential signal.
                baseline, _ = model(X_ctrl, torch.zeros_like(ids_t),
                                    torch.zeros_like(mask_t))
                shift = (perturbed - baseline).mean(dim=0).cpu().numpy()
                pred_profile = control_matrix.mean(axis=0) + shift

        de_mask = differential_genes(true_cells, control_matrix)
        results[pert] = evaluate_perturbation(true_cells.mean(axis=0), pred_profile, de_mask)

    return results


# --------------------------------------------------------------------------- #
# Self-test — no third-party dependencies
# --------------------------------------------------------------------------- #

def _selftest() -> int:
    ok = True

    adj = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
    expected = [0.05, 0.05, 0.05, 0.05, 0.05]
    if not all(abs(a - e) < 1e-9 for a, e in zip(adj, expected)):
        print(f"FAIL benjamini_hochberg: {adj} != {expected}")
        ok = False

    if benjamini_hochberg([]) != []:
        print("FAIL benjamini_hochberg on empty input")
        ok = False

    single = benjamini_hochberg([0.004])
    if abs(single[0] - 0.004) > 1e-12:
        print(f"FAIL benjamini_hochberg single: {single}")
        ok = False

    # Monotonicity: adjusted p-values must not decrease as raw p-values increase.
    raw = [0.001, 0.008, 0.02, 0.2, 0.9]
    adj = benjamini_hochberg(raw)
    if any(adj[i] > adj[i + 1] + 1e-12 for i in range(len(adj) - 1)):
        print(f"FAIL benjamini_hochberg monotonicity: {adj}")
        ok = False

    genes = {"AARS": 0, "CBL": 1, "CNN1": 2}
    cases = [
        ("ctrl", []),
        ("AARS+ctrl", [0]),
        ("CBL+CNN1", [1, 2]),
        ("AARS", [0]),
        ("aars", [0]),              # case-insensitive fallback
        ("UNKNOWN+ctrl", []),       # target absent from the measured gene set
        ("CBL,CNN1", [1, 2]),       # comma-separated variant
    ]
    for label, expected in cases:
        got = parse_perturbation(label, "ctrl", genes)
        if got != expected:
            print(f"FAIL parse_perturbation({label!r}): {got} != {expected}")
            ok = False

    parser = build_parser()
    args = parser.parse_args(["--synthetic", "--epochs", "1"])
    if not args.synthetic or args.epochs != 1:
        print("FAIL argument parsing")
        ok = False

    print("selftest: OK" if ok else "selftest: FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.selftest:
        return _selftest()

    if not args.synthetic and args.data is None:
        build_parser().error("provide --data PATH or --synthetic")
    if args.data is not None and not args.data.exists():
        build_parser().error(f"--data path does not exist: {args.data}")

    def log(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    import numpy as np

    if args.synthetic:
        log("Generating synthetic data (metrics will not match published numbers).")
        X, labels, gene_names, control_label = make_synthetic(seed=args.seed)
    else:
        log(f"Loading {args.data}")
        X, labels, gene_names = load_h5ad(args.data, args.perturbation_key,
                                          args.n_top_genes, args.max_cells, args.seed)
        control_label = args.control_label

    log(f"Matrix: {X.shape[0]} cells x {X.shape[1]} genes, "
        f"{len(set(labels))} conditions (control='{control_label}')")

    train_mask, test_mask, held_out = make_split(
        labels, control_label, args.split, args.folds, args.fold, args.seed
    )
    log(f"Split '{args.split}' fold {args.fold}: {int(train_mask.sum())} train / "
        f"{int(test_mask.sum())} test cells, {len(held_out)} held-out perturbations")

    model, gene_to_idx, device = train_model(
        X, labels, gene_names, train_mask, control_label, args, log
    )
    log(f"Training finished on {device}; scoring held-out perturbations.")

    per_pert = predict_and_score(model, X, labels, train_mask, test_mask,
                                 gene_to_idx, control_label, device, args.estimator)
    if not per_pert:
        raise SystemExit("No held-out perturbation had enough cells to score.")

    summary = aggregate(per_pert)

    args.out.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model": "CPA-X",
        "dataset": "adamson_crispri" if not args.synthetic else "synthetic",
        "split": args.split,
        "fold": args.fold,
        "seed": args.seed,
        "synthetic": bool(args.synthetic),
        "n_cells": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
        "de_definition": {
            "test": "wilcoxon_rank_sum",
            "correction": "benjamini_hochberg",
            "adjusted_p_threshold": DE_ADJ_P,
            "abs_log2fc_threshold": DE_LOG2FC,
        },
        "averaging": "perturbation_centric",
        "metrics": summary,
        "per_perturbation": per_pert,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }

    metrics_path = args.out / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"Wrote {metrics_path}")

    if args.save_checkpoint:
        import torch
        ckpt = args.out / "model.pt"
        torch.save({"state_dict": model.state_dict(), "pert_to_idx": pert_to_idx}, ckpt)
        log(f"Wrote {ckpt}")

    headline = "  ".join(
        f"{k}={summary[k]:.4f}" for k in ("mse", "pcc", "r2", "mse_de", "pcc_de")
        if k in summary
    )
    log(f"[{args.split}] {headline}  over {summary['n_perturbations']} perturbations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
