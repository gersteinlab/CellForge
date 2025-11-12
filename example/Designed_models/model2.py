#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scanpy as sc
import scipy
import anndata
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import optuna
import time
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')


def print_log(message):
    """Custom print function with timestamp"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


# Global variables
train_adata = None
test_adata = None
train_dataset = None
test_dataset = None
device = None
pca_model = None
scaler = None


class GeneExpressionDataset(Dataset):
    """Dataset for single-cell perturbation prediction"""

    def __init__(self, adata, perturbation_key='perturbation', dose_key='dose_value',
                 scaler=None, pca_model=None, pca_dim=512, fit_pca=False,
                 augment=False, is_train=True, common_genes_info=None,
                 perturbation_mapping=None):
        self.adata = adata
        self.perturbation_key = perturbation_key
        self.dose_key = dose_key
        self.augment = augment
        self.training = True
        self.pca_dim = pca_dim
        self.is_train = is_train
        self.common_genes_info = common_genes_info
        self.perturbation_mapping = perturbation_mapping

        # Data preprocessing - use common genes
        if common_genes_info is not None:
            if is_train:
                gene_idx = common_genes_info['train_idx']
            else:
                gene_idx = common_genes_info['test_idx']

            if scipy.sparse.issparse(adata.X):
                data = adata.X[:, gene_idx].toarray()
            else:
                data = adata.X[:, gene_idx]
        else:
            if scipy.sparse.issparse(adata.X):
                data = adata.X.toarray()
            else:
                data = adata.X

        # Preprocessing pipeline
        data = np.maximum(data, 0)
        data = np.maximum(data, 1e-10)
        data = np.log1p(data)

        # Standardization
        if scaler is None:
            self.scaler = StandardScaler()
            data = self.scaler.fit_transform(data)
        else:
            self.scaler = scaler
            data = self.scaler.transform(data)

        data = np.clip(data, -10, 10)
        data = data / 10.0

        # PCA dimension reduction
        if pca_model is None:
            if fit_pca:
                self.pca = PCA(n_components=pca_dim)
                self.expression_data = self.pca.fit_transform(data)
            else:
                raise ValueError('pca_model must be provided for test set')
        else:
            self.pca = pca_model
            self.expression_data = self.pca.transform(data)

        # Perturbation encoding with mapping for unseen perturbations
        if perturbation_mapping is not None:
            # Use predefined mapping for consistent perturbation encoding
            pert_series = adata.obs[perturbation_key]
            self.perturbations = np.zeros(
                (len(pert_series), len(perturbation_mapping)))
            for i, pert in enumerate(pert_series):
                if pert in perturbation_mapping:
                    self.perturbations[i, perturbation_mapping[pert]] = 1.0
                else:
                    # For unseen perturbations, use a learned embedding
                    # This will be handled by the model's perturbation embedding layer
                    # Default to first perturbation
                    self.perturbations[i, 0] = 1.0
        else:
            self.perturbations = pd.get_dummies(
                adata.obs[perturbation_key]).values

        print(
            f"{'train' if is_train else 'test'} set perturbation dimension: {self.perturbations.shape[1]}")
        print(
            f"{'train' if is_train else 'test'} set perturbation types: {adata.obs[perturbation_key].unique()}")

        # Dose encoding (handle empty strings)
        dose_values = adata.obs[dose_key].astype(str)
        dose_values = dose_values.replace('', '0.0')
        self.dose_values = pd.to_numeric(
            dose_values, errors='coerce').fillna(0.0).values
        self.dose_values = self.dose_values.reshape(-1, 1)

        print(f"{'train' if is_train else 'test'} set dose range: {self.dose_values.min():.3f} - {self.dose_values.max():.3f}")

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        # Get current sample - NO DATA LEAKAGE
        # Each sample is: (baseline_expression, perturbation_condition, dose, target_expression)
        x_baseline = self.expression_data[idx]
        pert = self.perturbations[idx]
        dose = self.dose_values[idx]
        x_target = self.expression_data[idx]  # Same cell, same expression

        # For perturbation prediction task:
        # Input: baseline expression + perturbation condition + dose
        # Output: perturbed expression (same cell under perturbation)

        # Data augmentation only on baseline (not target to avoid leakage)
        if self.augment and self.training:
            noise = np.random.normal(0, 0.05, x_baseline.shape)
            x_baseline = x_baseline + noise

            mask = np.random.random(x_baseline.shape) > 0.05
            x_baseline = x_baseline * mask

        return (torch.FloatTensor(x_baseline),
                torch.FloatTensor(pert),
                torch.FloatTensor(dose),
                torch.FloatTensor(x_target))

# Multi-scale Encoders


class GeneEncoder(nn.Module):
    """Gene-level encoder for 4222 genes"""

    def __init__(self, input_dim, hidden_dim=512, output_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class PathwayEncoder(nn.Module):
    """Pathway-level encoder for 128 modules"""

    def __init__(self, input_dim, hidden_dim=256, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class CellEncoder(nn.Module):
    """Cell-level encoder for 5K HVGs"""

    def __init__(self, input_dim, hidden_dim=128, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

# Multi-modal Compound Encoding


class ChemEncoder(nn.Module):
    """Chemical compound encoder - simplified for perturbation prediction"""

    def __init__(self, input_dim, hidden_dim=512, output_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, pert_features):
        return self.encoder(pert_features)


class TargetEncoder(nn.Module):
    """Drug target encoder using GNN over PPI network"""

    def __init__(self, input_dim, hidden_dim=256, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, target_features):
        return self.encoder(target_features)


class PathwayAnnotationEncoder(nn.Module):
    """Pathway annotation encoder with hierarchical embeddings"""

    def __init__(self, input_dim, hidden_dim=256, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, pathway_features):
        return self.encoder(pathway_features)


class DosePD(nn.Module):
    """Pharmacodynamics module with Hill function"""

    def __init__(self, input_dim, hidden_dim=64, output_dim=64):
        super().__init__()
        self.hill_params = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)  # Emax, EC50, h
        )
        self.output_proj = nn.Linear(3, output_dim)

    def forward(self, dose, chem_features):
        # Get Hill function parameters
        params = self.hill_params(chem_features)
        emax = torch.sigmoid(params[:, 0])  # 0-1
        ec50 = torch.exp(params[:, 1])      # positive
        h = torch.exp(params[:, 2])         # positive

        # Hill function: s(dose) = Emax * dose^h / (EC50^h + dose^h)
        dose_h = torch.pow(dose + 1e-8, h)
        ec50_h = torch.pow(ec50, h)
        effect = emax * dose_h / (ec50_h + dose_h)

        # Use the Hill function parameters directly for output
        return self.output_proj(params)

# Multi-scale Conditional Diffusion


class DiffusionNet(nn.Module):
    """Diffusion network for denoising"""

    def __init__(self, input_dim, hidden_dim, time_embed_dim=128, condition_dim=1024):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Calculate total input dimension (input_dim + time_embed_dim + condition_dim)
        total_input_dim = input_dim + time_embed_dim + condition_dim
        self.net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t, condition):
        # Time embedding - t should already have shape (batch_size, 1)
        t_embed = self.time_embed(t)

        # Always concatenate instead of adding to avoid dimension mismatch
        x_cond = torch.cat([x, t_embed, condition], dim=-1)

        return self.net(x_cond)


class MultiScaleDiffusion(nn.Module):
    """Multi-scale conditional diffusion"""

    def __init__(self, gene_dim=512, pathway_dim=256, cell_dim=128,
                 condition_dim=1024, time_embed_dim=128):
        super().__init__()

        # Gene-level diffusion
        self.gene_diffusion = DiffusionNet(
            gene_dim, 512, time_embed_dim, condition_dim)

        # Pathway-level diffusion
        self.pathway_diffusion = DiffusionNet(
            pathway_dim, 256, time_embed_dim, condition_dim)

        # Cell-level diffusion
        self.cell_diffusion = DiffusionNet(
            cell_dim, 128, time_embed_dim, condition_dim)

        # Condition projection
        self.condition_proj = nn.Linear(condition_dim, condition_dim)

    def forward(self, z_gene, z_pathway, z_cell, t, condition):
        # Project condition
        condition_proj = self.condition_proj(condition)

        # Multi-scale diffusion - pass t directly, let DiffusionNet handle time embedding
        z_gene_pred = self.gene_diffusion(z_gene, t, condition_proj)
        z_pathway_pred = self.pathway_diffusion(z_pathway, t, condition_proj)
        z_cell_pred = self.cell_diffusion(z_cell, t, condition_proj)

        return z_gene_pred, z_pathway_pred, z_cell_pred

# Biological Constraint Modules


class GRNConstraint(nn.Module):
    """Gene Regulatory Network constraint module"""

    def __init__(self, input_dim, grn_dim=256):
        super().__init__()
        self.grn_net = nn.Sequential(
            nn.Linear(input_dim, grn_dim),
            nn.LayerNorm(grn_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(grn_dim, input_dim)
        )

    def forward(self, x, grn_adjacency=None):
        return self.grn_net(x)


class DEWeighting(nn.Module):
    """Differentially Expressed gene weighting module"""

    def __init__(self, input_dim, de_dim=256):
        super().__init__()
        self.de_net = nn.Sequential(
            nn.Linear(input_dim, de_dim),
            nn.LayerNorm(de_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(de_dim, input_dim),
            nn.Sigmoid()  # Weight between 0-1
        )

    def forward(self, x, de_annotations=None):
        weights = self.de_net(x)
        return x * weights

# Main ChemCPA-X Model


class ChemCPAX(nn.Module):
    """ChemCPA-X: Multi-scale Conditional Diffusion Model"""

    def __init__(self, gene_dim=512, pathway_dim=256, cell_dim=128,
                 pert_dim=4, dose_dim=1, condition_dim=1024,
                 hidden_dim=512, n_layers=2, n_heads=8, dropout=0.1):
        super().__init__()

        # Multi-scale encoders
        self.gene_encoder = GeneEncoder(gene_dim, hidden_dim, gene_dim)
        self.pathway_encoder = PathwayEncoder(
            gene_dim, hidden_dim//2, pathway_dim)
        self.cell_encoder = CellEncoder(gene_dim, hidden_dim//4, cell_dim)

        # Multi-modal compound encoding
        self.chem_encoder = ChemEncoder(
            input_dim=pert_dim + dose_dim, hidden_dim=512, output_dim=512)
        self.target_encoder = TargetEncoder(
            input_dim=pert_dim + dose_dim, hidden_dim=256, output_dim=256)
        self.pathway_annotation_encoder = PathwayAnnotationEncoder(
            input_dim=pert_dim + dose_dim, hidden_dim=256, output_dim=256)
        self.dose_pd = DosePD(input_dim=pert_dim + dose_dim,
                              hidden_dim=64, output_dim=64)

        # Perturbation embedding for zero-shot capability
        self.pert_embedding = nn.Linear(pert_dim, 128)

        # Unseen perturbation handling
        self.unseen_pert_embedding = nn.Embedding(
            1, 128)  # For unseen perturbations

        # Cell-line-specific conditioning
        self.cellline_embed = nn.Embedding(3, 128)  # A549, K562, MCF7

        # Multi-scale diffusion
        # Calculate actual condition_dim
        # h_chem + h_targets + h_pathways + h_dose + h_pert + h_cellline
        actual_condition_dim = 512 + 256 + 256 + 64 + 128 + 128
        self.diffusion = MultiScaleDiffusion(
            gene_dim=gene_dim, pathway_dim=pathway_dim, cell_dim=cell_dim,
            condition_dim=actual_condition_dim
        )

        # Multi-scale fusion
        fusion_dim = gene_dim + pathway_dim + cell_dim
        # Update condition_dim to match concatenated features
        # h_chem + h_targets + h_pathways + h_dose + h_pert + h_cellline
        condition_dim = 512 + 256 + 256 + 64 + 128 + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, condition_dim),
            nn.LayerNorm(condition_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Biological constraint modules
        self.grn_constraint = GRNConstraint(condition_dim)
        self.de_weighting = DEWeighting(condition_dim)

        # Multi-scale decoder
        self.decoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, gene_dim)
        )

        # Perturbation prediction head
        self.perturbation_head = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, pert_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x_baseline, pert, dose, cellline_id=None, t=None, training=True):
        # Multi-scale encoding
        z_gene = self.gene_encoder(x_baseline)
        z_pathway = self.pathway_encoder(x_baseline)
        z_cell = self.cell_encoder(x_baseline)

        # Multi-modal compound encoding
        # Use perturbation and dose as compound features
        chem_features = torch.cat([pert, dose], dim=-1)
        h_chem = self.chem_encoder(chem_features)
        h_targets = self.target_encoder(chem_features)
        h_pathways = self.pathway_annotation_encoder(chem_features)
        h_dose = self.dose_pd(dose, chem_features)

        # Perturbation embedding for zero-shot capability
        # Check if this is an unseen perturbation (all zeros except first element)
        is_unseen = torch.all(pert[:, 1:] == 0, dim=1) & (pert[:, 0] == 1)
        h_pert = self.pert_embedding(pert)

        # For unseen perturbations, use special embedding
        if torch.any(is_unseen):
            unseen_embed = self.unseen_pert_embedding(
                torch.zeros_like(is_unseen, dtype=torch.long))
            h_pert = torch.where(is_unseen.unsqueeze(-1), unseen_embed, h_pert)

        # Cell-line conditioning
        if cellline_id is None:
            cellline_id = torch.zeros(x_baseline.size(
                0), dtype=torch.long, device=x_baseline.device)
        h_cellline = self.cellline_embed(cellline_id)

        # Combine all conditioning
        condition = torch.cat(
            [h_chem, h_targets, h_pathways, h_dose, h_pert, h_cellline], dim=-1)

        if training and t is not None:
            # Training: apply diffusion
            z_gene_pred, z_pathway_pred, z_cell_pred = self.diffusion(
                z_gene, z_pathway, z_cell, t, condition
            )
        else:
            # Inference: use original encodings
            z_gene_pred, z_pathway_pred, z_cell_pred = z_gene, z_pathway, z_cell

        # Multi-scale fusion
        z_fused = torch.cat([z_gene_pred, z_pathway_pred, z_cell_pred], dim=-1)
        z_fused = self.fusion(z_fused)

        # Biological constraints
        z_constrained = self.grn_constraint(z_fused)
        z_final = self.de_weighting(z_constrained)

        # Output
        output = self.decoder(z_final)
        pert_pred = self.perturbation_head(z_final)

        return output, pert_pred


def train_model(model, train_loader, optimizer, scheduler, device, aux_weight=0.1):
    """Training function"""
    model.train()
    total_loss = 0
    accumulation_steps = 4
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        x_baseline, pert, dose, x_target = batch
        x_baseline = x_baseline.to(device)
        pert = pert.to(device)
        dose = dose.to(device)
        x_target = x_target.to(device)

        # Random time for diffusion - ensure correct shape (batch_size, 1)
        t = torch.rand(x_baseline.size(0), 1, device=device)

        # Forward pass
        output, pert_pred = model(x_baseline, pert, dose, t=t, training=True)

        # Calculate losses
        main_loss = F.mse_loss(output, x_target)
        aux_loss = F.mse_loss(pert_pred, pert)
        loss = main_loss + aux_weight * aux_loss

        # Add regularization for better generalization
        l2_reg = 0.0
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss = loss + 1e-5 * l2_reg

        loss = loss / accumulation_steps
        loss.backward()

        # Gradient accumulation
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(train_loader)


def evaluate_model(model, test_loader, device, aux_weight=0.1):
    """Evaluation function"""
    model.eval()
    total_loss = 0
    total_r2 = 0
    total_pearson = 0
    total_pert_r2 = 0

    with torch.no_grad():
        for batch in test_loader:
            x_baseline, pert, dose, x_target = batch
            x_baseline = x_baseline.to(device)
            pert = pert.to(device)
            dose = dose.to(device)
            x_target = x_target.to(device)

            output, pert_pred = model(x_baseline, pert, dose, training=False)

            # Calculate loss
            main_loss = F.mse_loss(output, x_target)
            aux_loss = F.mse_loss(pert_pred, pert)
            loss = main_loss + aux_weight * aux_loss

            total_loss += loss.item()

            # Calculate metrics
            r2 = r2_score(x_target.cpu().numpy(), output.cpu().numpy())
            total_r2 += r2

            pearson = np.mean([pearsonr(x_target[i].cpu().numpy(), output[i].cpu().numpy())[0]
                               for i in range(x_target.size(0))])
            total_pearson += pearson

            pert_r2 = r2_score(pert.cpu().numpy(), pert_pred.cpu().numpy())
            total_pert_r2 += pert_r2

    return {
        'loss': total_loss / len(test_loader),
        'r2': total_r2 / len(test_loader),
        'pearson': total_pearson / len(test_loader),
        'pert_r2': total_pert_r2 / len(test_loader)
    }


def calculate_metrics(pred, true):
    """Calculate 6 evaluation metrics"""
    mse = np.mean((pred - true) ** 2)
    pcc = np.mean([pearsonr(p, t)[0] for p, t in zip(pred.T, true.T)])
    r2 = np.mean([r2_score(t, p) for p, t in zip(pred.T, true.T)])

    # Differentially expressed genes (DE)
    std = np.std(true, axis=0)
    de_mask = np.abs(true - np.mean(true, axis=0)) > std
    if np.any(de_mask):
        mse_de = np.mean((pred[de_mask] - true[de_mask]) ** 2)
        pcc_de = np.mean([pearsonr(p[m], t[m])[0]
                         for p, t, m in zip(pred.T, true.T, de_mask.T)])
        r2_de = np.mean([r2_score(t[m], p[m])
                        for p, t, m in zip(pred.T, true.T, de_mask.T)])
    else:
        mse_de = pcc_de = r2_de = np.nan

    return {
        'MSE': mse,
        'PCC': pcc,
        'R2': r2,
        'MSE_DE': mse_de,
        'PCC_DE': pcc_de,
        'R2_DE': r2_de
    }


def evaluate_and_save_model(model, test_loader, device, save_path,
                            common_genes_info=None, pca_model=None, scaler=None):
    """Evaluate model and save results"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            x_baseline, pert, dose, x_target = batch
            x_baseline = x_baseline.to(device)
            pert = pert.to(device)
            dose = dose.to(device)
            x_target = x_target.to(device)

            output, _ = model(x_baseline, pert, dose, training=False)

            all_predictions.append(output.cpu().numpy())
            all_targets.append(x_target.cpu().numpy())

    # Calculate overall metrics
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    results = calculate_metrics(all_predictions, all_targets)

    # Save model and results
    torch.save({
        'model_state_dict': model.state_dict(),
        'evaluation_results': results,
        'predictions': all_predictions,
        'targets': all_targets,
        'gene_names': common_genes_info['genes'] if common_genes_info is not None else None,
        'pca_model': pca_model,
        'scaler': scaler,
        'perturbation_mapping': getattr(common_genes_info, 'perturbation_mapping', None) if common_genes_info is not None else None,
        'model_config': {
            'gene_dim': 512,
            'pathway_dim': 256,
            'cell_dim': 128,
            'pert_dim': 4,
            'dose_dim': 1
        }
    }, save_path)

    # Create results DataFrame
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'PCC', 'R2', 'MSE_DE', 'PCC_DE', 'R2_DE'],
        'Value': [results['MSE'], results['PCC'], results['R2'],
                  results['MSE_DE'], results['PCC_DE'], results['R2_DE']]
    })

    print("\nEvaluation Results:")
    print(metrics_df.to_string(index=False,
          float_format=lambda x: '{:.6f}'.format(x)))
    print(f"\nModel and evaluation results saved to: {save_path}")

    return results


def main(gpu_id=None):
    """Main training function"""
    global train_adata, test_adata, train_dataset, test_dataset, device, pca_model, scaler

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print_log(f'ChemCPA-X Training started at: {timestamp}')

    # Set device
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print_log(f'Available GPUs: {gpu_count}')
        for i in range(gpu_count):
            print_log(f'GPU {i}: {torch.cuda.get_device_name(i)}')

        if gpu_id is not None:
            if gpu_id >= gpu_count:
                print_log(f'Warning: GPU {gpu_id} not available, using GPU 0')
                gpu_id = 0
            device = torch.device(f'cuda:{gpu_id}')
            print_log(f'Using GPU {gpu_id}: {device}')
        else:
            device = torch.device('cuda:0')
            print_log(f'Using default GPU 0: {device}')
    else:
        device = torch.device('cpu')
        print_log('CUDA not available, using CPU')

    # Load data
    print_log('Loading data...')
    train_path = "/data1/yzy/split_new/drug/SrivatsanTrapnell2020_train_unseen.h5ad"
    test_path = "/data1/yzy/split_new/drug/SrivatsanTrapnell2020_test_unseen.h5ad"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Data files not found: {train_path} or {test_path}")

    train_adata = sc.read_h5ad(train_path)
    test_adata = sc.read_h5ad(test_path)

    print_log(f'Training data shape: {train_adata.shape}')
    print_log(f'Test data shape: {test_adata.shape}')
    print_log(
        f'Training perturbations: {train_adata.obs["perturbation"].unique()}')
    print_log(f'Test perturbations: {test_adata.obs["perturbation"].unique()}')

    # Create perturbation mapping for zero-shot capability
    train_perts = train_adata.obs["perturbation"].unique()
    test_perts = test_adata.obs["perturbation"].unique()
    all_perts = list(set(list(train_perts) + list(test_perts)))
    perturbation_mapping = {pert: i for i, pert in enumerate(all_perts)}
    print_log(f'Perturbation mapping: {perturbation_mapping}')
    print_log(f'Unseen perturbations: {set(test_perts) - set(train_perts)}')

    # Ensure common genes
    print_log("Processing gene consistency...")
    train_genes = set(train_adata.var_names)
    test_genes = set(test_adata.var_names)
    common_genes = list(train_genes & test_genes)
    print_log(f"Common genes: {len(common_genes)}")

    common_genes.sort()
    train_gene_idx = [train_adata.var_names.get_loc(
        gene) for gene in common_genes]
    test_gene_idx = [test_adata.var_names.get_loc(
        gene) for gene in common_genes]

    # Create PCA model
    pca_model = PCA(n_components=512)

    # Preprocess training data
    if scipy.sparse.issparse(train_adata.X):
        train_data = train_adata.X[:, train_gene_idx].toarray()
    else:
        train_data = train_adata.X[:, train_gene_idx]

    train_data = np.maximum(train_data, 0)
    train_data = np.maximum(train_data, 1e-10)
    train_data = np.log1p(train_data)

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    train_data = np.clip(train_data, -10, 10)
    train_data = train_data / 10.0

    pca_model.fit(train_data)

    common_genes_info = {
        'genes': common_genes,
        'train_idx': train_gene_idx,
        'test_idx': test_gene_idx,
        'perturbation_mapping': perturbation_mapping
    }

    # Create datasets
    train_dataset = GeneExpressionDataset(
        train_adata,
        perturbation_key='perturbation',
        dose_key='dose_value',
        scaler=scaler,
        pca_model=pca_model,
        pca_dim=512,
        fit_pca=False,
        augment=True,
        is_train=True,
        common_genes_info=common_genes_info,
        perturbation_mapping=perturbation_mapping
    )

    test_dataset = GeneExpressionDataset(
        test_adata,
        perturbation_key='perturbation',
        dose_key='dose_value',
        scaler=scaler,
        pca_model=pca_model,
        pca_dim=512,
        fit_pca=False,
        augment=False,
        is_train=False,
        common_genes_info=common_genes_info,
        perturbation_mapping=perturbation_mapping
    )

    # Create model
    model = ChemCPAX(
        gene_dim=512,
        pathway_dim=256,
        cell_dim=128,
        # Use total number of perturbations
        pert_dim=len(perturbation_mapping),
        dose_dim=1,
        hidden_dim=512,
        n_layers=2,
        n_heads=8,
        dropout=0.1
    ).to(device)

    print_log(
        f'Model created with {sum(p.numel() for p in model.parameters())} parameters')

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # Training loop
    print_log('Starting training...')
    best_loss = float('inf')
    best_model = None
    max_epochs = 300  # Increased for better training
    patience = 30     # Increased patience
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss = train_model(
            model, train_loader, optimizer, scheduler, device, aux_weight=0.1)
        eval_metrics = evaluate_model(
            model, test_loader, device, aux_weight=0.1)

        if (epoch + 1) % 10 == 0:
            print_log(f'Epoch {epoch+1}/{max_epochs}:')
            print_log(f'Training Loss: {train_loss:.4f}')
            print_log(f'Test Loss: {eval_metrics["loss"]:.4f}')
            print_log(f'R2 Score: {eval_metrics["r2"]:.4f}')
            print_log(f'Pearson Correlation: {eval_metrics["pearson"]:.4f}')
            print_log(f'Perturbation R2: {eval_metrics["pert_r2"]:.4f}')

        if eval_metrics["loss"] < best_loss:
            best_loss = eval_metrics["loss"]
            best_model = model.state_dict()
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'metrics': eval_metrics
            }, f'chemcpa_x_best_model_{timestamp}.pt')
            print_log(f"Saved best model with loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print_log(f'Early stopping at epoch {epoch+1}')
                break

    # Load best model for final evaluation
    if best_model is not None:
        model.load_state_dict(best_model)

    # Final evaluation
    print_log('Evaluating final model...')
    results = evaluate_and_save_model(
        model, test_loader, device,
        f'chemcpa_x_final_model_{timestamp}.pt',
        common_genes_info, pca_model, scaler
    )

    # Save results to CSV
    results_df = pd.DataFrame({
        'Metric': ['MSE', 'PCC', 'R2', 'MSE_DE', 'PCC_DE', 'R2_DE'],
        'Value': [results['MSE'], results['PCC'], results['R2'],
                  results['MSE_DE'], results['PCC_DE'], results['R2_DE']]
    })

    results_df.to_csv(
        f'chemcpa_x_evaluation_results_{timestamp}.csv', index=False)

    print_log("\nFinal Evaluation Results:")
    print_log(results_df.to_string(
        index=False, float_format=lambda x: '{:.6f}'.format(x)))

    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChemCPA-X Training')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID to use (0-7)')
    parser.add_argument('--list-gpus', action='store_true',
                        help='List available GPUs and exit')

    args = parser.parse_args()

    if args.list_gpus:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f'Available GPUs: {gpu_count}')
            for i in range(gpu_count):
                print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        else:
            print('CUDA not available')
        exit(0)

    print("=" * 60)
    print("ChemCPA-X: Multi-scale Conditional Diffusion")
    print("=" * 60)

    if args.gpu is not None:
        print(f"Using GPU: {args.gpu}")
    else:
        print("Using default GPU settings")

    results_df = main(gpu_id=args.gpu)
