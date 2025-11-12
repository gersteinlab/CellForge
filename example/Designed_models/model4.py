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
from IPython.display import display
import time
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')


def print_log(message):
    """Custom print function to simulate notebook output style"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


# Global variables
train_adata = None
test_adata = None
train_dataset = None
test_dataset = None
device = None
pca_model = None
scaler = None


class CITERNADataset(Dataset):
    def __init__(self, adata, perturbation_key='perturbation', scaler=None, pca_model=None,
                 pca_dim=128, fit_pca=False, augment=False, is_train=True, common_genes_info=None):
        self.adata = adata
        self.perturbation_key = perturbation_key
        self.augment = augment
        self.training = True
        self.pca_dim = pca_dim
        self.is_train = is_train
        self.common_genes_info = common_genes_info

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

        # Handle sparse data
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

        # PCA dimensionality reduction
        if pca_model is None:
            if fit_pca:
                self.pca = PCA(n_components=pca_dim)
                self.expression_data = self.pca.fit_transform(data)
            else:
                raise ValueError('pca_model must be provided for test set')
        else:
            self.pca = pca_model
            self.expression_data = self.pca.transform(data)

        # Perturbation encoding
        self.perturbations = pd.get_dummies(adata.obs[perturbation_key]).values
        print(
            f"{'train' if is_train else 'test'} set perturbation dimension: {self.perturbations.shape[1]}")

        # Create baseline-perturbation pairing data
        self._create_baseline_perturbation_pairs()

    def _create_baseline_perturbation_pairs(self):
        """Create baseline and perturbed expression pairs to prevent data leakage"""
        # Use control samples as baselines
        control_mask = self.adata.obs[self.perturbation_key] == 'control'
        control_indices = np.where(control_mask)[0]

        # Collect perturbed samples
        perturbed_mask = ~control_mask
        perturbed_indices = np.where(perturbed_mask)[0]

        print(f"Control samples: {len(control_indices)}")
        print(f"Perturbed samples: {len(perturbed_indices)}")

        # Pair each perturbed sample with a random control baseline
        self.pairs = []
        for i, perturbed_idx in enumerate(perturbed_indices):
            # Randomly select a control sample as the baseline
            baseline_idx = np.random.choice(control_indices)
            self.pairs.append({
                'baseline_idx': baseline_idx,
                'perturbed_idx': perturbed_idx,
                'perturbation': self.perturbations[perturbed_idx]
            })

        # Create self-pairs for control samples (baseline equals perturbed)
        for control_idx in control_indices:
            self.pairs.append({
                'baseline_idx': control_idx,
                'perturbed_idx': control_idx,
                'perturbation': self.perturbations[control_idx]
            })

        print(f"Total pairs created: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Retrieve baseline and perturbed expressions
        baseline_expr = self.expression_data[pair['baseline_idx']]
        perturbed_expr = self.expression_data[pair['perturbed_idx']]
        perturbation = pair['perturbation']

        # Apply data augmentation only to the baseline expression
        if self.augment and self.training:
            noise = np.random.normal(0, 0.05, baseline_expr.shape)
            baseline_expr = baseline_expr + noise

            mask = np.random.random(baseline_expr.shape) > 0.05
            baseline_expr = baseline_expr * mask

            scale = np.random.uniform(0.95, 1.05)
            baseline_expr = baseline_expr * scale

        return (torch.FloatTensor(baseline_expr),
                torch.FloatTensor(perturbation),
                torch.FloatTensor(perturbed_expr))


class PerturbationEmbedding(nn.Module):
    def __init__(self, pert_dim, emb_dim):
        super().__init__()
        self.embedding = nn.Linear(pert_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, pert):
        x = self.embedding(pert)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class CITERNATransformerModel(nn.Module):
    def __init__(self, input_dim, pert_dim, hidden_dim=512, n_layers=3, n_heads=8,
                 dropout=0.1, attention_dropout=0.1, ffn_dropout=0.1,
                 use_pert_emb=True, pert_emb_dim=64):
        super(CITERNATransformerModel, self).__init__()
        self.input_dim = input_dim
        self.pert_dim = pert_dim
        self.hidden_dim = hidden_dim
        self.use_pert_emb = use_pert_emb

        # Expression encoder
        self.expression_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Perturbation embedding
        if use_pert_emb:
            self.pert_encoder = PerturbationEmbedding(pert_dim, pert_emb_dim)
            pert_out_dim = pert_emb_dim
        else:
            self.pert_encoder = nn.Sequential(
                nn.Linear(pert_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            pert_out_dim = hidden_dim

        # Fusion dimension adjustment
        fusion_dim = hidden_dim + pert_out_dim
        self.fusion_dim = ((fusion_dim + n_heads - 1) // n_heads) * n_heads
        if self.fusion_dim != fusion_dim:
            self.fusion_proj = nn.Linear(fusion_dim, self.fusion_dim)
        else:
            self.fusion_proj = nn.Identity()

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim*4,
            dropout=ffn_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        # Output layers
        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

        # Perturbation prediction head
        self.perturbation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
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

    def forward(self, baseline_expr, pert):
        # Baseline expression encoding
        expr_feat = self.expression_encoder(baseline_expr)

        # Perturbation encoding
        pert_feat = self.pert_encoder(pert)

        # Fusion
        fusion_input = torch.cat([expr_feat, pert_feat], dim=1)
        fusion_input = self.fusion_proj(fusion_input)
        fusion_input = fusion_input.unsqueeze(1)  # (batch, seq=1, dim)

        # Transformer processing
        x_trans = self.transformer(fusion_input).squeeze(1)

        # Output
        fused = self.fusion(x_trans)
        output = self.output(fused)  # Predicted perturbed expression
        pert_pred = self.perturbation_head(fused)

        return output, pert_pred


def train_model(model, train_loader, optimizer, scheduler, device, aux_weight=0.1):
    model.train()
    total_loss = 0
    accumulation_steps = 4
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        baseline_expr, pert, target_expr = batch
        baseline_expr, pert, target_expr = baseline_expr.to(
            device), pert.to(device), target_expr.to(device)

        # Forward pass: baseline expression + perturbation condition -> perturbed expression
        output, pert_pred = model(baseline_expr, pert)

        # Calculate loss: predicted perturbed expression vs ground truth
        main_loss = F.mse_loss(output, target_expr)
        aux_loss = F.mse_loss(pert_pred, pert)
        loss = main_loss + aux_weight * aux_loss

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
    model.eval()
    total_loss = 0
    total_r2 = 0
    total_pearson = 0
    total_pert_r2 = 0

    with torch.no_grad():
        for batch in test_loader:
            baseline_expr, pert, target_expr = batch
            baseline_expr, pert, target_expr = baseline_expr.to(
                device), pert.to(device), target_expr.to(device)
            output, pert_pred = model(baseline_expr, pert)

            # Calculate loss: predicted perturbed expression vs ground truth
            main_loss = F.mse_loss(output, target_expr)
            aux_loss = F.mse_loss(pert_pred, pert)
            loss = main_loss + aux_weight * aux_loss

            total_loss += loss.item()
            r2 = r2_score(target_expr.cpu().numpy(), output.cpu().numpy())
            total_r2 += r2
            pearson = np.mean([pearsonr(target_expr[i].cpu().numpy(), output[i].cpu().numpy())[0]
                               for i in range(target_expr.size(0))])
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


def objective(trial, timestamp):
    global train_dataset, test_dataset, device, pca_model

    params = {
        'pca_dim': 128,
        'n_hidden': trial.suggest_int('n_hidden', 256, 1024),
        'n_layers': trial.suggest_int('n_layers', 2, 4),
        'n_heads': trial.suggest_int('n_heads', 4, 8),
        'dropout': trial.suggest_float('dropout', 0.1, 0.3),
        'attention_dropout': trial.suggest_float('attention_dropout', 0.1, 0.2),
        'ffn_dropout': trial.suggest_float('ffn_dropout', 0.1, 0.2),
        'aux_weight': trial.suggest_float('aux_weight', 0.05, 0.15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-4, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'use_pert_emb': trial.suggest_categorical('use_pert_emb', [True, False]),
        'pert_emb_dim': trial.suggest_int('pert_emb_dim', 32, 128)
    }

    # Create model
    model = CITERNATransformerModel(
        input_dim=128,
        pert_dim=train_dataset.perturbations.shape[1],
        hidden_dim=params['n_hidden'],
        n_layers=params['n_layers'],
        n_heads=params['n_heads'],
        dropout=params['dropout'],
        attention_dropout=params['attention_dropout'],
        ffn_dropout=params['ffn_dropout'],
        use_pert_emb=params['use_pert_emb'],
        pert_emb_dim=params['pert_emb_dim']
    ).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Training loop
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    max_epochs = 150

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}'):
            baseline_expr, pert, target_expr = batch
            baseline_expr, pert, target_expr = baseline_expr.to(
                device), pert.to(device), target_expr.to(device)

            optimizer.zero_grad()
            output, pert_pred = model(baseline_expr, pert)

            mse_loss = F.mse_loss(output, target_expr)
            pert_loss = F.mse_loss(pert_pred, pert)
            loss = mse_loss + params['aux_weight'] * pert_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                baseline_expr, pert, target_expr = batch
                baseline_expr, pert, target_expr = baseline_expr.to(
                    device), pert.to(device), target_expr.to(device)

                output, pert_pred = model(baseline_expr, pert)

                mse_loss = F.mse_loss(output, target_expr)
                pert_loss = F.mse_loss(pert_pred, pert)
                loss = mse_loss + params['aux_weight'] * pert_loss

                val_loss += loss.item()

        val_loss /= len(test_loader)
        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       f'best_rna_model_trial_{trial.number}_{timestamp}.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        if (epoch + 1) % 20 == 0:
            print(
                f'Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')

    return best_val_loss


def evaluate_and_save_model(model, test_loader, device, save_path, common_genes_info=None, pca_model=None, scaler=None):
    """Evaluate model and save results"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_baselines = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            baseline_expr, pert, target_expr = batch
            baseline_expr, pert, target_expr = baseline_expr.to(
                device), pert.to(device), target_expr.to(device)
            output, _ = model(baseline_expr, pert)

            all_predictions.append(output.cpu().numpy())
            all_targets.append(target_expr.cpu().numpy())
            all_baselines.append(baseline_expr.cpu().numpy())

    # Calculate metrics
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_baselines = np.concatenate(all_baselines, axis=0)

    results = calculate_metrics(all_predictions, all_targets)

    # Save model and results
    torch.save({
        'model_state_dict': model.state_dict(),
        'evaluation_results': results,
        'predictions': all_predictions,  # Predicted perturbed expression
        'targets': all_targets,  # Ground-truth perturbed expression
        'baselines': all_baselines,  # Baseline expression
        'gene_names': common_genes_info['genes'] if common_genes_info is not None else None,
        'pca_model': pca_model,
        'scaler': scaler,
        'model_config': {
            'input_dim': 128,
            'pert_dim': train_dataset.perturbations.shape[1],
            'hidden_dim': getattr(model, 'hidden_dim', 512),
            'n_layers': getattr(model, 'n_layers', 3),
            'n_heads': getattr(model, 'n_heads', 8),
            'dropout': getattr(model, 'dropout', 0.1),
            'use_pert_emb': getattr(model, 'use_pert_emb', True)
        }
    }, save_path)

    # Display results
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'PCC', 'R2', 'MSE_DE', 'PCC_DE', 'R2_DE'],
        'Value': [results['MSE'], results['PCC'], results['R2'],
                  results['MSE_DE'], results['PCC_DE'], results['R2_DE']]
    })

    print("\nRNA Model Evaluation Results:")
    print(metrics_df.to_string(index=False,
          float_format=lambda x: '{:.6f}'.format(x)))
    print(f"\nModel and evaluation results saved to: {save_path}")

    return results


def main(gpu_id=None):
    global train_adata, test_adata, train_dataset, test_dataset, device, pca_model, scaler

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'RNA Model Training started at: {timestamp}')

    # Set device
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f'Available GPUs: {gpu_count}')

        if gpu_id is not None:
            if gpu_id >= gpu_count:
                print(f'Warning: GPU {gpu_id} not available, using GPU 0')
                gpu_id = 0
            device = torch.device(f'cuda:{gpu_id}')
            print(f'Using GPU {gpu_id}: {device}')
        else:
            device = torch.device('cuda:0')
            print(f'Using default GPU 0: {device}')
    else:
        device = torch.device('cpu')
        print('CUDA not available, using CPU')

    # Load data
    print('Loading CITE-seq RNA data...')
    train_path = "/data1/yzy/split_new/CITE/PapalexiSatija2021_eccite_RNA_train.h5ad"
    test_path = "/data1/yzy/split_new/CITE/PapalexiSatija2021_eccite_RNA_test.h5ad"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Data files not found: {train_path} or {test_path}")

    train_adata = sc.read_h5ad(train_path)
    test_adata = sc.read_h5ad(test_path)

    print(f'Training data shape: {train_adata.shape}')
    print(f'Test data shape: {test_adata.shape}')

    # Handle common genes
    print("Processing gene consistency...")
    train_genes = set(train_adata.var_names)
    test_genes = set(test_adata.var_names)
    common_genes = list(train_genes & test_genes)
    print(f"Training genes: {len(train_genes)}")
    print(f"Test genes: {len(test_genes)}")
    print(f"Common genes: {len(common_genes)}")

    common_genes.sort()
    train_gene_idx = [train_adata.var_names.get_loc(
        gene) for gene in common_genes]
    test_gene_idx = [test_adata.var_names.get_loc(
        gene) for gene in common_genes]

    # Create PCA model
    pca_model = PCA(n_components=128)

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

    # Common genes info
    common_genes_info = {
        'genes': common_genes,
        'train_idx': train_gene_idx,
        'test_idx': test_gene_idx
    }

    # Create datasets
    train_dataset = CITERNADataset(
        train_adata,
        perturbation_key='perturbation',
        scaler=scaler,
        pca_model=pca_model,
        pca_dim=128,
        fit_pca=False,
        augment=True,
        is_train=True,
        common_genes_info=common_genes_info
    )

    test_dataset = CITERNADataset(
        test_adata,
        perturbation_key='perturbation',
        scaler=scaler,
        pca_model=pca_model,
        pca_dim=128,
        fit_pca=False,
        augment=False,
        is_train=False,
        common_genes_info=common_genes_info
    )

    # Hyperparameter optimization
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
    )

    print('Starting hyperparameter optimization...')
    study.optimize(lambda trial: objective(trial, timestamp), n_trials=30)

    print('Best parameters:')
    for key, value in study.best_params.items():
        print(f'{key}: {value}')

    # Train final model
    best_params = study.best_params
    final_model = CITERNATransformerModel(
        input_dim=128,
        pert_dim=train_dataset.perturbations.shape[1],
        hidden_dim=best_params['n_hidden'],
        n_layers=best_params['n_layers'],
        n_heads=best_params['n_heads'],
        dropout=best_params['dropout'],
        attention_dropout=best_params['attention_dropout'],
        ffn_dropout=best_params['ffn_dropout'],
        use_pert_emb=best_params['use_pert_emb'],
        pert_emb_dim=best_params['pert_emb_dim']
    ).to(device)

    # Final training
    train_loader = DataLoader(
        train_dataset,
        batch_size=best_params['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=best_params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    print('Training final RNA model...')
    best_loss = float('inf')
    best_model = None
    max_epochs = 200

    for epoch in range(max_epochs):
        train_loss = train_model(final_model, train_loader, optimizer, scheduler, device,
                                 aux_weight=best_params['aux_weight'])
        eval_metrics = evaluate_model(final_model, test_loader, device,
                                      aux_weight=best_params['aux_weight'])

        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{max_epochs}:')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Test Loss: {eval_metrics["loss"]:.4f}')
            print(f'R2 Score: {eval_metrics["r2"]:.4f}')
            print(f'Pearson Correlation: {eval_metrics["pearson"]:.4f}')
            print(f'Perturbation R2: {eval_metrics["pert_r2"]:.4f}')

        if eval_metrics["loss"] < best_loss:
            best_loss = eval_metrics["loss"]
            best_model = final_model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'metrics': eval_metrics,
                'best_params': best_params
            }, f'cite_rna_best_model_{timestamp}.pt')
            print(f"Saved best RNA model with loss: {best_loss:.4f}")

    # Load best model and evaluate
    final_model.load_state_dict(best_model)

    print('Evaluating final RNA model...')
    results = evaluate_and_save_model(final_model, test_loader, device,
                                      f'cite_rna_final_model_{timestamp}.pt',
                                      common_genes_info, pca_model, scaler)

    # Save results
    results_df = pd.DataFrame({
        'Metric': ['MSE', 'PCC', 'R2', 'MSE_DE', 'PCC_DE', 'R2_DE'],
        'Value': [results['MSE'], results['PCC'], results['R2'],
                  results['MSE_DE'], results['PCC_DE'], results['R2_DE']],
        'Best_Params': [str(best_params)] * 6
    })

    results_df.to_csv(
        f'cite_rna_evaluation_results_{timestamp}.csv', index=False)

    print("\nFinal RNA Model Evaluation Results:")
    display(results_df)

    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CITE-seq RNA Model Training')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Specify GPU ID (0-7)')
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
    print("CITE-seq RNA Model Training")
    print("=" * 60)

    if args.gpu is not None:
        print(f"Using GPU: {args.gpu}")
    else:
        print("Using default GPU settings")

    results_df = main(gpu_id=args.gpu)
