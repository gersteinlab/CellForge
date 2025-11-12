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
import math
import warnings
warnings.filterwarnings('ignore')


def print_log(message):
    """Custom print function to simulate notebook output style"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


# Global variables
train_adata = None
train_dataset = None
test_dataset = None
device = None
pca_model = None


class CytokinePerturbationDataset(Dataset):
    def __init__(self, adata, perturbation_key='perturbation', age_key='age',
                 scaler=None, pca_model=None, pca_dim=128, fit_pca=False,
                 augment=False, is_train=True, common_genes_info=None):
        self.adata = adata
        self.perturbation_key = perturbation_key
        self.age_key = age_key
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

        # Data preprocessing
        data = np.maximum(data, 0)
        data = np.maximum(data, 1e-10)
        data = np.log1p(data)

        # Use the provided scaler or create a new one
        if scaler is None:
            self.scaler = StandardScaler()
            data = self.scaler.fit_transform(data)
        else:
            self.scaler = scaler
            data = self.scaler.transform(data)

        data = np.clip(data, -10, 10)
        data = data / 10.0

        # PCA Dimension Reduction
        if pca_model is None:
            if fit_pca:
                self.pca = PCA(n_components=pca_dim)
                self.expression_data = self.pca.fit_transform(data)
            else:
                raise ValueError('pca_model must be provided for test set')
        else:
            self.pca = pca_model
            self.expression_data = self.pca.transform(data)

        # Perturbation One-hot encoding
        self.perturbations = pd.get_dummies(adata.obs[perturbation_key]).values
        print(
            f"{'train' if is_train else 'test'} set perturbation dimension: {self.perturbations.shape[1]}")

        # Time embedding - convert time points into numeric features
        self.time_embeddings = self._encode_timepoints(adata.obs[age_key])
        print(
            f"{'train' if is_train else 'test'} set time embedding dimension: {self.time_embeddings.shape[1]}")

        # Create perturbation-time mapping for pairing
        self._create_perturbation_time_mapping()

    def _encode_timepoints(self, timepoints):
        """Encode time points into numerical features"""
        time_mapping = {}
        unique_times = sorted(timepoints.unique())

        # Handle special time points
        for i, time_str in enumerate(unique_times):
            if time_str == 'iPSC' or time_str == 'iPSCs':
                time_mapping[time_str] = 20.0  # Final state
            elif time_str.startswith('D'):
                try:
                    time_mapping[time_str] = float(time_str[1:])
                except:
                    time_mapping[time_str] = 0.0
            else:
                time_mapping[time_str] = 0.0

        # Create time feature vectors [sin(time), cos(time), time_normalized]
        time_features = []
        for tp in timepoints:
            time_val = time_mapping[tp]
            # Normalize to 0-20 days
            sin_time = np.sin(2 * np.pi * time_val / 20.0)
            cos_time = np.cos(2 * np.pi * time_val / 20.0)
            norm_time = time_val / 20.0
            time_features.append([sin_time, cos_time, norm_time])

        return np.array(time_features)

    def _create_perturbation_time_mapping(self):
        """Create perturbation-time mapping for correct pairing"""
        self.pert_time_mapping = {}
        for i, (pert, time_emb) in enumerate(zip(self.perturbations, self.time_embeddings)):
            pert_key = tuple(pert)
            time_key = tuple(time_emb)
            key = (pert_key, time_key)
            if key not in self.pert_time_mapping:
                self.pert_time_mapping[key] = []
            self.pert_time_mapping[key].append(i)

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        # Retrieve current sample
        x_baseline = self.expression_data[idx]
        pert = self.perturbations[idx]
        time_emb = self.time_embeddings[idx]

        # For perturbation prediction we create baseline-target pairs
        # Use control samples as baselines and other perturbations as targets
        if np.argmax(pert) == 0:  # control condition
            # For control samples, find other perturbations at the same time point
            current_time = time_emb
            # Randomly select a non-control perturbation as the target
            non_control_indices = np.where(
                np.argmax(self.perturbations, axis=1) != 0)[0]
            if len(non_control_indices) > 0:
                target_idx = np.random.choice(non_control_indices)
                x_target = self.expression_data[target_idx]
                pert_target = self.perturbations[target_idx]
            else:
                # If no non-control samples exist, fall back to the current sample
                x_target = x_baseline
                pert_target = pert
        else:
            # For perturbation conditions, use a control sample as the baseline
            control_indices = np.where(
                np.argmax(self.perturbations, axis=1) == 0)[0]
            if len(control_indices) > 0:
                baseline_idx = np.random.choice(control_indices)
                x_baseline = self.expression_data[baseline_idx]
            x_target = self.expression_data[idx]
            pert_target = pert

        if self.augment and self.training:
            # Add noise to the baseline
            noise = np.random.normal(0, 0.05, x_baseline.shape)
            x_baseline = x_baseline + noise

            # Apply random masking
            mask = np.random.random(x_baseline.shape) > 0.05
            x_baseline = x_baseline * mask

        return torch.FloatTensor(x_baseline), torch.FloatTensor(pert_target), torch.FloatTensor(time_emb), torch.FloatTensor(x_target)

# Transformer-based perturbation prediction model


class CytokineTransformerModel(nn.Module):
    def __init__(self, input_dim, pert_dim, time_dim=3, hidden_dim=512,
                 n_layers=4, n_heads=8, dropout=0.1, use_attention=True):
        super(CytokineTransformerModel, self).__init__()
        self.input_dim = input_dim
        self.pert_dim = pert_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention

        # Input projection layer
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Perturbation embedding layer
        self.perturbation_embedding = nn.Sequential(
            nn.Linear(pert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Time embedding layer
        self.time_embedding = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(3, hidden_dim))

        if use_attention:
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers)
        else:
            # Simple MLP
            self.mlp_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ) for _ in range(n_layers)
            ])

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

        # Perturbation prediction head
        self.perturbation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, pert_dim)
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

    def forward(self, x_baseline, pert, time_emb):
        batch_size = x_baseline.size(0)

        # Project to hidden dimension
        x_proj = self.input_projection(x_baseline)
        pert_proj = self.perturbation_embedding(pert)
        time_proj = self.time_embedding(time_emb)

        if self.use_attention:
            # Create sequence input [baseline, perturbation, time]
            sequence_input = torch.stack(
                # (batch, 3, hidden_dim)
                [x_proj, pert_proj, time_proj], dim=1)

            # Add positional encoding
            sequence_input = sequence_input + \
                self.positional_encoding.unsqueeze(0)

            # Transformer encoding
            # (batch, 3, hidden_dim)
            encoded = self.transformer(sequence_input)

            # Fuse all features
            fused = encoded.view(batch_size, -1)  # (batch, 3 * hidden_dim)
        else:
            # Simple MLP processing
            features = [x_proj, pert_proj, time_proj]
            for layer in self.mlp_layers:
                features = [layer(feat) for feat in features]

            # Concatenate features
            fused = torch.cat(features, dim=1)  # (batch, 3 * hidden_dim)

        # Fusion layer
        fused = self.fusion_layer(fused)

        # Predict perturbed expression
        delta_expr = self.output_layer(fused)
        predicted_expr = x_baseline + delta_expr

        # Perturbation prediction
        pert_pred = self.perturbation_head(fused)

        return predicted_expr, delta_expr, pert_pred


def train_model(model, train_loader, optimizer, scheduler, device,
                aux_weight=0.1, l2_weight=1e-4):
    model.train()
    total_loss = 0
    accumulation_steps = 4
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        x_baseline, pert, time_emb, x_target = batch
        x_baseline, pert, time_emb, x_target = x_baseline.to(
            device), pert.to(device), time_emb.to(device), x_target.to(device)

        # Forward pass
        predicted_expr, delta_expr, pert_pred = model(
            x_baseline, pert, time_emb)

        # Compute losses
        # 1. Main loss - predict perturbed expression
        main_loss = F.mse_loss(predicted_expr, x_target)

        # 2. Auxiliary loss - perturbation prediction
        aux_loss = F.mse_loss(pert_pred, pert)

        # 3. L2 regularization
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param, 2)
        l2_loss = l2_weight * l2_reg

        # Total loss
        loss = main_loss + aux_weight * aux_loss + l2_loss
        loss = loss / accumulation_steps

        # Backpropagation
        loss.backward()

        # Gradient accumulation
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(train_loader)


def evaluate_model(model, test_loader, device, aux_weight=0.1, l2_weight=1e-4):
    model.eval()
    total_loss = 0
    total_r2 = 0
    total_pearson = 0
    total_pert_r2 = 0

    with torch.no_grad():
        for batch in test_loader:
            x_baseline, pert, time_emb, x_target = batch
            x_baseline, pert, time_emb, x_target = x_baseline.to(
                device), pert.to(device), time_emb.to(device), x_target.to(device)

            predicted_expr, delta_expr, pert_pred = model(
                x_baseline, pert, time_emb)

            # Compute losses
            main_loss = F.mse_loss(predicted_expr, x_target)
            aux_loss = F.mse_loss(pert_pred, pert)

            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)
            l2_loss = l2_weight * l2_reg

            loss = main_loss + aux_weight * aux_loss + l2_loss

            total_loss += loss.item()

            # Compute metrics
            r2 = r2_score(x_target.cpu().numpy(), predicted_expr.cpu().numpy())
            total_r2 += r2
            pearson = np.mean([pearsonr(x_target[i].cpu().numpy(), predicted_expr[i].cpu().numpy())[0]
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
    # pred, true: [num_samples, num_genes]
    mse = np.mean((pred - true) ** 2)
    pcc = np.mean([pearsonr(p, t)[0] for p, t in zip(pred.T, true.T)])
    r2 = np.mean([r2_score(t, p) for p, t in zip(pred.T, true.T)])

    # Differentially expressed genes (DE): identify genes whose change exceeds 1 standard deviation
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
    all_baselines = []
    all_perturbations = []
    all_time_embeddings = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            x_baseline, pert, time_emb, x_target = batch
            x_baseline, pert, time_emb, x_target = x_baseline.to(
                device), pert.to(device), time_emb.to(device), x_target.to(device)

            predicted_expr, delta_expr, pert_pred = model(
                x_baseline, pert, time_emb)

            # Collect predictions and true values
            all_predictions.append(predicted_expr.cpu().numpy())
            all_targets.append(x_target.cpu().numpy())
            all_baselines.append(x_baseline.cpu().numpy())
            all_perturbations.append(pert.cpu().numpy())
            all_time_embeddings.append(time_emb.cpu().numpy())

    # Calculate overall metrics
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_baselines = np.concatenate(all_baselines, axis=0)
    all_perturbations = np.concatenate(all_perturbations, axis=0)
    all_time_embeddings = np.concatenate(all_time_embeddings, axis=0)

    # Calculate evaluation metrics
    results = calculate_metrics(all_predictions, all_targets)

    # Save model and evaluation results
    torch.save({
        'model_state_dict': model.state_dict(),
        'evaluation_results': results,
        'predictions': all_predictions,
        'targets': all_targets,
        'baselines': all_baselines,
        'perturbations': all_perturbations,
        'time_embeddings': all_time_embeddings,
        'gene_names': common_genes_info['genes'] if common_genes_info is not None else None,
        'pca_model': pca_model,
        'scaler': scaler,
        'model_config': {
            'input_dim': 128,
            'pert_dim': train_dataset.perturbations.shape[1],
            'time_dim': 3,
            'hidden_dim': 512,
            'n_layers': 4,
            'n_heads': 8,
            'dropout': 0.1,
            'use_attention': True
        }
    }, save_path)

    # Create DataFrame to display results
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


def objective(trial, timestamp):
    global train_dataset, test_dataset, device, pca_model

    # Hyperparameter search space
    params = {
        'pca_dim': 128,  # Fixed PCA dimension
        'hidden_dim': trial.suggest_categorical('hidden_dim', [256, 512, 768]),
        'n_layers': trial.suggest_int('n_layers', 2, 6),
        'n_heads': trial.suggest_categorical('n_heads', [4, 8, 16]),
        'dropout': trial.suggest_float('dropout', 0.05, 0.3),
        'aux_weight': trial.suggest_float('aux_weight', 0.05, 0.2),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'use_attention': trial.suggest_categorical('use_attention', [True, False])
    }

    # Create model
    model = CytokineTransformerModel(
        input_dim=128,
        pert_dim=train_dataset.perturbations.shape[1],
        time_dim=3,
        hidden_dim=params['hidden_dim'],
        n_layers=params['n_layers'],
        n_heads=params['n_heads'],
        dropout=params['dropout'],
        use_attention=params['use_attention']
    ).to(device)

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=min(params['batch_size'] * 2, 256),
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Training loop
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    max_epochs = 80

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}', leave=False):
            x_baseline, pert, time_emb, x_target = batch
            x_baseline, pert, time_emb, x_target = x_baseline.to(
                device), pert.to(device), time_emb.to(device), x_target.to(device)

            optimizer.zero_grad()
            predicted_expr, delta_expr, pert_pred = model(
                x_baseline, pert, time_emb)

            # Calculate loss
            main_loss = F.mse_loss(predicted_expr, x_target)
            aux_loss = F.mse_loss(pert_pred, pert)
            loss = main_loss + params['aux_weight'] * aux_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        val_batches_processed = 0
        max_val_batches = 50

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Validation Epoch {epoch+1}', leave=False):
                if val_batches_processed >= max_val_batches:
                    break

                x_baseline, pert, time_emb, x_target = batch
                x_baseline, pert, time_emb, x_target = x_baseline.to(
                    device), pert.to(device), time_emb.to(device), x_target.to(device)

                predicted_expr, delta_expr, pert_pred = model(
                    x_baseline, pert, time_emb)

                main_loss = F.mse_loss(predicted_expr, x_target)
                aux_loss = F.mse_loss(pert_pred, pert)
                loss = main_loss + params['aux_weight'] * aux_loss

                val_loss += loss.item()
                val_batches_processed += 1

        val_loss /= val_batches_processed
        scheduler.step()

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       f'best_model_trial_{trial.number}_{timestamp}.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        # Report intermediate result for pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss


def main(gpu_id=None):
    global train_adata, train_dataset, test_dataset, device, pca_model

    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'Training started at: {timestamp}')

    # Set device and GPU configuration
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f'Available GPUs: {gpu_count}')
        for i in range(gpu_count):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')

        # Select GPU
        if gpu_id is not None:
            if gpu_id >= gpu_count:
                print(
                    f'Warning: specified GPU {gpu_id} not available, using GPU 0')
                gpu_id = 0
            device = torch.device(f'cuda:{gpu_id}')
            print(f'Using requested GPU {gpu_id}: {device}')
        else:
            device = torch.device('cuda:0')
            print(f'Using default GPU 0: {device}')
    else:
        device = torch.device('cpu')
        print('CUDA not available, using CPU')

    # Load data
    print('Loading data...')
    train_path = "/data1/yzy/split_new/Cytokines/SchiebingerLander2019_train_processed.h5ad"
    test_path = "/data1/yzy/split_new/Cytokines/SchiebingerLander2019_test_processed.h5ad"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Data files not found: {train_path} or {test_path}")

    train_adata = sc.read_h5ad(train_path)
    test_adata = sc.read_h5ad(test_path)

    print(f'Training data shape: {train_adata.shape}')
    print(f'Test data shape: {test_adata.shape}')

    # Ensure training and test sets share the same gene set
    print("Ensuring consistent gene sets...")
    train_genes = set(train_adata.var_names)
    test_genes = set(test_adata.var_names)
    common_genes = list(train_genes & test_genes)
    print(f"Training gene count: {len(train_genes)}")
    print(f"Test gene count: {len(test_genes)}")
    print(f"Common gene count: {len(common_genes)}")

    # Sort common genes to ensure consistent ordering
    common_genes.sort()

    # Gather indices for the common genes
    train_gene_idx = [train_adata.var_names.get_loc(
        gene) for gene in common_genes]
    test_gene_idx = [test_adata.var_names.get_loc(
        gene) for gene in common_genes]

    # Create PCA model
    pca_model = PCA(n_components=128)

    # Preprocess training data (common genes only)
    if scipy.sparse.issparse(train_adata.X):
        train_data = train_adata.X[:, train_gene_idx].toarray()
    else:
        train_data = train_adata.X[:, train_gene_idx]
    train_data = np.maximum(train_data, 0)
    train_data = np.maximum(train_data, 1e-10)
    train_data = np.log1p(train_data)

    # Standardize with StandardScaler
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    train_data = np.clip(train_data, -10, 10)
    train_data = train_data / 10.0

    # Fit PCA model on training data
    pca_model.fit(train_data)

    # Save metadata for common genes
    common_genes_info = {
        'genes': common_genes,
        'train_idx': train_gene_idx,
        'test_idx': test_gene_idx
    }

    # Create datasets
    train_dataset = CytokinePerturbationDataset(
        train_adata,
        perturbation_key='perturbation',
        age_key='age',
        scaler=scaler,
        pca_model=pca_model,
        pca_dim=128,
        fit_pca=False,
        augment=True,
        is_train=True,
        common_genes_info=common_genes_info
    )

    test_dataset = CytokinePerturbationDataset(
        test_adata,
        perturbation_key='perturbation',
        age_key='age',
        scaler=scaler,
        pca_model=pca_model,
        pca_dim=128,
        fit_pca=False,
        augment=False,
        is_train=False,
        common_genes_info=common_genes_info
    )

    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    )

    # Run hyperparameter optimization
    print('Starting hyperparameter optimization...')
    study.optimize(lambda trial: objective(trial, timestamp), n_trials=30)

    # Print best parameters
    print('Best parameters:')
    for key, value in study.best_params.items():
        print(f'{key}: {value}')

    # Train final model with best parameters
    best_params = study.best_params
    final_model = CytokineTransformerModel(
        input_dim=128,
        pert_dim=train_dataset.perturbations.shape[1],
        time_dim=3,
        hidden_dim=best_params['hidden_dim'],
        n_layers=best_params['n_layers'],
        n_heads=best_params['n_heads'],
        dropout=best_params['dropout'],
        use_attention=best_params['use_attention']
    ).to(device)

    # Create data loaders for final training
    train_loader = DataLoader(
        train_dataset,
        batch_size=best_params['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=min(best_params['batch_size'] * 2, 256),
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Optimizer and scheduler for final training
    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # Train final model
    print('Training final model...')
    best_loss = float('inf')
    best_model = None
    max_epochs = 120  # Increase number of epochs

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
                'model_state_dict': final_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'metrics': eval_metrics,
                'best_params': best_params
            }, f'cytokines_best_model_{timestamp}.pt')
            print(f"Saved best model with loss: {best_loss:.4f}")

    # Load best model for final evaluation
    final_model.load_state_dict(best_model)

    # Evaluate and save final results
    print('Evaluating final model...')
    results = evaluate_and_save_model(final_model, test_loader, device,
                                      f'cytokines_final_model_{timestamp}.pt',
                                      common_genes_info, pca_model, scaler)

    # Create detailed results DataFrame
    results_df = pd.DataFrame({
        'Metric': ['MSE', 'PCC', 'R2', 'MSE_DE', 'PCC_DE', 'R2_DE'],
        'Value': [results['MSE'], results['PCC'], results['R2'],
                  results['MSE_DE'], results['PCC_DE'], results['R2_DE']],
        'Best_Params': [str(best_params)] * 6
    })

    # Save results to CSV
    results_df.to_csv(
        f'cytokines_evaluation_results_{timestamp}.csv', index=False)

    # Display results
    print("\nFinal Evaluation Results:")
    display(results_df)

    return results_df


def load_model_for_analysis(model_path, device='cuda'):
    """
    Load the trained model for downstream DEG and KEGG analysis.

    Args:
        model_path: path to the saved model file
        device: device to load weights on ('cuda' or 'cpu')

    Returns:
        dict: dictionary containing model outputs, gene names, and metadata
    """
    print(f"Loading model from {model_path}")

    # Address PyTorch 2.6+ security concerns
    try:
        # Attempt secure loading first
        checkpoint = torch.load(
            model_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"Safe loading failed: {e}")
        print("Trying with weights_only=False (trusted source)...")
        # Fall back to the legacy loader when the source is trusted
        checkpoint = torch.load(
            model_path, map_location=device, weights_only=False)

    # Extract stored information
    model_state = checkpoint['model_state_dict']
    predictions = checkpoint['predictions']
    targets = checkpoint['targets']

    # Use targets as baseline if the checkpoint does not include baseline data
    if 'baselines' in checkpoint:
        baselines = checkpoint['baselines']
    else:
        print("Warning: baseline data missing in checkpoint, using targets as baseline")
        baselines = targets.copy()

    # Retrieve optional data
    perturbations = checkpoint.get('perturbations', None)
    time_embeddings = checkpoint.get('time_embeddings', None)
    gene_names = checkpoint['gene_names']
    pca_model = checkpoint['pca_model']
    scaler = checkpoint['scaler']
    model_config = checkpoint['model_config']
    evaluation_results = checkpoint['evaluation_results']

    # Skip model reconstruction and reuse saved predictions directly
    print("Skipping model reconstruction; using saved predictions for KEGG analysis")
    model = None  # No model object required

    print(f"Model loaded successfully!")
    print(f"Gene names: {len(gene_names) if gene_names else 'None'}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Baselines shape: {baselines.shape}")

    return {
        'model': model,  # None, model object not needed
        'predictions': predictions,
        'targets': targets,
        'baselines': baselines,
        'perturbations': perturbations,
        'time_embeddings': time_embeddings,
        'gene_names': gene_names,
        'pca_model': pca_model,
        'scaler': scaler,
        'model_config': model_config,
        'evaluation_results': evaluation_results
    }


def create_anndata_for_analysis(predictions, targets, baselines, gene_names,
                                perturbations=None, time_embeddings=None):
    """
    Create AnnData objects for downstream analysis (DEGs, KEGG, etc.).

    Args:
        predictions: predicted expression values (n_samples, n_genes)
        targets: observed expression values (n_samples, n_genes)
        baselines: baseline expression (n_samples, n_genes)
        gene_names: list of gene names
        perturbations: perturbation information (optional)
        time_embeddings: time embeddings (optional)

    Returns:
        tuple: (pred_adata, target_adata, baseline_adata) AnnData objects for predictions, observations, and baselines
    """
    import anndata as ad

    # Create AnnData for predictions
    pred_adata = ad.AnnData(X=predictions)
    pred_adata.var_names = gene_names
    pred_adata.var['feature_types'] = 'Gene Expression'

    # Create AnnData for observations
    target_adata = ad.AnnData(X=targets)
    target_adata.var_names = gene_names
    target_adata.var['feature_types'] = 'Gene Expression'

    # Create AnnData for baselines
    baseline_adata = ad.AnnData(X=baselines)
    baseline_adata.var_names = gene_names
    baseline_adata.var['feature_types'] = 'Gene Expression'

    # Add perturbation information if available
    if perturbations is not None:
        # Convert one-hot perturbations into readable labels
        pert_names = []
        for pert in perturbations:
            pert_idx = np.argmax(pert)
            pert_names.append(f'perturbation_{pert_idx}')

        pred_adata.obs['perturbation'] = pert_names
        target_adata.obs['perturbation'] = pert_names
        baseline_adata.obs['perturbation'] = pert_names

    # Add time information if available
    if time_embeddings is not None:
        pred_adata.obs['time_sin'] = time_embeddings[:, 0]
        pred_adata.obs['time_cos'] = time_embeddings[:, 1]
        pred_adata.obs['time_norm'] = time_embeddings[:, 2]

        target_adata.obs['time_sin'] = time_embeddings[:, 0]
        target_adata.obs['time_cos'] = time_embeddings[:, 1]
        target_adata.obs['time_norm'] = time_embeddings[:, 2]

        baseline_adata.obs['time_sin'] = time_embeddings[:, 0]
        baseline_adata.obs['time_cos'] = time_embeddings[:, 1]
        baseline_adata.obs['time_norm'] = time_embeddings[:, 2]

    # Annotate sample type
    pred_adata.obs['sample_type'] = 'predicted'
    target_adata.obs['sample_type'] = 'observed'
    baseline_adata.obs['sample_type'] = 'baseline'

    print(f"Created AnnData objects:")
    print(f"  Predictions: {pred_adata.shape}")
    print(f"  Targets: {target_adata.shape}")
    print(f"  Baselines: {baseline_adata.shape}")

    return pred_adata, target_adata, baseline_adata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cytokine Perturbation Model Training')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Specify GPU index (e.g., --gpu 0 uses GPU 0)')
    parser.add_argument('--list-gpus', action='store_true',
                        help='List all available GPUs and exit')

    args = parser.parse_args()

    if args.list_gpus:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f'Number of available GPUs: {gpu_count}')
            for i in range(gpu_count):
                print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        else:
            print('CUDA not available')
        exit(0)

    print("=" * 60)
    print("Cytokine Perturbation Model Training")
    print("=" * 60)

    if args.gpu is not None:
        print(f"Using specified GPU: {args.gpu}")
    else:
        print("Using default GPU settings")

    results_df = main(gpu_id=args.gpu)
