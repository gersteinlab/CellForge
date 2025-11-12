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


class ATACDataset(Dataset):
    def __init__(self, adata, perturbation_key='perturbation', scaler=None, pca_model=None,
                 pca_dim=128, fit_pca=False, augment=False, is_train=True,
                 common_genes_info=None):
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

        # ATAC-specific preprocessing
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

        # Perturbation One-hot
        self.perturbations = pd.get_dummies(adata.obs[perturbation_key]).values
        print(
            f"{'train' if is_train else 'test'} set perturbation dimension: {self.perturbations.shape[1]}")

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        # Retrieve the current sample
        x_baseline = self.expression_data[idx]
        pert = self.perturbations[idx]

        # For perturbation prediction, create baseline-target pairs
        if np.argmax(pert) == 0:  # control condition
            # If the condition is control, randomly select a non-control perturbation as the target
            non_control_indices = np.where(
                np.argmax(self.perturbations, axis=1) != 0)[0]
            if len(non_control_indices) > 0:
                target_idx = np.random.choice(non_control_indices)
                x_target = self.expression_data[target_idx]
                pert_target = self.perturbations[target_idx]
            else:
                x_target = x_baseline
                pert_target = pert
        else:
            # If the condition is a perturbation, use a control sample as the baseline
            control_indices = np.where(
                np.argmax(self.perturbations, axis=1) == 0)[0]
            if len(control_indices) > 0:
                baseline_idx = np.random.choice(control_indices)
                x_baseline = self.expression_data[baseline_idx]
            x_target = self.expression_data[idx]
            pert_target = pert

        if self.augment and self.training:
            # Add noise to the baseline sample
            noise = np.random.normal(0, 0.05, x_baseline.shape)
            x_baseline = x_baseline + noise

            # Apply random masking
            mask = np.random.random(x_baseline.shape) > 0.05
            x_baseline = x_baseline * mask

        return torch.FloatTensor(x_baseline), torch.FloatTensor(pert_target), torch.FloatTensor(x_target)

# Diffusion Model Components


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        # Ensure the output dimension is correct
        if embeddings.shape[-1] != self.dim:
            # Pad or truncate when the dimension does not match
            if embeddings.shape[-1] < self.dim:
                padding = torch.zeros(
                    embeddings.shape[0], self.dim - embeddings.shape[-1], device=device)
                embeddings = torch.cat([embeddings, padding], dim=-1)
            else:
                embeddings = embeddings[:, :self.dim]

        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.block1 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res_conv = nn.Linear(
            in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb
        h = self.block2(h)
        return h + self.res_conv(x)


class ConditionalDiffusionTransformer(nn.Module):
    def __init__(self, input_dim, train_pert_dim, test_pert_dim, hidden_dim=512,
                 n_layers=4, n_heads=8, dropout=0.1, time_emb_dim=128,
                 diffusion_steps=1000, pert_emb_dim=64):
        super(ConditionalDiffusionTransformer, self).__init__()
        self.input_dim = input_dim
        self.train_pert_dim = train_pert_dim
        self.test_pert_dim = test_pert_dim
        # Ensure hidden_dim is divisible by n_heads
        self.hidden_dim = ((hidden_dim + n_heads - 1) // n_heads) * n_heads
        self.diffusion_steps = diffusion_steps
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_embeddings = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU()
        )

        # Input projection
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)

        # Perturbation embedding
        self.train_pert_encoder = nn.Sequential(
            nn.Linear(train_pert_dim, pert_emb_dim),
            nn.LayerNorm(pert_emb_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.test_pert_encoder = nn.Sequential(
            nn.Linear(test_pert_dim, pert_emb_dim),
            nn.LayerNorm(pert_emb_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Conditional embedding projection
        self.cond_proj = nn.Linear(
            pert_emb_dim + time_emb_dim, self.hidden_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=n_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, input_dim)
        )

        # Multi-task heads
        self.train_perturbation_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim//2, train_pert_dim)
        )

        self.test_perturbation_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim//2, test_pert_dim)
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

    def forward(self, x, pert, timestep, is_train=True):
        batch_size = x.shape[0]

        # Time embedding
        time_emb = self.time_embeddings(timestep)
        time_emb = self.time_mlp(time_emb)

        # Perturbation embedding
        if is_train:
            pert_emb = self.train_pert_encoder(pert)
        else:
            pert_emb = self.test_pert_encoder(pert)

        # Combine perturbation and time embeddings
        cond_emb = torch.cat([pert_emb, time_emb], dim=-1)
        cond_emb = self.cond_proj(cond_emb)

        # Input projection
        x_proj = self.input_proj(x)

        # Add conditional information
        x_cond = x_proj.unsqueeze(1) + cond_emb.unsqueeze(1)

        # Transformer processing
        x_trans = self.transformer(x_cond)

        # Output projection
        output = self.output_proj(x_trans.squeeze(1))

        # Perturbation prediction
        if is_train:
            pert_pred = self.train_perturbation_head(x_trans.squeeze(1))
        else:
            pert_pred = self.test_perturbation_head(x_trans.squeeze(1))

        return output, pert_pred


class DiffusionModel(nn.Module):
    def __init__(self, model, diffusion_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.diffusion_steps = diffusion_steps

        # Linear beta schedule - register as buffers so they move to the correct device automatically
        self.register_buffer('betas', torch.linspace(
            beta_start, beta_end, diffusion_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))

    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bars[t]).view(-1, 1)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(
            1 - self.alpha_bars[t]).view(-1, 1)

        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

    def p_sample(self, x, pert, t, is_train=True):
        """Sample from p(x_{t-1} | x_t, pert)"""
        with torch.no_grad():
            pred_noise, _ = self.model(x, pert, t, is_train)
            alpha_t = self.alphas[t].view(-1, 1)
            alpha_bar_t = self.alpha_bars[t].view(-1, 1)
            beta_t = self.betas[t].view(-1, 1)

            # Predict x_0
            pred_x_start = (x - torch.sqrt(1 - alpha_bar_t) *
                            pred_noise) / torch.sqrt(alpha_bar_t)

            # Sample x_{t-1}
            if t[0] > 0:  # Check the first element because t is a batch tensor
                noise = torch.randn_like(x)
                mean = (pred_x_start * torch.sqrt(alpha_bar_t) +
                        x * torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_t)
                return mean + torch.sqrt(beta_t) * noise
            else:
                return pred_x_start

    def p_sample_loop(self, shape, pert, is_train=True):
        """Generate samples from the diffusion model"""
        device = next(self.parameters()).device
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.diffusion_steps)):
            t_tensor = torch.full(
                (shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, pert, t_tensor, is_train)

        return x

    def forward(self, x_start, pert, is_train=True):
        """Training forward pass"""
        batch_size = x_start.shape[0]
        device = x_start.device

        # Sample random timesteps
        t = torch.randint(0, self.diffusion_steps,
                          (batch_size,), device=device)

        # Sample noise
        noise = torch.randn_like(x_start)

        # Add noise to x_start
        x_noisy = self.q_sample(x_start, t, noise)

        # Predict noise
        pred_noise, pert_pred = self.model(x_noisy, pert, t, is_train)

        return pred_noise, pert_pred, noise, t


def train_model(model, train_loader, optimizer, scheduler, device, aux_weight=0.1):
    model.train()
    total_loss = 0
    accumulation_steps = 4
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        x_baseline, pert, x_target = batch
        x_baseline, pert, x_target = x_baseline.to(
            device), pert.to(device), x_target.to(device)

        # Forward pass through diffusion model
        pred_noise, pert_pred, noise, t = model(x_target, pert, is_train=True)

        # Calculate losses
        diffusion_loss = F.mse_loss(pred_noise, noise)
        aux_loss = F.mse_loss(pert_pred, pert)
        loss = diffusion_loss + aux_weight * aux_loss
        loss = loss / accumulation_steps

        # Backward pass
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
            x_baseline, pert, x_target = batch
            x_baseline, pert, x_target = x_baseline.to(
                device), pert.to(device), x_target.to(device)

            # Generate predictions using diffusion sampling
            pred_x = model.p_sample_loop(x_target.shape, pert, is_train=False)

            # Also get perturbation prediction
            _, pert_pred, _, _ = model(x_target, pert, is_train=False)

            # Calculate losses
            diffusion_loss = F.mse_loss(pred_x, x_target)
            aux_loss = F.mse_loss(pert_pred, pert)
            loss = diffusion_loss + aux_weight * aux_loss

            total_loss += loss.item()
            r2 = r2_score(x_target.cpu().numpy(), pred_x.cpu().numpy())
            total_r2 += r2
            pearson = np.mean([pearsonr(x_target[i].cpu().numpy(), pred_x[i].cpu().numpy())[0]
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


def evaluate_and_save_model(model, test_loader, device, save_path='atac_diffusion_best.pt',
                            common_genes_info=None, pca_model=None, scaler=None):
    """Evaluate model and save results"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            x_baseline, pert, x_target = batch
            x_baseline, pert, x_target = x_baseline.to(
                device), pert.to(device), x_target.to(device)

            # Generate predictions using diffusion sampling
            pred_x = model.p_sample_loop(x_target.shape, pert, is_train=False)

            all_predictions.append(pred_x.cpu().numpy())
            all_targets.append(x_target.cpu().numpy())

    # Calculate overall metrics
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate evaluation metrics
    results = calculate_metrics(all_predictions, all_targets)

    # Save model and evaluation results
    torch.save({
        'model_state_dict': model.state_dict(),
        'evaluation_results': results,
        'predictions': all_predictions,
        'targets': all_targets,
        'gene_names': common_genes_info['genes'] if common_genes_info is not None else None,
        'pca_model': pca_model,
        'scaler': scaler,
        'model_config': {
            'input_dim': 128,
            'train_pert_dim': train_dataset.perturbations.shape[1],
            'test_pert_dim': test_dataset.perturbations.shape[1],
            'hidden_dim': getattr(model.model, 'hidden_dim', 512),
            'n_layers': getattr(model.model, 'n_layers', 4),
            'n_heads': getattr(model.model, 'n_heads', 8),
            'dropout': getattr(model.model, 'dropout', 0.1),
            'diffusion_steps': getattr(model, 'diffusion_steps', 1000)
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
        'n_hidden': trial.suggest_int('n_hidden', 256, 1024),
        'n_layers': trial.suggest_int('n_layers', 2, 6),
        'n_heads': trial.suggest_int('n_heads', 4, 8),
        'dropout': trial.suggest_float('dropout', 0.1, 0.3),
        'aux_weight': trial.suggest_float('aux_weight', 0.05, 0.15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-4, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'time_emb_dim': trial.suggest_int('time_emb_dim', 64, 256),
        'pert_emb_dim': trial.suggest_int('pert_emb_dim', 32, 128),
        'diffusion_steps': trial.suggest_categorical('diffusion_steps', [500, 1000, 2000])
    }

    # Create model
    base_model = ConditionalDiffusionTransformer(
        input_dim=128,
        train_pert_dim=train_dataset.perturbations.shape[1],
        test_pert_dim=test_dataset.perturbations.shape[1],
        hidden_dim=params['n_hidden'],
        n_layers=params['n_layers'],
        n_heads=params['n_heads'],
        dropout=params['dropout'],
        time_emb_dim=params['time_emb_dim'],
        pert_emb_dim=params['pert_emb_dim']
    )

    model = DiffusionModel(
        base_model, diffusion_steps=params['diffusion_steps']).to(device)

    # Optimizer and scheduler
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
        # Training phase
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}'):
            x_baseline, pert, x_target = batch
            x_baseline, pert, x_target = x_baseline.to(
                device), pert.to(device), x_target.to(device)

            optimizer.zero_grad()

            pred_noise, pert_pred, noise, t = model(
                x_target, pert, is_train=True)

            # Calculate loss
            diffusion_loss = F.mse_loss(pred_noise, noise)
            pert_loss = F.mse_loss(pert_pred, pert)
            loss = diffusion_loss + params['aux_weight'] * pert_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        val_batches_processed = 0
        max_val_batches = 50  # Limit validation batches for speed

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Validation Epoch {epoch+1}', leave=False):
                if val_batches_processed >= max_val_batches:
                    break

                x_baseline, pert, x_target = batch
                x_baseline, pert, x_target = x_baseline.to(
                    device), pert.to(device), x_target.to(device)

                pred_noise, pert_pred, noise, t = model(
                    x_target, pert, is_train=False)

                diffusion_loss = F.mse_loss(pred_noise, noise)
                pert_loss = F.mse_loss(pert_pred, pert)
                loss = diffusion_loss + params['aux_weight'] * pert_loss

                val_loss += loss.item()
                val_batches_processed += 1

        val_loss /= val_batches_processed
        scheduler.step()

        # Early stopping
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

        print(
            f'Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')

    return best_val_loss


def main(gpu_id=None):
    global train_adata, train_dataset, test_dataset, device, pca_model

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'Training started at: {timestamp}')

    # Set device and GPU configuration
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f'Available GPUs: {gpu_count}')
        for i in range(gpu_count):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')

        if gpu_id is not None:
            if gpu_id >= gpu_count:
                print(
                    f'Warning: Specified GPU {gpu_id} not available, using GPU 0')
                gpu_id = 0
            device = torch.device(f'cuda:{gpu_id}')
            print(f'Using specified GPU {gpu_id}: {device}')
        else:
            device = torch.device('cuda:0')
            print(f'Using default GPU 0: {device}')
    else:
        device = torch.device('cpu')
        print('CUDA not available, using CPU')

    # Load data
    print('Loading ATAC data...')
    train_path = "/data1/yzy/split_new/ATAC/LiscovitchBrauerSanjana2021_train.h5ad"
    test_path = "/data1/yzy/split_new/ATAC/LiscovitchBrauerSanjana2021_test.h5ad"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Data files not found: {train_path} or {test_path}")

    train_adata = sc.read_h5ad(train_path)
    test_adata = sc.read_h5ad(test_path)

    print(f'Train data shape: {train_adata.shape}')
    print(f'Test data shape: {test_adata.shape}')

    # Validate data consistency
    train_pert = set(train_adata.obs['perturbation'].unique())
    test_pert = set(test_adata.obs['perturbation'].unique())

    print(f"Number of perturbation types in training set: {len(train_pert)}")
    print(f"Number of perturbation types in test set: {len(test_pert)}")

    # Ensure consistent gene sets
    print("Processing gene set consistency...")
    train_genes = set(train_adata.var_names)
    test_genes = set(test_adata.var_names)
    common_genes = list(train_genes & test_genes)
    print(f"Train genes: {len(train_genes)}")
    print(f"Test genes: {len(test_genes)}")
    print(f"Common genes: {len(common_genes)}")

    # Sort common genes for consistency
    common_genes.sort()

    # Get common gene indices
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

    # Standardization
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    train_data = np.clip(train_data, -10, 10)
    train_data = train_data / 10.0

    # Fit PCA model
    pca_model.fit(train_data)

    # Save common genes info
    common_genes_info = {
        'genes': common_genes,
        'train_idx': train_gene_idx,
        'test_idx': test_gene_idx
    }

    # Create datasets
    train_dataset = ATACDataset(
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

    test_dataset = ATACDataset(
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

    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
    )

    # Run hyperparameter optimization
    print('Starting hyperparameter optimization...')
    study.optimize(lambda trial: objective(trial, timestamp), n_trials=50)

    # Print best parameters
    print('Best parameters:')
    for key, value in study.best_params.items():
        print(f'{key}: {value}')

    # Train final model with best parameters
    best_params = study.best_params
    base_model = ConditionalDiffusionTransformer(
        input_dim=128,
        train_pert_dim=train_dataset.perturbations.shape[1],
        test_pert_dim=test_dataset.perturbations.shape[1],
        hidden_dim=best_params['n_hidden'],
        n_layers=best_params['n_layers'],
        n_heads=best_params['n_heads'],
        dropout=best_params['dropout'],
        time_emb_dim=best_params['time_emb_dim'],
        pert_emb_dim=best_params['pert_emb_dim']
    )

    final_model = DiffusionModel(
        base_model, diffusion_steps=best_params['diffusion_steps']).to(device)

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
        batch_size=best_params['batch_size'],
        shuffle=False,
        num_workers=4,
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
    max_epochs = 300  # Train longer for better results

    for epoch in range(max_epochs):
        train_loss = train_model(final_model, train_loader, optimizer, scheduler, device,
                                 aux_weight=best_params['aux_weight'])

        eval_metrics = evaluate_model(final_model, test_loader, device,
                                      aux_weight=best_params['aux_weight'])

        if (epoch + 1) % 30 == 0:
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
            }, f'atac_diffusion_best_model_{timestamp}.pt')
            print(f"Saved best model with loss: {best_loss:.4f}")

    # Load best model for final evaluation
    final_model.load_state_dict(best_model)

    # Evaluate and save final results
    print('Evaluating final model...')
    results = evaluate_and_save_model(final_model, test_loader, device,
                                      f'atac_diffusion_final_model_{timestamp}.pt',
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
        f'atac_diffusion_evaluation_results_{timestamp}.csv', index=False)

    # Display results
    print("\nFinal Evaluation Results:")
    display(results_df)

    return results_df


def load_model_for_prediction(model_path, device='cuda'):
    """
    Load trained model for downstream analysis

    Args:
        model_path: Path to model file
        device: Device ('cuda' or 'cpu')

    Returns:
        dict: Dictionary containing model, predictions, gene names, etc.
    """
    print(f"Loading model from {model_path}")

    # Load saved data
    checkpoint = torch.load(
        model_path, map_location=device, weights_only=False)

    # Extract information
    model_state = checkpoint['model_state_dict']
    predictions = checkpoint['predictions']
    targets = checkpoint['targets']
    gene_names = checkpoint['gene_names']
    pca_model = checkpoint['pca_model']
    scaler = checkpoint['scaler']
    model_config = checkpoint['model_config']
    evaluation_results = checkpoint['evaluation_results']

    # Rebuild model
    base_model = ConditionalDiffusionTransformer(
        input_dim=model_config['input_dim'],
        train_pert_dim=model_config['train_pert_dim'],
        test_pert_dim=model_config['test_pert_dim'],
        hidden_dim=model_config['hidden_dim'],
        n_layers=model_config['n_layers'],
        n_heads=model_config['n_heads'],
        dropout=model_config['dropout']
    )

    model = DiffusionModel(
        base_model, diffusion_steps=model_config['diffusion_steps']).to(device)

    # Load model weights
    model.load_state_dict(model_state)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Gene names: {len(gene_names) if gene_names else 'None'}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")

    return {
        'model': model,
        'predictions': predictions,
        'targets': targets,
        'gene_names': gene_names,
        'pca_model': pca_model,
        'scaler': scaler,
        'model_config': model_config,
        'evaluation_results': evaluation_results
    }


def create_anndata_for_analysis(predictions, targets, gene_names, perturbation_info=None):
    """
    Create AnnData objects for downstream analysis (DEGs, KEGG, etc.)

    Args:
        predictions: Prediction results (n_samples, n_genes)
        targets: True values (n_samples, n_genes)
        gene_names: Gene names list
        perturbation_info: Perturbation information (optional)

    Returns:
        tuple: (pred_adata, target_adata) AnnData objects for predictions and targets
    """
    import anndata as ad

    # Create AnnData for predictions
    pred_adata = ad.AnnData(X=predictions)
    pred_adata.var_names = gene_names
    pred_adata.var['feature_types'] = 'ATAC Peaks'

    # Create AnnData for targets
    target_adata = ad.AnnData(X=targets)
    target_adata.var_names = gene_names
    target_adata.var['feature_types'] = 'ATAC Peaks'

    # Add perturbation information if available
    if perturbation_info is not None:
        pred_adata.obs['perturbation'] = perturbation_info
        target_adata.obs['perturbation'] = perturbation_info

    # Add sample identifiers
    pred_adata.obs['sample_type'] = 'predicted'
    target_adata.obs['sample_type'] = 'observed'

    print(f"Created AnnData objects:")
    print(f"  Predictions: {pred_adata.shape}")
    print(f"  Targets: {target_adata.shape}")

    return pred_adata, target_adata


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='ATAC Conditional Diffusion Transformer Model Training')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Specify GPU number to use (e.g., --gpu 0 for GPU 0, --gpu 1 for GPU 1)')
    parser.add_argument('--list-gpus', action='store_true',
                        help='List all available GPUs and exit')

    args = parser.parse_args()

    # If just listing GPUs, show info and exit
    if args.list_gpus:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f'Available GPUs: {gpu_count}')
            for i in range(gpu_count):
                print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        else:
            print('CUDA not available')
        exit(0)

    # Start training
    print("=" * 60)
    print("ATAC Conditional Diffusion Transformer Model Training")
    print("=" * 60)

    if args.gpu is not None:
        print(f"Using specified GPU: {args.gpu}")
    else:
        print("Using default GPU settings")

    results_df = main(gpu_id=args.gpu)
