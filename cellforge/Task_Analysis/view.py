"""
View utilities for Task Analysis module
"""

import scanpy as sc
import anndata as ad
from typing import Optional

class View:
    """Data viewing utilities for single-cell analysis"""

    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path
        self.adata = None

    def load_dataset(self, dataset_path: str):
        """Load a dataset from path"""
        try:
            self.adata = sc.read_h5ad(dataset_path)
            self.dataset_path = dataset_path
            print(f"✅ Dataset loaded successfully: {dataset_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return False

    def show_basic_info(self):
        """Show basic information about the loaded dataset"""
        if self.adata is None:
            print("❌ No dataset loaded. Use load_dataset() first.")
            return

        print("=" * 50)
        print("DATASET INFORMATION")
        print("=" * 50)
        print(f"Shape: {self.adata.shape}")
        print(f"Number of cells: {self.adata.n_obs}")
        print(f"Number of genes: {self.adata.n_vars}")
        print("\nObservation (cells) columns:")
        print(self.adata.obs.columns.tolist())
        print("\nVariable (genes) columns:")
        print(self.adata.var.columns.tolist())
        print("=" * 50)

    def show_sample_data(self, n_cells: int = 5, n_genes: int = 5):
        """Show sample data from the dataset"""
        if self.adata is None:
            print("❌ No dataset loaded. Use load_dataset() first.")
            return

        print("\nSample observation data (first 5 cells):")
        print(self.adata.obs.head(n_cells))
        print("\nSample variable data (first 5 genes):")
        print(self.adata.var.head(n_genes))
