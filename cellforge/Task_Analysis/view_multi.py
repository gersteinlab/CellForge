import os
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import h5py
import json
import pickle
import gzip
import tarfile
import zipfile
from pathlib import Path
from ..paths import data_path

def read_h5ad(file_path):
    """read the h5ad file and return the basic information"""
    try:
        adata = sc.read_h5ad(file_path)
        info = {
            "type": "h5ad",
            "shape": adata.shape,
            "obs_columns": list(adata.obs.columns),
            "var_columns": list(adata.var.columns),
            "obs_head": adata.obs.head().to_dict(),
            "var_head": adata.var.head().to_dict(),
            "layers": list(adata.layers.keys()) if hasattr(adata, 'layers') else [],
            "obsm_keys": list(adata.obsm.keys()) if hasattr(adata, 'obsm') else [],
            "varm_keys": list(adata.varm.keys()) if hasattr(adata, 'varm') else [],
            "obsp_keys": list(adata.obsp.keys()) if hasattr(adata, 'obsp') else [],
            "varp_keys": list(adata.varp.keys()) if hasattr(adata, 'varp') else [],
        }
        return info
    except Exception as e:
        return {"type": "h5ad", "error": str(e)}

def read_h5(file_path):
    """read the h5 file and return the basic information"""
    try:
        with h5py.File(file_path, 'r') as f:
            info = {
                "type": "h5",
                "keys": list(f.keys()),
                "structure": {}
            }

            def get_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    info["structure"][name] = {
                        "type": "dataset",
                        "shape": obj.shape,
                        "dtype": str(obj.dtype)
                    }
                elif isinstance(obj, h5py.Group):
                    info["structure"][name] = {
                        "type": "group",
                        "keys": list(obj.keys())
                    }

            f.visititems(get_structure)
            return info
    except Exception as e:
        return {"type": "h5", "error": str(e)}

def read_csv(file_path):
    """read the csv file and return the basic information"""
    try:
        df = pd.read_csv(file_path)
        info = {
            "type": "csv",
            "shape": df.shape,
            "columns": list(df.columns),
            "head": df.head().to_dict()
        }
        return info
    except Exception as e:
        return {"type": "csv", "error": str(e)}

def read_tsv(file_path):
    """read the tsv file and return the basic information"""
    try:
        df = pd.read_csv(file_path, sep='\t')
        info = {
            "type": "tsv",
            "shape": df.shape,
            "columns": list(df.columns),
            "head": df.head().to_dict()
        }
        return info
    except Exception as e:
        return {"type": "tsv", "error": str(e)}

def read_txt(file_path):
    """read the txt file and return the basic information"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()[:10]
            info = {
                "type": "txt",
                "preview": lines,
                "total_lines": sum(1 for _ in open(file_path, 'r'))
            }
            return info
    except Exception as e:
        return {"type": "txt", "error": str(e)}

def read_json(file_path):
    """read the json file and return the basic information"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            info = {
                "type": "json",
                "keys": list(data.keys()) if isinstance(data, dict) else "Not a dictionary",
                "preview": str(data)[:500] + "..." if len(str(data)) > 500 else str(data)
            }
            return info
    except Exception as e:
        return {"type": "json", "error": str(e)}

def read_pickle(file_path):
    """read the pickle file and return the basic information"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            info = {
                "type": "pickle",
                "data_type": type(data).__name__,
                "preview": str(data)[:500] + "..." if len(str(data)) > 500 else str(data)
            }
            return info
    except Exception as e:
        return {"type": "pickle", "error": str(e)}

def read_npy(file_path):
    """read the npy file and return the basic information"""
    try:
        data = np.load(file_path)
        info = {
            "type": "npy",
            "shape": data.shape,
            "dtype": str(data.dtype),
            "preview": str(data.flatten()[:10]) + "..." if data.size > 10 else str(data)
        }
        return info
    except Exception as e:
        return {"type": "npy", "error": str(e)}

def read_npz(file_path):
    """read the npz file and return the basic information"""
    try:
        data = np.load(file_path)
        info = {
            "type": "npz",
            "files": list(data.files),
            "preview": {k: str(data[k].shape) for k in data.files}
        }
        return info
    except Exception as e:
        return {"type": "npz", "error": str(e)}

def read_compressed_file(file_path):
    """read the compressed file and return the basic information"""
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                lines = f.readlines()[:10]
                info = {
                    "type": "gz",
                    "preview": lines,
                    "total_lines": sum(1 for _ in gzip.open(file_path, 'rt'))
                }
                return info
        elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
            with tarfile.open(file_path, 'r:gz') as f:
                info = {
                    "type": "tar.gz",
                    "files": f.getnames()[:10],
                    "total_files": len(f.getnames())
                }
                return info
        elif file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as f:
                info = {
                    "type": "zip",
                    "files": f.namelist()[:10],
                    "total_files": len(f.namelist())
                }
                return info
        else:
            return {"type": "unknown_compressed", "error": "Unsupported compression format"}
    except Exception as e:
        return {"type": "compressed", "error": str(e)}

def read_file(file_path):
    """based on the file extension, choose the appropriate read method"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.h5ad':
        return read_h5ad(file_path)
    elif ext == '.h5':
        return read_h5(file_path)
    elif ext == '.csv':
        return read_csv(file_path)
    elif ext == '.tsv':
        return read_tsv(file_path)
    elif ext == '.txt':
        return read_txt(file_path)
    elif ext == '.json':
        return read_json(file_path)
    elif ext == '.pkl' or ext == '.pickle':
        return read_pickle(file_path)
    elif ext == '.npy':
        return read_npy(file_path)
    elif ext == '.npz':
        return read_npz(file_path)
    elif ext in ['.gz', '.tar.gz', '.tgz', '.zip']:
        return read_compressed_file(file_path)
    else:
        return {"type": "unknown", "error": f"Unsupported file type: {ext}"}

def explore_directory(directory):
    """explore all the files in the directory and return the information"""
    results = {}

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, directory)
            print(f"reading: {rel_path}")
            results[rel_path] = read_file(file_path)

    return results

class MultiView:
    """Multi-dataset viewing utilities for single-cell analysis"""

    def __init__(self, datasets_dir: str = None):
        if datasets_dir is None:
            self.datasets_dir = data_path("datasets")
        else:
            self.datasets_dir = Path(datasets_dir)
        self.datasets = {}

    def load_all_datasets(self):
        """Load all datasets in the directory"""
        if not self.datasets_dir.exists():
            print(f"❌ Directory not found: {self.datasets_dir}")
            return

        for file_path in self.datasets_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.h5ad', '.h5', '.csv', '.tsv']:
                try:
                    info = read_file(str(file_path))
                    self.datasets[str(file_path)] = info
                    print(f"✅ Loaded: {file_path.name}")
                except Exception as e:
                    print(f"❌ Failed to load {file_path.name}: {e}")

    def show_dataset_summary(self):
        """Show summary of all loaded datasets"""
        print("=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)

        for path, info in self.datasets.items():
            print(f"\n📁 {Path(path).name}")
            print(f"   Path: {path}")
            print(f"   Type: {info.get('type', 'Unknown')}")

            if 'shape' in info:
                print(f"   Shape: {info['shape']}")

            if 'error' in info:
                print(f"   ❌ Error: {info['error']}")

        print("=" * 60)

    def compare_datasets(self):
        """Compare multiple datasets"""
        if len(self.datasets) < 2:
            print("⚠️  Need at least 2 datasets to compare")
            return

        print("=" * 60)
        print("DATASET COMPARISON")
        print("=" * 60)

        for path, info in self.datasets.items():
            if 'shape' in info:
                print(f"{Path(path).name}: {info['shape']}")

        print("=" * 60)
