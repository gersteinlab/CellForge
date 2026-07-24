"""Legacy dataset parser used only by the first-generation RAG stack."""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import scanpy as sc
import anndata
from scipy import sparse
from datetime import datetime
import os
import logging
from pathlib import Path
import re

@dataclass
class DatasetMetadata:
    """Structured metadata for single-cell datasets"""
    dataset_name: str
    n_cells: int
    n_features: int
    feature_types: List[str]
    perturbation_types: List[str]
    cell_types: List[str]
    modality: str
    batch_effects: Optional[Dict[str, Any]]
    quality_metrics: Dict[str, float]
    experimental_design: Dict[str, Any]

@dataclass
class DatasetInfo:
    """Structured dataset information"""
    dataset_name: str
    modality: str
    perturbation_type: str
    metadata: Dict[str, Any]
    excel_info: Optional[Dict[str, Any]] = None
    view_info: Optional[Dict[str, Any]] = None

class DataParser:
    """Parser for single-cell datasets with enhanced metadata extraction"""

    def __init__(self, output_path: str = None, excel_path: str = None):
        """
        Initialize the parser

        Args:
            output_path: Path to save output files (default: use existing results directory)
            excel_path: Path to the Excel file containing dataset information (default: use existing documents directory)
        """
        # 使用现有的目录结构，基于项目根目录
        project_root = Path(__file__).parent.parent  # cellforge目录

        if output_path is None:
            # 使用现有的results目录
            self.output_path = project_root / "data" / "results"
        else:
            self.output_path = Path(output_path)

        if excel_path is None:
            # Prefer internal metadata under knowledge_base/papers, then fall back to legacy paths.
            project_repo_root = project_root.parent
            candidates = [
                project_repo_root / "knowledge_base" / "papers" / "dataset info.xlsx",
                project_repo_root / "knowledge_base" / "papers" / "dataset_info.xlsx",
                project_root / "data" / "documents" / "dataset_info.xlsx",
            ]
            self.excel_path = next((p for p in candidates if p.exists()), candidates[0])
        else:
            self.excel_path = Path(excel_path)

        # 不创建新目录，只使用现有目录
        self.excel_data = self._load_excel_metadata()
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the parser"""
        handlers = [logging.StreamHandler()]  # 总是添加控制台输出

        # 如果输出目录存在，添加文件输出
        if self.output_path.exists():
            try:
                handlers.append(logging.FileHandler(self.output_path / 'parser.log'))
            except Exception as e:
                print(f"Warning: Could not create log file: {e}")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        self.logger = logging.getLogger('DataParser')

    def _load_excel_metadata(self) -> pd.DataFrame:
        """Load dataset metadata from Excel file"""
        try:
            return pd.read_excel(self.excel_path)
        except Exception as e:
            print(f"Warning: Could not load Excel metadata: {e}")
            return pd.DataFrame()

    def parse_dataset(self, dataset_name: str, adata: sc.AnnData) -> DatasetInfo:
        """
        Parse dataset and extract comprehensive information

        Args:
            dataset_name: Name of the dataset
            adata: AnnData object containing the dataset

        Returns:
            DatasetInfo object containing structured dataset information
        """
        try:
            # Extract basic information
            modality = self._infer_modality(adata)
            perturbation_type = self._extract_perturbation_type(adata)

            # Extract metadata
            metadata = self._extract_metadata(adata)

            # Get Excel information
            excel_info = self._get_excel_info(dataset_name)

            # Parse view output if exists
            view_info = self._parse_view_output(dataset_name)

            # Create dataset info object
            dataset_info = DatasetInfo(
                dataset_name=dataset_name,
                modality=modality,
                perturbation_type=perturbation_type,
                metadata=metadata,
                excel_info=excel_info,
                view_info=view_info
            )

            # Save outputs
            self._save_raw_output(dataset_name, adata)
            self._save_structured_info(dataset_info)

            return dataset_info

        except Exception as e:
            self._handle_parsing_error(dataset_name, e)
            raise

    def _extract_metadata(self, adata: sc.AnnData) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from the dataset

        Args:
            adata: AnnData object

        Returns:
            Dictionary containing metadata
        """
        metadata = {
            'basic_info': {
                'title': self._extract_title(adata),
                'abstract': self._extract_abstract(adata),
                'organism': self._extract_organism(adata),
                'tissue': self._extract_tissue(adata),
                'disease': self._extract_disease(adata),
                'cell_type': self._extract_cell_type(adata),
                'method': self._extract_method(adata)
            },
            'cell_metadata': self._extract_cell_metadata(adata),
            'feature_metadata': self._extract_feature_metadata(adata),
            'quality_metrics': self._calculate_quality_metrics(adata)
        }
        return metadata

    def _calculate_quality_metrics(self, adata: sc.AnnData) -> Dict[str, Any]:
        """
        Calculate quality metrics for the dataset

        Args:
            adata: AnnData object

        Returns:
            Dictionary containing quality metrics
        """
        return {
            'missing_values': float(adata.X.isna().sum() / adata.X.size),
            'zero_counts': float((adata.X == 0).sum() / adata.X.size),
            'mean_expression': float(adata.X.mean()),
            'std_expression': float(adata.X.std()),
            'cell_counts': {
                'total': adata.n_obs,
                'per_condition': adata.obs['condition'].value_counts().to_dict()
            }
        }

    def _save_structured_info(self, dataset_info: DatasetInfo):
        """
        Save structured dataset information

        Args:
            dataset_info: DatasetInfo object
        """
        try:
            output = {
                'version': '1.0',
                'timestamp': datetime.now().isoformat(),
                'dataset_info': asdict(dataset_info),
                'compatibility': {
                    'scanpy_version': sc.__version__,
                    'pandas_version': pd.__version__
                }
            }

            output_file = self.output_path / f"{dataset_info.dataset_name}_structured_info.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved structured information to {output_file}")
        except Exception as e:
            self.logger.warning(f"Could not save structured information: {e}")
            # 如果无法保存到文件，至少打印到控制台
            print(f"Dataset info: {json.dumps(output, indent=2, ensure_ascii=False)}")

    def _handle_parsing_error(self, dataset_name: str, error: Exception):
        """
        Handle parsing errors with detailed logging

        Args:
            dataset_name: Name of the dataset
            error: Exception that occurred
        """
        try:
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'dataset': dataset_name,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'stack_trace': str(error.__traceback__)
            }

            error_file = self.output_path / f"{dataset_name}_error.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_info, f, indent=2)

            self.logger.error(f"Error parsing dataset {dataset_name}: {str(error)}")
        except Exception as e:
            # 如果无法保存错误文件，至少打印到控制台
            print(f"Error parsing dataset {dataset_name}: {str(error)}")
            print(f"Could not save error log: {e}")

    def _match_column_names(self, adata: sc.AnnData, patterns: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Intelligently match column names to standard terms

        Args:
            adata: AnnData object
            patterns: Dictionary mapping standard terms to possible column names

        Returns:
            Dictionary mapping standard terms to actual column names
        """
        matches = {}
        for standard, possible_names in patterns.items():
            for col in adata.obs.columns:
                if any(re.search(pattern, col, re.IGNORECASE) for pattern in possible_names):
                    matches[standard] = col
                    break
        return matches

    def _parse_view_output(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Parse output from view.py

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary containing parsed view information
        """
        view_file = self.output_path / f"{dataset_name}_view_output.txt"
        if not view_file.exists():
            return None

        try:
            with open(view_file, 'r', encoding='utf-8') as f:
                content = f.read()

            sections = content.split('\n\n')
            view_info = {
                'observation_columns': self._parse_observation_columns(sections[0]),
                'observation_data': self._parse_observation_data(sections[1]),
                'variable_data': self._parse_variable_data(sections[2]),
                'shape': self._parse_shape(sections[3])
            }
            return view_info

        except Exception as e:
            self.logger.warning(f"Error parsing view output for {dataset_name}: {str(e)}")
            return None

    def _infer_feature_types(self, adata: anndata.AnnData) -> List[str]:
        """
        Infer types of features in the dataset

        Args:
            adata: AnnData object

        Returns:
            List of feature types (e.g., genes, proteins, peaks)
        """
        feature_types = []

        # Check for gene expression
        if 'gene' in adata.var.columns:
            feature_types.append('gene_expression')

        # Check for protein expression
        if 'protein' in adata.var.columns:
            feature_types.append('protein_expression')

        # Check for chromatin accessibility
        if 'peak' in adata.var.columns:
            feature_types.append('chromatin_accessibility')

        return feature_types

    def _extract_perturbation_types(self, adata: anndata.AnnData) -> List[str]:
        """
        Extract types of perturbations in the dataset

        Args:
            adata: AnnData object

        Returns:
            List of perturbation types
        """
        perturbation_types = []

        if 'perturbation' in adata.obs.columns:
            perturbations = adata.obs['perturbation'].unique()

            for pert in perturbations:
                if 'KO' in pert or 'knockout' in pert.lower():
                    perturbation_types.append('gene_knockout')
                elif 'KD' in pert or 'knockdown' in pert.lower():
                    perturbation_types.append('gene_knockdown')
                elif 'OE' in pert or 'overexpression' in pert.lower():
                    perturbation_types.append('gene_overexpression')
                elif any(drug in pert.lower() for drug in ['drug', 'compound', 'treatment']):
                    perturbation_types.append('drug_treatment')
                elif any(cyt in pert.lower() for cyt in ['cytokine', 'stimulus']):
                    perturbation_types.append('cytokine_stimulation')

        return perturbation_types

    def _extract_cell_types(self, adata: anndata.AnnData) -> List[str]:
        """
        Extract cell types from the dataset

        Args:
            adata: AnnData object

        Returns:
            List of cell types
        """
        if 'cell_type' in adata.obs.columns:
            return list(adata.obs['cell_type'].unique())
        return []

    def _infer_modality(self, adata) -> str:
        """Infer data modality"""
        # Check for protein expression
        if 'protein' in adata.var.columns:
            return 'CITE-seq'
        # Check for chromatin accessibility
        elif 'peak' in adata.var.columns:
            return 'ATAC-seq'
        else:
            return 'RNA'

    def _analyze_batch_effects(self, adata: anndata.AnnData) -> Dict[str, Any]:
        """
        Analyze potential batch effects in the dataset

        Args:
            adata: AnnData object

        Returns:
            Dictionary containing batch effect analysis
        """
        batch_effects = {}

        if 'batch' in adata.obs.columns:
            # Calculate batch-specific metrics
            for batch in adata.obs['batch'].unique():
                batch_data = adata[adata.obs['batch'] == batch]
                batch_effects[batch] = {
                    'n_cells': len(batch_data),
                    'mean_expression': float(batch_data.X.mean()),
                    'n_genes': int((batch_data.X > 0).sum(axis=1).mean())
                }

        return batch_effects

    def _extract_experimental_design(self, adata: anndata.AnnData) -> Dict[str, Any]:
        """
        Extract experimental design information

        Args:
            adata: AnnData object

        Returns:
            Dictionary containing experimental design details
        """
        design = {}

        # Extract time points if available
        if 'time' in adata.obs.columns:
            design['time_points'] = list(adata.obs['time'].unique())

        # Extract replicates if available
        if 'replicate' in adata.obs.columns:
            design['n_replicates'] = len(adata.obs['replicate'].unique())

        # Extract experimental conditions
        if 'condition' in adata.obs.columns:
            design['conditions'] = list(adata.obs['condition'].unique())

        return design

    def _perform_quality_control(self, adata: anndata.AnnData) -> None:
        """
        Perform quality control on the dataset

        Args:
            adata: AnnData object
        """
        # Calculate quality metrics
        sc.pp.calculate_qc_metrics(adata, inplace=True)

        # Filter cells based on quality thresholds
        sc.pp.filter_cells(adata, min_genes=self.quality_thresholds['min_genes_per_cell'])
        sc.pp.filter_genes(adata, min_cells=self.quality_thresholds['min_cells_per_gene'])

        # Normalize data
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    def _save_metadata(self, metadata: DatasetMetadata, dataset_name: str) -> None:
        """
        Save metadata to JSON file

        Args:
            metadata: DatasetMetadata object
            dataset_name: Name of the dataset
        """
        metadata_dict = {
            'dataset_name': metadata.dataset_name,
            'n_cells': metadata.n_cells,
            'n_features': metadata.n_features,
            'feature_types': metadata.feature_types,
            'perturbation_types': metadata.perturbation_types,
            'cell_types': metadata.cell_types,
            'modality': metadata.modality,
            'batch_effects': metadata.batch_effects,
            'quality_metrics': metadata.quality_metrics,
            'experimental_design': metadata.experimental_design
        }

        with open(f"{dataset_name}_metadata.json", "w") as f:
            json.dump(metadata_dict, f, indent=4)

    def _extract_perturbation_type(self, adata) -> str:
        """Extract perturbation type"""
        if 'perturbation_type' in adata.obs.columns:
            return adata.obs['perturbation_type'].iloc[0]
        return "Unknown"

    def _extract_title(self, adata) -> str:
        """Extract dataset title"""
        # Try Excel first
        if self.excel_data.empty:
            return "Unknown"
        if 'title' in adata.uns:
            return adata.uns['title']
        return "Unknown"

    def _extract_abstract(self, adata) -> str:
        """Extract dataset abstract"""
        # Try Excel first
        if self.excel_data.empty:
            return "Unknown"
        if 'abstract' in adata.uns:
            return adata.uns['abstract']
        return "Unknown"

    def _extract_organism(self, adata) -> str:
        """Extract organism information"""
        # Try Excel first
        if self.excel_data.empty:
            return "Unknown"
        if 'organism' in adata.obs.columns:
            return adata.obs['organism'].iloc[0]
        return "Unknown"

    def _extract_tissue(self, adata) -> str:
        """Extract tissue information"""
        # Try Excel first
        if self.excel_data.empty:
            return "Unknown"
        if 'tissue_type' in adata.obs.columns:
            return adata.obs['tissue_type'].iloc[0]
        return "Unknown"

    def _extract_disease(self, adata) -> str:
        """Extract disease information"""
        # Try Excel first
        if self.excel_data.empty:
            return "Unknown"
        if 'disease' in adata.obs.columns:
            return adata.obs['disease'].iloc[0]
        return "Unknown"

    def _extract_cell_type(self, adata) -> str:
        """Extract cell type information"""
        # Try Excel first
        if self.excel_data.empty:
            return "Unknown"
        if 'celltype' in adata.obs.columns:
            return adata.obs['celltype'].iloc[0]
        return "Unknown"

    def _extract_method(self, adata) -> str:
        """Extract experimental method"""
        # Try Excel first
        if self.excel_data.empty:
            return "Unknown"
        if 'method' in adata.uns:
            return adata.uns['method']
        return "Unknown"

    def _extract_cell_metadata(self, adata) -> Dict[str, Any]:
        """Extract cell metadata statistics"""
        metadata = {}

        # Extract basic statistics
        metadata['n_cells'] = adata.n_obs
        metadata['obs_columns'] = list(adata.obs.columns)

        # Extract statistics for numeric columns
        numeric_cols = adata.obs.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            metadata[col] = {
                'mean': float(adata.obs[col].mean()),
                'std': float(adata.obs[col].std()),
                'min': float(adata.obs[col].min()),
                'max': float(adata.obs[col].max())
            }

        # Extract value counts for categorical columns
        categorical_cols = adata.obs.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            metadata[col] = adata.obs[col].value_counts().to_dict()

        return metadata

    def _extract_feature_metadata(self, adata) -> Dict[str, Any]:
        """Extract feature metadata statistics"""
        metadata = {}

        # Extract basic statistics
        metadata['n_features'] = adata.n_vars
        metadata['var_columns'] = list(adata.var.columns)

        # Extract feature names
        metadata['feature_names'] = list(adata.var_names)

        return metadata

    def _save_raw_output(self, dataset_name: str, adata) -> None:
        """Save raw print output"""
        try:
            output = []

            # Add obs columns
            output.append("obs columns\n")
            output.append(str(adata.obs.columns))
            output.append("\n---\n")

            # Add obs head
            output.append("obs\n")
            output.append(str(adata.obs.head()))
            output.append("\n---\n")

            # Add var head
            output.append("var\n")
            output.append(str(adata.var.head()))
            output.append("\n---\n")

            # Add shape
            output.append("shape\n")
            output.append(str(adata.shape))

            # Save to file
            output_path = self.output_path / f"{dataset_name}_raw_output.txt"
            with open(output_path, "w") as f:
                f.write("\n".join(output))
        except Exception as e:
            self.logger.warning(f"Could not save raw output for {dataset_name}: {e}")
            # 如果无法保存到文件，至少打印到控制台
            print(f"Raw output for {dataset_name}:")
            print("\n".join(output))

    def _get_excel_info(self, dataset_name: str) -> Dict[str, Any]:
        """Extract information from Excel for specific dataset"""
        if self.excel_data.empty:
            return {}

        # Try to find matching row in info.xlsx
        # You might need to adjust the matching logic based on your Excel structure
        mask = self.excel_data['dataset_name'].str.contains(dataset_name, case=False, na=False)
        if mask.any():
            row = self.excel_data[mask].iloc[0]
            return row.to_dict()
        return {}
