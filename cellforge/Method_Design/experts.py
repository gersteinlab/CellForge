from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging
import warnings
from ..llm import LLMInterface
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpertDomain(Enum):
    DATA_ENGINEERING = "data_engineering"
    SINGLE_CELL_BIOLOGY = "single_cell_biology"
    DEEP_LEARNING = "deep_learning"
    MOLECULAR_BIOLOGY = "molecular_biology"
    BIOINFORMATICS = "bioinformatics"
    STATISTICS = "statistics"
    DRUG_DISCOVERY = "drug_discovery"
    GENE_REGULATION = "gene_regulation"
    EXPERIMENTAL_DESIGN = "experimental_design"
    VALIDATION = "validation"
    BASELINE_ASSESSMENT = "baseline_assessment"
    TRAINING = "training"
    DRUG_RESPONSE = "drug_response"
    PATHWAY = "pathway"
    CELL_COMMUNICATION = "cell_communication"
    OMICS = "omics"

@dataclass
class Expert:
    name: str
    domain: ExpertDomain
    expertise_level: float
    confidence: float = 0.0
    knowledge_base: Dict[str, Any] = None
    constraints: List[str] = None
    skills: List[str] = None
    background: str = ""
    specializations: List[str] = None
    tools: Dict[str, List[str]] = None

    def __post_init__(self):
        if self.knowledge_base is None:
            self.knowledge_base = {}
        if self.constraints is None:
            self.constraints = []
        if self.skills is None:
            self.skills = []
        if self.specializations is None:
            self.specializations = []
        if self.tools is None:
            self.tools = {}

class ExpertPool:
    def __init__(self):
        self.experts = self._initialize_experts()

    def _initialize_experts(self) -> List[Expert]:
        return [
            Expert(
                name="Data Engineer",
                domain=ExpertDomain.DATA_ENGINEERING,
                expertise_level=0.95,
                background="Distinguished Professor with 50+ years of experience in computational biology and single-cell data analysis, pioneer in high-dimensional data processing",
                skills=[
                    "High-dimensional data preprocessing",
                    "Quality control and filtering",
                    "Batch effect correction",
                    "Feature selection and engineering",
                    "Dimensionality reduction",
                    "Data integration",
                    "Performance optimization"
                ],
                specializations=[
                    "Single-cell RNA-seq",
                    "Single-cell ATAC-seq",
                    "CITE-seq",
                    "Perturbation data processing",
                    "Multi-modal integration"
                ],
                constraints=[
                    "Requires raw count matrix",
                    "Needs comprehensive metadata",
                    "Requires quality metrics",
                    "Needs batch information",
                    "Requires cell type annotations"
                ],
                knowledge_base={
                    "data_types": {
                        "scRNA_seq": {
                            "format": [".h5ad", ".loom", ".mtx"],
                            "required_fields": ["counts", "var_names", "obs_names"],
                            "optional_fields": ["layers", "obsm", "varm"]
                        },
                        "scATAC_seq": {
                            "format": [".h5ad", ".mtx"],
                            "required_fields": ["counts", "var_names", "obs_names"],
                            "optional_fields": ["peaks", "fragments"]
                        },
                        "CITE_seq": {
                            "format": [".h5ad"],
                            "required_fields": ["counts", "protein_expression"],
                            "optional_fields": ["surface_proteins"]
                        }
                    },
                    "preprocessing_pipeline": {
                        "quality_control": {
                            "cell_filtering": {
                                "min_genes": 200,
                                "max_genes": 6000,
                                "min_counts": 1000,
                                "max_counts": 50000,
                                "max_mito_percent": 20
                            },
                            "gene_filtering": {
                                "min_cells": 3,
                                "min_counts": 10
                            },
                            "doublet_detection": {
                                "expected_doublet_rate": 0.1,
                                "min_counts": 1000
                            }
                        },
                        "normalization": {
                            "total_counts": 10000,
                            "log_transform": True,
                            "regression_vars": ["total_counts", "pct_counts_mt"]
                        },
                        "feature_selection": {
                            "highly_variable": {
                                "min_mean": 0.0125,
                                "max_mean": 3,
                                "min_disp": 0.5
                            },
                            "scaling": {
                                "max_value": 10,
                                "zero_center": True
                            }
                        },
                        "batch_correction": {
                            "methods": ["regress_out", "combat", "bbknn", "harmony"],
                            "parameters": {
                                "n_neighbors": 10,
                                "batch_key": "batch"
                            }
                        },
                        "dimensionality_reduction": {
                            "pca": {
                                "n_comps": 50,
                                "svd_solver": "arpack"
                            },
                            "umap": {
                                "n_neighbors": 15,
                                "min_dist": 0.5,
                                "metric": "euclidean"
                            },
                            "tsne": {
                                "n_pcs": 40,
                                "perplexity": 30
                            }
                        }
                    },
                    "perturbation_processing": {
                        "data_requirements": {
                            "metadata": [
                                "perturbation_type",
                                "perturbation_target",
                                "time_point",
                                "cell_type",
                                "batch_info",
                                "replicate_id"
                            ],
                            "quality_metrics": [
                                "n_genes_by_counts",
                                "total_counts",
                                "pct_counts_mt",
                                "doublet_score",
                                "cell_cycle_score"
                            ]
                        },
                        "analysis_steps": {
                            "differential_expression": {
                                "method": "wilcoxon",
                                "min_cells": 10,
                                "correction": "fdr"
                            },
                            "trajectory_analysis": {
                                "method": "paga",
                                "resolution": 1.0
                            },
                            "cell_type_annotation": {
                                "method": "leiden",
                                "resolution": 1.0
                            }
                        }
                    }
                },
                tools={
                    "python_packages": [
                        "scanpy",
                        "anndata",
                        "pandas",
                        "numpy",
                        "scipy",
                        "scikit-learn",
                        "torch",
                        "torch_geometric"
                    ],
                    "data_processing": {
                        "quality_control": [
                            "scanpy.pp.calculate_qc_metrics",
                            "scanpy.pp.filter_cells",
                            "scanpy.pp.filter_genes",
                            "scanpy.pp.scrublet"
                        ],
                        "normalization": [
                            "scanpy.pp.normalize_total",
                            "scanpy.pp.log1p",
                            "scanpy.pp.regress_out",
                            "scanpy.pp.scale"
                        ],
                        "feature_selection": [
                            "scanpy.pp.highly_variable_genes",
                            "scanpy.pp.scale",
                            "scanpy.pp.pca"
                        ],
                        "batch_correction": [
                            "scanpy.pp.regress_out",
                            "scanpy.pp.combat",
                            "scanpy.pp.bbknn",
                            "scanpy.pp.harmony"
                        ],
                        "dimensionality_reduction": [
                            "scanpy.pp.pca",
                            "scanpy.pp.neighbors",
                            "scanpy.tl.umap",
                            "scanpy.tl.tsne"
                        ]
                    },
                    "visualization": {
                        "quality_control": [
                            "scanpy.pl.violin",
                            "scanpy.pl.scatter",
                            "scanpy.pl.highest_expr_genes"
                        ],
                        "dimensionality_reduction": [
                            "scanpy.pl.pca",
                            "scanpy.pl.umap",
                            "scanpy.pl.tsne"
                        ],
                        "differential_expression": [
                            "scanpy.pl.rank_genes_groups",
                            "scanpy.pl.heatmap",
                            "scanpy.pl.stacked_violin"
                        ],
                        "trajectory": [
                            "scanpy.pl.paga",
                            "scanpy.pl.dpt",
                            "scanpy.pl.diffmap"
                        ]
                    },
                    "perturbation_analysis": {
                        "differential_expression": [
                            "scanpy.tl.rank_genes_groups",
                            "scanpy.tl.dendrogram",
                            "scanpy.tl.filter_rank_genes_groups"
                        ],
                        "trajectory_analysis": [
                            "scanpy.tl.paga",
                            "scanpy.tl.dpt",
                            "scanpy.tl.diffmap"
                        ],
                        "cell_type_annotation": [
                            "scanpy.tl.leiden",
                            "scanpy.tl.louvain",
                            "scanpy.tl.rank_genes_groups"
                        ]
                    },
                    "performance_optimization": {
                        "memory_optimization": [
                            "anndata.AnnData.sparse",
                            "pandas.DataFrame.astype",
                            "numpy.memmap"
                        ],
                        "computation_optimization": [
                            "numpy.vectorize",
                            "numba.jit",
                            "torch.cuda"
                        ]
                    }
                }
            ),
            Expert(
                name="Single Cell Biology Expert",
                domain=ExpertDomain.SINGLE_CELL_BIOLOGY,
                expertise_level=0.95,
                background="World-renowned Professor with 40+ years of experience in single-cell biology, pioneer in cellular responses and perturbation analysis, expert in biological interpretability",
                skills=[
                    "Cell type identification and annotation",
                    "Differential expression analysis",
                    "Pathway and gene set enrichment",
                    "Cell state transition analysis",
                    "Cell-cell communication inference",
                    "Trajectory analysis",
                    "Biological interpretability",
                    "Perturbation response analysis"
                ],
                specializations=[
                    "Immune cell biology",
                    "Cancer cell biology",
                    "Stem cell differentiation",
                    "Cellular responses to perturbations",
                    "Single-cell multi-omics",
                    "Biological mechanism elucidation"
                ],
                constraints=[
                    "Requires cell type annotations",
                    "Needs biological context",
                    "Requires pathway information",
                    "Needs cell state markers",
                    "Requires perturbation metadata",
                    "Needs mechanistic validation"
                ],
                knowledge_base={
                    "cell_types": {
                        "immune_cells": {
                            "T_cells": {
                                "markers": ["CD3D", "CD4", "CD8A", "FOXP3", "IL2RA"],
                                "subtypes": ["CD4+", "CD8+", "Treg", "Memory T", "Naive T"],
                                "perturbation_responses": {
                                    "cytokine": ["IL2", "IFNg", "TNFa"],
                                    "drug": ["Cyclosporin", "Rapamycin", "Tacrolimus"]
                                }
                            },
                            "B_cells": {
                                "markers": ["CD19", "MS4A1", "CD27", "IGHM", "IGHD"],
                                "subtypes": ["Naive B", "Memory B", "Plasma cells"],
                                "perturbation_responses": {
                                    "cytokine": ["IL4", "IL21", "BAFF"],
                                    "drug": ["Rituximab", "Ibrutinib", "Venetoclax"]
                                }
                            }
                        },
                        "cancer_cells": {
                            "tumor_cells": {
                                "markers": ["EPCAM", "KRT8", "KRT18"],
                                "subtypes": ["Epithelial", "Mesenchymal", "Stem-like"],
                                "perturbation_responses": {
                                    "drug": ["Cisplatin", "Paclitaxel", "Doxorubicin"],
                                    "targeted": ["EGFRi", "BRAFi", "PARPi"]
                                }
                            }
                        }
                    },
                    "pathways": {
                        "signaling": {
                            "MAPK": {
                                "components": ["RAS", "RAF", "MEK", "ERK"],
                                "perturbation_effects": {
                                    "inhibition": ["Cell cycle arrest", "Apoptosis"],
                                    "activation": ["Proliferation", "Differentiation"]
                                }
                            },
                            "PI3K_AKT": {
                                "components": ["PI3K", "AKT", "mTOR"],
                                "perturbation_effects": {
                                    "inhibition": ["Metabolic stress", "Autophagy"],
                                    "activation": ["Growth", "Survival"]
                                }
                            }
                        },
                        "stress_response": {
                            "heat_shock": {
                                "components": ["HSP70", "HSP90", "HSF1"],
                                "perturbation_effects": {
                                    "induction": ["Protein folding", "Cell survival"],
                                    "inhibition": ["Protein aggregation", "Cell death"]
                                }
                            },
                            "oxidative": {
                                "components": ["Nrf2", "KEAP1", "HIF1a"],
                                "perturbation_effects": {
                                    "induction": ["Antioxidant response", "Metabolic adaptation"],
                                    "inhibition": ["ROS accumulation", "Cell death"]
                                }
                            }
                        }
                    },
                    "perturbation_mechanisms": {
                        "gene_knockout": {
                            "direct_effects": {
                                "transcription": ["Expression changes", "Splicing alterations"],
                                "protein": ["Stability changes", "Function loss"],
                                "pathway": ["Network rewiring", "Compensation"]
                            },
                            "indirect_effects": {
                                "feedback": ["Negative feedback", "Positive feedback"],
                                "crosstalk": ["Pathway crosstalk", "Compensatory pathways"]
                            }
                        },
                        "drug_treatment": {
                            "target_effects": {
                                "inhibition": ["Competitive", "Allosteric", "Covalent"],
                                "activation": ["Agonist", "Partial agonist", "Allosteric modulator"]
                            },
                            "off_target": {
                                "binding": ["Secondary targets", "Metabolites"],
                                "pathways": ["Related pathways", "Compensatory mechanisms"]
                            }
                        },
                        "cytokine_stimulation": {
                            "signaling": {
                                "receptor": ["Dimerization", "Phosphorylation", "Internalization"],
                                "cascade": ["Kinase activation", "Transcription factor activation"]
                            },
                            "response": {
                                "acute": ["Immediate response", "Signal amplification"],
                                "chronic": ["Sustained response", "Adaptation"]
                            }
                        }
                    }
                },
                tools={
                    "python_packages": [
                        "scanpy",
                        "gseapy",
                        "mygene",
                        "cellphonedb",
                        "omnipath",
                        "scvelo",
                        "cellrank"
                    ],
                    "analysis_tools": {
                        "cell_type_annotation": [
                            "scanpy.tl.rank_genes_groups",
                            "scanpy.tl.leiden",
                            "scanpy.tl.louvain"
                        ],
                        "pathway_analysis": [
                            "gseapy.enrichr",
                            "gseapy.prerank",
                            "gseapy.gsea"
                        ],
                        "trajectory_analysis": [
                            "scvelo.tl.velocity",
                            "cellrank.tl.terminal_states",
                            "scanpy.tl.paga"
                        ],
                        "cell_cell_communication": [
                            "cellphonedb.cpdb_statistical_analysis",
                            "omnipath.interactions"
                        ]
                    },
                    "databases": {
                        "pathway_databases": [
                            "KEGG",
                            "Reactome",
                            "GO",
                            "MSigDB"
                        ],
                        "interaction_databases": [
                            "STRING",
                            "CellPhoneDB",
                            "OmniPath"
                        ]
                    },
                    "visualization": {
                        "cell_type": [
                            "scanpy.pl.umap",
                            "scanpy.pl.violin",
                            "scanpy.pl.stacked_violin"
                        ],
                        "pathways": [
                            "gseapy.barplot",
                            "gseapy.heatmap",
                            "gseapy.enrichr"
                        ],
                        "trajectory": [
                            "scvelo.pl.velocity_embedding",
                            "cellrank.pl.terminal_states",
                            "scanpy.pl.paga"
                        ]
                    }
                }
            ),
            Expert(
                name="Single Cell Model Architect",
                domain=ExpertDomain.SINGLE_CELL_BIOLOGY,
                expertise_level=0.95,
                background="Distinguished Professor with 40+ years of experience in single-cell modeling, pioneer in perturbation response prediction and cellular dynamics modeling",
                skills=[
                    "Perturbation response modeling",
                    "Cell state transition prediction",
                    "Gene regulatory network inference",
                    "Multi-modal integration",
                    "Temporal dynamics modeling",
                    "Cell-cell communication modeling",
                    "Biological constraint integration"
                ],
                specializations=[
                    "Perturbation prediction",
                    "Cell state dynamics",
                    "Regulatory network modeling",
                    "Multi-modal integration",
                    "Biological interpretability"
                ],
                constraints=[
                    "Requires perturbation data",
                    "Needs cell type annotations",
                    "Requires temporal information",
                    "Needs biological constraints",
                    "Requires validation data"
                ],
                knowledge_base={
                    "perturbation_modeling": {
                        "response_types": {
                            "gene_knockout": {
                                "direct_effects": [
                                    "Gene expression changes",
                                    "Pathway alterations",
                                    "Network rewiring"
                                ],
                                "indirect_effects": [
                                    "Feedback loops",
                                    "Compensatory mechanisms",
                                    "Crosstalk"
                                ]
                            },
                            "drug_treatment": {
                                "response_patterns": [
                                    "Dose-dependent",
                                    "Time-dependent",
                                    "Cell-type specific"
                                ],
                                "mechanisms": [
                                    "Target inhibition",
                                    "Pathway modulation",
                                    "Off-target effects"
                                ]
                            },
                            "cytokine_stimulation": {
                                "signaling_cascades": [
                                    "JAK-STAT",
                                    "MAPK",
                                    "NF-kB"
                                ],
                                "temporal_patterns": [
                                    "Acute response",
                                    "Chronic response",
                                    "Oscillatory response"
                                ]
                            }
                        },
                        "model_components": {
                            "encoder": {
                                "types": [
                                    "Transformer",
                                    "Graph neural network",
                                    "Autoencoder"
                                ],
                                "features": [
                                    "Gene expression",
                                    "Cell type",
                                    "Perturbation type"
                                ]
                            },
                            "decoder": {
                                "types": [
                                    "Conditional generation",
                                    "Trajectory prediction",
                                    "Response prediction"
                                ],
                                "outputs": [
                                    "Expression profiles",
                                    "Cell states",
                                    "Pathway activities"
                                ]
                            },
                            "constraints": {
                                "biological": [
                                    "Gene regulatory networks",
                                    "Pathway interactions",
                                    "Cell type specificity"
                                ],
                                "physical": [
                                    "Mass conservation",
                                    "Energy constraints",
                                    "Rate limitations"
                                ]
                            }
                        }
                    }
                },
                tools={
                    "python_packages": [
                        "torch",
                        "torch_geometric",
                        "scanpy",
                        "scvelo",
                        "cellrank",
                        "omnipath",
                        "gseapy"
                    ],
                    "model_components": {
                        "neural_networks": [
                            "torch.nn.Transformer",
                            "torch_geometric.nn.GATConv",
                            "torch.nn.LSTM"
                        ],
                        "biological_constraints": [
                            "gseapy.enrichr",
                            "omnipath.interactions",
                            "scanpy.tl.rank_genes_groups"
                        ],
                        "evaluation": [
                            "scvelo.tl.velocity",
                            "cellrank.tl.terminal_states",
                            "scanpy.tl.paga"
                        ]
                    },
                    "visualization": {
                        "perturbation": [
                            "scanpy.pl.heatmap",
                            "scanpy.pl.violin",
                            "scanpy.pl.umap"
                        ],
                        "dynamics": [
                            "scvelo.pl.velocity_embedding",
                            "cellrank.pl.terminal_states",
                            "scanpy.pl.paga"
                        ],
                        "networks": [
                            "scanpy.pl.rank_genes_groups",
                            "scanpy.pl.matrixplot",
                            "scanpy.pl.stacked_violin"
                        ]
                    }
                }
            ),
            Expert(
                name="Deep Learning Expert",
                domain=ExpertDomain.DEEP_LEARNING,
                expertise_level=0.95,
                background="Distinguished Professor of Computer Science with 35+ years of experience in deep learning, pioneer in biological sequence modeling and perturbation prediction",
                skills=[
                    "Neural architecture design",
                    "Model optimization",
                    "Transfer learning",
                    "Multi-modal learning",
                    "Attention mechanisms",
                    "Graph neural networks",
                    "Transformer models",
                    "Interpretable AI"
                ],
                specializations=[
                    "Perturbation prediction",
                    "Cell state modeling",
                    "Multi-modal integration",
                    "Biological sequence modeling",
                    "Interpretable deep learning"
                ],
                constraints=[
                    "Requires GPU resources",
                    "Needs large datasets",
                    "Requires clear performance metrics",
                    "Needs biological validation",
                    "Requires interpretability"
                ],
                knowledge_base={
                    "architectures": {
                        "perturbation_models": {
                            "transformer": {
                                "components": [
                                    "Multi-head attention",
                                    "Position-wise feed-forward",
                                    "Layer normalization",
                                    "Residual connections"
                                ],
                                "variants": [
                                    "BERT",
                                    "GPT",
                                    "T5",
                                    "Performer"
                                ]
                            },
                            "graph_neural_networks": {
                                "types": [
                                    "GCN",
                                    "GAT",
                                    "GraphSAGE",
                                    "GIN"
                                ],
                                "applications": [
                                    "Gene regulatory networks",
                                    "Cell-cell communication",
                                    "Pathway interactions"
                                ]
                            },
                            "autoencoder": {
                                "types": [
                                    "Vanilla AE",
                                    "VAE",
                                    "Denoising AE",
                                    "Contractive AE"
                                ],
                                "applications": [
                                    "Dimensionality reduction",
                                    "Feature learning",
                                    "Data denoising"
                                ]
                            }
                        },
                        "optimization": {
                            "loss_functions": {
                                "reconstruction": [
                                    "MSE",
                                    "MAE",
                                    "Huber"
                                ],
                                "biological": [
                                    "Pathway consistency",
                                    "Network structure",
                                    "Cell type specificity"
                                ],
                                "regularization": [
                                    "L1",
                                    "L2",
                                    "Dropout"
                                ]
                            },
                            "training_strategies": {
                                "transfer_learning": [
                                    "Pre-training",
                                    "Fine-tuning",
                                    "Domain adaptation"
                                ],
                                "multi_task": [
                                    "Shared encoder",
                                    "Task-specific heads",
                                    "Gradient balancing"
                                ],
                                "curriculum": [
                                    "Difficulty progression",
                                    "Sample weighting",
                                    "Dynamic batching"
                                ]
                            }
                        }
                    }
                },
                tools={
                    "python_packages": [
                        "torch",
                        "torch_geometric",
                        "pytorch_lightning",
                        "wandb",
                        "optuna",
                        "captum",
                        "torchmetrics"
                    ],
                    "model_components": {
                        "transformer": [
                            "torch.nn.TransformerEncoder",
                            "torch.nn.TransformerDecoder",
                            "torch.nn.MultiheadAttention"
                        ],
                        "graph_neural_networks": [
                            "torch_geometric.nn.GATConv",
                            "torch_geometric.nn.GCNConv",
                            "torch_geometric.nn.GraphSAGE"
                        ],
                        "autoencoder": [
                            "torch.nn.Linear",
                            "torch.nn.Conv1d",
                            "torch.nn.BatchNorm1d"
                        ]
                    },
                    "optimization": {
                        "optimizers": [
                            "torch.optim.Adam",
                            "torch.optim.AdamW",
                            "torch.optim.SGD"
                        ],
                        "schedulers": [
                            "torch.optim.lr_scheduler.CosineAnnealingLR",
                            "torch.optim.lr_scheduler.ReduceLROnPlateau"
                        ],
                        "regularization": [
                            "torch.nn.Dropout",
                            "torch.nn.BatchNorm1d",
                            "torch.nn.LayerNorm"
                        ]
                    },
                    "interpretability": {
                        "attribution": [
                            "captum.attr.IntegratedGradients",
                            "captum.attr.Saliency",
                            "captum.attr.GuidedGradCam"
                        ],
                        "visualization": [
                            "captum.attr.visualization",
                            "torch.utils.tensorboard",
                            "matplotlib.pyplot"
                        ]
                    }
                }
            ),
            Expert(
                name="Molecular Biology Specialist",
                domain=ExpertDomain.MOLECULAR_BIOLOGY,
                expertise_level=0.95,
                background="PhD in Molecular Biology with extensive experience in gene regulation and cellular signaling",
                skills=[
                    "Gene expression analysis",
                    "Pathway mapping and analysis",
                    "Network analysis",
                    "Molecular mechanism elucidation",
                    "Protein-DNA interactions",
                    "Epigenetic regulation",
                    "Signal transduction"
                ],
                specializations=[
                    "Transcriptional regulation",
                    "Signal transduction",
                    "Epigenetics",
                    "Cellular responses",
                    "Molecular networks"
                ],
                constraints=[
                    "Requires gene annotations",
                    "Needs pathway information",
                    "Requires experimental validation",
                    "Needs molecular context",
                    "Requires interaction data"
                ],
                knowledge_base={
                    "pathways": {
                        "signaling": {
                            "MAPK": {
                                "components": ["RAS", "RAF", "MEK", "ERK"],
                                "functions": ["Proliferation", "Differentiation", "Survival"],
                                "regulators": ["Growth factors", "Cytokines", "Stress"]
                            },
                            "PI3K_AKT": {
                                "components": ["PI3K", "AKT", "mTOR"],
                                "functions": ["Metabolism", "Growth", "Survival"],
                                "regulators": ["Insulin", "Growth factors", "Nutrients"]
                            },
                            "JAK_STAT": {
                                "components": ["JAK", "STAT", "SOCS"],
                                "functions": ["Immune response", "Inflammation", "Development"],
                                "regulators": ["Cytokines", "Interferons", "Growth factors"]
                            },
                            "WNT": {
                                "components": ["Frizzled", "LRP", "beta-catenin"],
                                "functions": ["Development", "Stemness", "Differentiation"],
                                "regulators": ["WNT ligands", "DKK", "SFRP"]
                            },
                            "TGF_beta": {
                                "components": ["TGFBR", "SMAD", "TGFB"],
                                "functions": ["Differentiation", "Apoptosis", "EMT"],
                                "regulators": ["TGFB ligands", "BMP", "Activin"]
                            }
                        },
                        "transcription": {
                            "general": {
                                "components": ["RNA Pol II", "GTF", "Mediator"],
                                "regulation": ["Promoters", "Enhancers", "Silencers"],
                                "modifications": ["Histone marks", "DNA methylation"]
                            },
                            "epigenetic": {
                                "marks": ["H3K4me3", "H3K27ac", "H3K27me3"],
                                "enzymes": ["HDAC", "HAT", "DNMT"],
                                "functions": ["Gene activation", "Gene repression"]
                            }
                        },
                        "metabolism": {
                            "glycolysis": {
                                "enzymes": ["HK", "PFK", "PK"],
                                "regulation": ["HIF1a", "MYC", "AMPK"],
                                "products": ["ATP", "NADH", "Pyruvate"]
                            },
                            "TCA_cycle": {
                                "enzymes": ["CS", "IDH", "OGDH"],
                                "regulation": ["Calcium", "NAD+/NADH", "ATP/ADP"],
                                "products": ["ATP", "NADH", "FADH2"]
                            }
                        }
                    },
                    "perturbation_mechanisms": {
                        "gene_knockout": {
                            "direct_effects": [
                                "Loss of protein function",
                                "Altered protein interactions",
                                "Compensatory mechanisms"
                            ],
                            "indirect_effects": [
                                "Pathway rewiring",
                                "Feedback loops",
                                "Crosstalk"
                            ]
                        },
                        "drug_treatment": {
                            "target_interaction": [
                                "Direct binding",
                                "Allosteric modulation",
                                "Competitive inhibition"
                            ],
                            "downstream_effects": [
                                "Pathway inhibition",
                                "Signal transduction",
                                "Gene expression"
                            ]
                        },
                        "cytokine_stimulation": {
                            "receptor_activation": [
                                "Ligand binding",
                                "Receptor dimerization",
                                "Kinase activation"
                            ],
                            "signal_transduction": [
                                "JAK-STAT activation",
                                "MAPK cascade",
                                "NF-kB pathway"
                            ]
                        }
                    }
                },
                tools={
                    "python_packages": [
                        "networkx",
                        "gseapy",
                        "mygene",
                        "biopython",
                        "pandas",
                        "numpy"
                    ],
                    "analysis_tools": {
                        "pathway_analysis": [
                            "gseapy.enrichr",
                            "gseapy.prerank",
                            "gseapy.gsea"
                        ],
                        "network_analysis": [
                            "networkx.Graph",
                            "networkx.algorithms.centrality",
                            "networkx.algorithms.community"
                        ],
                        "sequence_analysis": [
                            "biopython.Seq",
                            "biopython.SeqRecord",
                            "biopython.SeqIO"
                        ]
                    },
                    "databases": {
                        "pathway_databases": [
                            "KEGG",
                            "Reactome",
                            "WikiPathways",
                            "MSigDB"
                        ],
                        "interaction_databases": [
                            "STRING",
                            "BioGRID",
                            "IntAct",
                            "MINT"
                        ],
                        "transcription_databases": [
                            "JASPAR",
                            "TRANSFAC",
                            "ENCODE",
                            "Roadmap"
                        ]
                    },
                    "visualization": {
                        "pathways": [
                            "networkx.draw",
                            "matplotlib.pyplot",
                            "seaborn"
                        ],
                        "networks": [
                            "networkx.draw_networkx",
                            "networkx.draw_spring",
                            "networkx.draw_circular"
                        ],
                        "expression": [
                            "seaborn.heatmap",
                            "seaborn.clustermap",
                            "matplotlib.pyplot"
                        ]
                    }
                }
            ),
            Expert(
                name="Bioinformatician",
                domain=ExpertDomain.BIOINFORMATICS,
                expertise_level=0.95,
                background="PhD in Bioinformatics with extensive experience in computational biology and single-cell analysis",
                skills=[
                    "Sequence analysis",
                    "Network inference",
                    "Statistical modeling",
                    "Machine learning",
                    "Data integration",
                    "Algorithm development",
                    "Performance optimization"
                ],
                specializations=[
                    "Gene regulatory networks",
                    "Pathway analysis",
                    "Single-cell analysis",
                    "Multi-omics integration",
                    "Computational modeling"
                ],
                constraints=[
                    "Requires computational resources",
                    "Needs biological validation",
                    "Requires high-quality data",
                    "Needs clear objectives",
                    "Requires performance metrics"
                ],
                knowledge_base={
                    "algorithms": {
                        "network_inference": {
                            "methods": [
                                "GENIE3",
                                "GRNBoost2",
                                "SCENIC",
                                "PIDC"
                            ],
                            "metrics": [
                                "AUPR",
                                "AUROC",
                                "Precision@k"
                            ]
                        },
                        "pathway_analysis": {
                            "methods": [
                                "GSEA",
                                "ORA",
                                "SPIA",
                                "PathNet"
                            ],
                            "databases": [
                                "KEGG",
                                "Reactome",
                                "GO",
                                "MSigDB"
                            ]
                        },
                        "single_cell": {
                            "clustering": [
                                "Leiden",
                                "Louvain",
                                "Spectral",
                                "DBSCAN"
                            ],
                            "trajectory": [
                                "PAGA",
                                "Monocle3",
                                "Slingshot",
                                "SCORPIUS"
                            ]
                        }
                    },
                    "data_integration": {
                        "multi_omics": {
                            "methods": [
                                "MOFA",
                                "Seurat",
                                "SCENIC+",
                                "SCALEX"
                            ],
                            "applications": [
                                "RNA-ATAC",
                                "RNA-Protein",
                                "RNA-Metabolite"
                            ]
                        },
                        "batch_correction": {
                            "methods": [
                                "Harmony",
                                "BBKNN",
                                "Scanorama",
                                "LIGER"
                            ],
                            "metrics": [
                                "kBET",
                                "LISI",
                                "ASW"
                            ]
                        }
                    },
                    "performance_optimization": {
                        "memory": [
                            "Sparse matrices",
                            "Memory mapping",
                            "Chunked processing"
                        ],
                        "computation": [
                            "Parallel processing",
                            "GPU acceleration",
                            "Vectorization"
                        ]
                    }
                },
                tools={
                    "python_packages": [
                        "scanpy",
                        "networkx",
                        "scipy",
                        "sklearn",
                        "torch",
                        "pandas",
                        "numpy",
                        "anndata"
                    ],
                    "analysis_tools": {
                        "network_analysis": [
                            "networkx.algorithms",
                            "networkx.algorithms.centrality",
                            "networkx.algorithms.community"
                        ],
                        "statistical_analysis": [
                            "scipy.stats",
                            "statsmodels",
                            "sklearn.metrics"
                        ],
                        "machine_learning": [
                            "sklearn.ensemble",
                            "sklearn.model_selection",
                            "torch.nn"
                        ]
                    },
                    "visualization": {
                        "networks": [
                            "networkx.draw",
                            "networkx.draw_spring",
                            "networkx.draw_circular"
                        ],
                        "statistics": [
                            "seaborn",
                            "matplotlib",
                            "plotly"
                        ],
                        "single_cell": [
                            "scanpy.pl",
                            "scanpy.pl.umap",
                            "scanpy.pl.tsne"
                        ]
                    },
                    "performance": {
                        "profiling": [
                            "cProfile",
                            "memory_profiler",
                            "line_profiler"
                        ],
                        "optimization": [
                            "numba",
                            "Cython",
                            "torch.jit"
                        ]
                    }
                }
            ),
            Expert(
                name="Statistician",
                domain=ExpertDomain.STATISTICS,
                expertise_level=0.95,
                background="PhD in Statistics with extensive experience in biological data analysis and high-dimensional statistics",
                skills=[
                    "Statistical testing",
                    "Multiple testing correction",
                    "Effect size estimation",
                    "Power analysis",
                    "Experimental design",
                    "Longitudinal analysis",
                    "Bayesian statistics"
                ],
                specializations=[
                    "High-dimensional data",
                    "Longitudinal analysis",
                    "Survival analysis",
                    "Mixed effects models",
                    "Bayesian inference"
                ],
                constraints=[
                    "Requires sufficient sample size",
                    "Needs proper controls",
                    "Requires clear hypotheses",
                    "Needs effect size estimates",
                    "Requires power analysis"
                ],
                knowledge_base={
                    "statistical_tests": {
                        "parametric": {
                            "t_test": {
                                "types": ["One-sample", "Two-sample", "Paired"],
                                "assumptions": ["Normality", "Equal variance"],
                                "corrections": ["Bonferroni", "FDR", "Holm"]
                            },
                            "anova": {
                                "types": ["One-way", "Two-way", "Repeated measures"],
                                "post_hoc": ["Tukey", "Scheffe", "Dunnett"],
                                "assumptions": ["Normality", "Homoscedasticity", "Independence"]
                            }
                        },
                        "non_parametric": {
                            "mann_whitney": {
                                "assumptions": ["Ordinal data", "Independent samples"],
                                "corrections": ["Bonferroni", "FDR"]
                            },
                            "wilcoxon": {
                                "types": ["Signed-rank", "Rank-sum"],
                                "assumptions": ["Ordinal data", "Symmetric distribution"]
                            },
                            "kruskal_wallis": {
                                "assumptions": ["Ordinal data", "Independent samples"],
                                "post_hoc": ["Dunn", "Nemenyi"]
                            }
                        }
                    },
                    "multiple_testing": {
                        "corrections": {
                            "family_wise": [
                                "Bonferroni",
                                "Holm",
                                "Sidak",
                                "Hochberg"
                            ],
                            "false_discovery": [
                                "Benjamini-Hochberg",
                                "Storey",
                                "Adaptive"
                            ]
                        },
                        "power_analysis": {
                            "sample_size": [
                                "Effect size based",
                                "Power based",
                                "Precision based"
                            ],
                            "methods": [
                                "Simulation",
                                "Analytical",
                                "Bootstrap"
                            ]
                        }
                    },
                    "longitudinal_analysis": {
                        "models": {
                            "mixed_effects": {
                                "types": [
                                    "Linear",
                                    "Nonlinear",
                                    "Generalized"
                                ],
                                "components": [
                                    "Fixed effects",
                                    "Random effects",
                                    "Correlation structure"
                                ]
                            },
                            "survival": {
                                "methods": [
                                    "Cox proportional hazards",
                                    "Kaplan-Meier",
                                    "Accelerated failure time"
                                ],
                                "assumptions": [
                                    "Proportional hazards",
                                    "Independent censoring"
                                ]
                            }
                        }
                    }
                },
                tools={
                    "python_packages": [
                        "scipy",
                        "statsmodels",
                        "sklearn",
                        "pandas",
                        "numpy",
                        "pymc3",
                        "arviz"
                    ],
                    "analysis_tools": {
                        "statistical_tests": [
                            "scipy.stats.ttest_ind",
                            "scipy.stats.f_oneway",
                            "scipy.stats.mannwhitneyu",
                            "scipy.stats.wilcoxon"
                        ],
                        "multiple_testing": [
                            "statsmodels.stats.multitest.fdrcorrection",
                            "statsmodels.stats.multitest.multipletests"
                        ],
                        "power_analysis": [
                            "statsmodels.stats.power.TTestPower",
                            "statsmodels.stats.power.FTestPower"
                        ]
                    },
                    "modeling": {
                        "mixed_effects": [
                            "statsmodels.regression.mixed_linear_model.MixedLM",
                            "statsmodels.regression.linear_model.OLS"
                        ],
                        "survival": [
                            "lifelines.CoxPHFitter",
                            "lifelines.KaplanMeierFitter"
                        ],
                        "bayesian": [
                            "pymc3.Model",
                            "pymc3.sample",
                            "arviz.plot_trace"
                        ]
                    },
                    "visualization": {
                        "statistical": [
                            "seaborn.boxplot",
                            "seaborn.violinplot",
                            "seaborn.regplot"
                        ],
                        "longitudinal": [
                            "seaborn.lineplot",
                            "matplotlib.pyplot.errorbar",
                            "lifelines.plotting.plot_lifetimes"
                        ],
                        "diagnostic": [
                            "statsmodels.graphics.gofplots.qqplot",
                            "statsmodels.graphics.regressionplots.plot_regress_exog",
                            "seaborn.residplot"
                        ]
                    }
                }
            ),
            Expert(
                name="Drug Discovery Expert",
                domain=ExpertDomain.DRUG_DISCOVERY,
                expertise_level=0.95,
                background="PhD in Medicinal Chemistry with extensive experience in drug-target interactions and computational drug discovery",
                skills=[
                    "Drug-target prediction",
                    "ADMET properties",
                    "Structure-activity relationships",
                    "Molecular docking",
                    "Virtual screening",
                    "Lead optimization",
                    "Drug repurposing"
                ],
                specializations=[
                    "Small molecules",
                    "Protein-ligand interactions",
                    "Drug metabolism",
                    "Toxicity prediction",
                    "Drug combination"
                ],
                constraints=[
                    "Requires drug structures",
                    "Needs target information",
                    "Requires activity data",
                    "Needs ADMET properties",
                    "Requires validation data"
                ],
                knowledge_base={
                    "drug_properties": {
                        "physicochemical": {
                            "lipinski": {
                                "rules": [
                                    "MW ≤ 500",
                                    "LogP ≤ 5",
                                    "HBD ≤ 5",
                                    "HBA ≤ 10"
                                ],
                                "exceptions": [
                                    "Natural products",
                                    "Antibiotics",
                                    "Vitamins"
                                ]
                            },
                            "veber": {
                                "rules": [
                                    "Rotatable bonds ≤ 10",
                                    "TPSA ≤ 140"
                                ],
                                "applications": [
                                    "Oral bioavailability",
                                    "Membrane permeability"
                                ]
                            }
                        },
                        "admet": {
                            "absorption": {
                                "properties": [
                                    "Solubility",
                                    "Permeability",
                                    "Bioavailability"
                                ],
                                "predictors": [
                                    "LogS",
                                    "Caco-2",
                                    "PAMPA"
                                ]
                            },
                            "distribution": {
                                "properties": [
                                    "Plasma protein binding",
                                    "Volume of distribution",
                                    "Blood-brain barrier"
                                ],
                                "predictors": [
                                    "PPB",
                                    "VD",
                                    "BBB"
                                ]
                            },
                            "metabolism": {
                                "properties": [
                                    "CYP450 inhibition",
                                    "CYP450 induction",
                                    "Metabolic stability"
                                ],
                                "predictors": [
                                    "CYP1A2",
                                    "CYP2D6",
                                    "CYP3A4"
                                ]
                            },
                            "excretion": {
                                "properties": [
                                    "Renal clearance",
                                    "Hepatic clearance",
                                    "Half-life"
                                ],
                                "predictors": [
                                    "CL",
                                    "t1/2",
                                    "MRT"
                                ]
                            },
                            "toxicity": {
                                "properties": [
                                    "hERG inhibition",
                                    "Ames mutagenicity",
                                    "Carcinogenicity"
                                ],
                                "predictors": [
                                    "hERG",
                                    "AMES",
                                    "CARC"
                                ]
                            }
                        }
                    },
                    "drug_target_interactions": {
                        "binding_modes": {
                            "types": [
                                "Competitive",
                                "Allosteric",
                                "Covalent"
                            ],
                            "forces": [
                                "Hydrogen bonds",
                                "Hydrophobic",
                                "Electrostatic"
                            ]
                        },
                        "prediction_methods": {
                            "structure_based": [
                                "Molecular docking",
                                "MD simulation",
                                "MM-PBSA"
                            ],
                            "ligand_based": [
                                "QSAR",
                                "Pharmacophore",
                                "Similarity search"
                            ]
                        }
                    },
                    "drug_combination": {
                        "synergy_analysis": {
                            "methods": [
                                "Bliss independence",
                                "Loewe additivity",
                                "Chou-Talalay"
                            ],
                            "metrics": [
                                "CI",
                                "DRI",
                                "Synergy score"
                            ]
                        },
                        "mechanisms": {
                            "types": [
                                "Additive",
                                "Synergistic",
                                "Antagonistic"
                            ],
                            "targets": [
                                "Same pathway",
                                "Different pathways",
                                "Compensatory"
                            ]
                        }
                    }
                },
                tools={
                    "python_packages": [
                        "rdkit",
                        "pandas",
                        "numpy",
                        "scipy",
                        "sklearn",
                        "deepchem",
                        "mordred"
                    ],
                    "analysis_tools": {
                        "molecular_descriptors": [
                            "rdkit.Chem.Descriptors",
                            "rdkit.Chem.Lipinski",
                            "mordred.Calculator"
                        ],
                        "fingerprints": [
                            "rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect",
                            "rdkit.Chem.AllChem.GetMACCSKeysFingerprint",
                            "rdkit.Chem.AllChem.GetAtomPairFingerprint"
                        ],
                        "docking": [
                            "deepchem.dock",
                            "deepchem.pose_generation",
                            "deepchem.scoring"
                        ]
                    },
                    "prediction": {
                        "admet": [
                            "deepchem.models.GraphConvModel",
                            "deepchem.models.MultitaskClassifier",
                            "deepchem.models.MultitaskRegressor"
                        ],
                        "activity": [
                            "deepchem.models.GraphConvModel",
                            "deepchem.models.WeaveModel",
                            "deepchem.models.MPNNModel"
                        ],
                        "toxicity": [
                            "deepchem.models.GraphConvModel",
                            "deepchem.models.MultitaskClassifier",
                            "deepchem.models.MultitaskRegressor"
                        ]
                    },
                    "visualization": {
                        "molecules": [
                            "rdkit.Chem.Draw.MolToImage",
                            "rdkit.Chem.Draw.MolToFile",
                            "rdkit.Chem.Draw.MolsToGridImage"
                        ],
                        "properties": [
                            "seaborn.scatterplot",
                            "seaborn.boxplot",
                            "seaborn.heatmap"
                        ],
                        "interactions": [
                            "matplotlib.pyplot",
                            "seaborn.jointplot",
                            "plotly.graph_objects"
                        ]
                    }
                }
            ),
            Expert(
                name="Gene Regulation Expert",
                domain=ExpertDomain.GENE_REGULATION,
                expertise_level=0.91,
                background="PhD in Gene Regulation with focus on transcriptional control",
                skills=["TF binding analysis", "Enhancer prediction", "Regulatory network inference"],
                specializations=["Transcription factors", "Epigenetic regulation"],
                constraints=["Requires TF binding data", "Needs regulatory sequences"],
                knowledge_base={
                    "regulators": ["TFs", "Co-factors", "Chromatin modifiers"],
                    "mechanisms": ["Transcription", "Splicing", "Translation"],
                    "networks": ["Regulatory", "Co-expression"]
                },
                tools={
                    "python_packages": ["networkx", "gseapy", "mygene"],
                    "analysis_tools": ["networkx.Graph", "gseapy.enrichr"],
                    "visualization": ["networkx.draw", "matplotlib"]
                }
            ),
            Expert(
                name="Experimental Designer",
                domain=ExpertDomain.EXPERIMENTAL_DESIGN,
                expertise_level=0.86,
                background="PhD in Experimental Biology with focus on perturbation studies",
                skills=["Experimental design", "Control selection", "Replicate planning"],
                specializations=["Perturbation experiments", "Time series"],
                constraints=["Requires experimental constraints", "Needs biological context"],
                knowledge_base={
                    "designs": ["Factorial", "Time series", "Dose response"],
                    "controls": ["Positive", "Negative", "Vehicle"],
                    "replicates": ["Technical", "Biological"]
                },
                tools={
                    "python_packages": ["pandas", "numpy", "scipy"],
                    "analysis_tools": ["scipy.stats", "pandas.DataFrame"],
                    "visualization": ["seaborn", "matplotlib"]
                }
            ),
            Expert(
                name="Self-critic",
                domain=ExpertDomain.VALIDATION,
                expertise_level=0.95,
                background="Distinguished Professor of Systems Biology with 45+ years of experience in model validation and critical analysis, pioneer in computational method evaluation",
                skills=[
                    "Model evaluation",
                    "Critical analysis",
                    "Performance assessment",
                    "Quality metrics",
                    "Statistical validation",
                    "Method comparison",
                    "Bias detection"
                ],
                specializations=[
                    "Single-cell model validation",
                    "Perturbation model validation",
                    "Deep learning evaluation",
                    "Biological verification",
                    "Technical validation"
                ],
                constraints=[
                    "Requires validation data",
                    "Needs ground truth",
                    "Requires performance metrics",
                    "Needs biological context",
                    "Requires statistical power"
                ],
                knowledge_base={
                    "validation_methods": {
                        "model_evaluation": {
                            "metrics": {
                                "regression": [
                                    "MSE",
                                    "RMSE",
                                    "R2",
                                    "PCC"
                                ],
                                "classification": [
                                    "Accuracy",
                                    "Precision",
                                    "Recall",
                                    "F1-score",
                                    "AUROC"
                                ]
                            },
                            "bias_detection": {
                                "types": [
                                    "Selection bias",
                                    "Measurement bias",
                                    "Confounding bias"
                                ],
                                "mitigation": [
                                    "Data balancing",
                                    "Covariate adjustment",
                                    "Sensitivity analysis"
                                ]
                            }
                        },
                        "critical_analysis": {
                            "model_limitations": {
                                "technical": [
                                    "Computational efficiency",
                                    "Scalability",
                                    "Memory requirements"
                                ],
                                "biological": [
                                    "Cell type specificity",
                                    "Pathway coverage",
                                    "Mechanistic interpretability"
                                ],
                                "practical": [
                                    "Data requirements",
                                    "Implementation complexity",
                                    "Deployment challenges"
                                ]
                            },
                            "improvement_suggestions": {
                                "architecture": [
                                    "Model complexity",
                                    "Layer design",
                                    "Attention mechanisms"
                                ],
                                "training": [
                                    "Loss functions",
                                    "Optimization",
                                    "Regularization"
                                ],
                                "data": [
                                    "Quality control",
                                    "Feature selection",
                                    "Data augmentation"
                                ]
                            }
                        }
                    }
                },
                tools={
                    "python_packages": [
                        "sklearn",
                        "scipy",
                        "pandas",
                        "numpy",
                        "statsmodels",
                        "matplotlib",
                        "seaborn",
                        "torchmetrics"
                    ],
                    "analysis_tools": {
                        "model_evaluation": [
                            "sklearn.metrics.mean_squared_error",
                            "sklearn.metrics.roc_auc_score",
                            "torchmetrics.functional"
                        ],
                        "bias_detection": [
                            "sklearn.metrics.confusion_matrix",
                            "statsmodels.stats.multitest",
                            "scipy.stats.chi2_contingency"
                        ],
                        "critical_analysis": [
                            "sklearn.model_selection.cross_val_score",
                            "sklearn.inspection.permutation_importance",
                            "shap.Explainer"
                        ]
                    },
                    "visualization": {
                        "performance": [
                            "seaborn.boxplot",
                            "seaborn.violinplot",
                            "matplotlib.pyplot.scatter"
                        ],
                        "bias": [
                            "seaborn.heatmap",
                            "seaborn.regplot",
                            "matplotlib.pyplot.plot"
                        ],
                        "limitations": [
                            "seaborn.distplot",
                            "seaborn.jointplot",
                            "matplotlib.pyplot.hist"
                        ]
                    }
                }
            ),
            Expert(
                name="Baseline Assessment Expert",
                domain=ExpertDomain.BASELINE_ASSESSMENT,
                expertise_level=0.93,
                background="PhD in Computational Biology with extensive experience in method benchmarking and evaluation",
                skills=[
                    "Method benchmarking",
                    "Performance analysis",
                    "Limitation identification",
                    "Comparative evaluation",
                    "Statistical analysis"
                ],
                specializations=[
                    "Single-cell method evaluation",
                    "Deep learning model assessment",
                    "Biological validation"
                ],
                constraints=[
                    "Focuses on method evaluation",
                    "Does not design new methods",
                    "Requires comprehensive baseline data"
                ],
                knowledge_base={
                    "evaluation_metrics": [
                        "MSE",
                        "PCC",
                        "R2",
                        "AUROC",
                        "Biological relevance"
                    ],
                    "baseline_methods": {
                        "gene_knockout": [
                            "scGPT",
                            "Random Forest",
                            "Linear Regression"
                        ],
                        "drug_perturbation": [
                            "ChemCPA",
                            "Random Forest",
                            "Linear Regression"
                        ],
                        "cytokine_stimulation": [
                            "Random Forest",
                            "Linear Regression"
                        ]
                    },
                    "limitations": {
                        "technical": [
                            "Computational efficiency",
                            "Scalability",
                            "Memory requirements"
                        ],
                        "biological": [
                            "Cell type specificity",
                            "Pathway coverage",
                            "Mechanistic interpretability"
                        ],
                        "practical": [
                            "Data requirements",
                            "Implementation complexity",
                            "Deployment challenges"
                        ]
                    }
                },
                tools={
                    "python_packages": [
                        "scipy",
                        "statsmodels",
                        "sklearn",
                        "torchmetrics",
                        "pandas",
                        "numpy"
                    ],
                    "analysis_tools": [
                        "scipy.stats",
                        "sklearn.metrics",
                        "torchmetrics.functional",
                        "pandas.DataFrame.compare"
                    ],
                    "visualization": [
                        "seaborn",
                        "matplotlib",
                        "plotly"
                    ],
                    "evaluation_frameworks": [
                        "Cross-validation",
                        "Leave-one-out",
                        "Bootstrap",
                        "Permutation tests"
                    ]
                }
            ),
            Expert(
                name="Training Expert",
                domain=ExpertDomain.TRAINING,
                expertise_level=0.95,
                background="Senior Research Scientist with 25+ years of experience in deep learning training and optimization, specializing in biological sequence modeling and perturbation prediction",
                skills=[
                    "Training strategy design",
                    "Hyperparameter optimization",
                    "Model evaluation",
                    "Performance analysis",
                    "Training monitoring",
                    "Resource optimization",
                    "Distributed training",
                    "Model debugging"
                ],
                specializations=[
                    "Biological sequence modeling",
                    "Perturbation prediction",
                    "Multi-modal training",
                    "Transfer learning",
                    "Curriculum learning"
                ],
                constraints=[
                    "Requires clear metrics",
                    "Needs validation data",
                    "Requires monitoring tools",
                    "Needs computational resources",
                    "Requires reproducibility"
                ],
                knowledge_base={
                    "training_strategies": {
                        "optimization": {
                            "optimizers": [
                                "Adam",
                                "AdamW",
                                "SGD",
                                "RAdam"
                            ],
                            "schedulers": [
                                "CosineAnnealingLR",
                                "ReduceLROnPlateau",
                                "OneCycleLR",
                                "WarmupLR"
                            ],
                            "regularization": [
                                "Dropout",
                                "Weight decay",
                                "Gradient clipping",
                                "Early stopping"
                            ]
                        },
                        "evaluation": {
                            "metrics": {
                                "regression": [
                                    "MSE",
                                    "MAE",
                                    "R2",
                                    "Explained variance"
                                ],
                                "classification": [
                                    "Accuracy",
                                    "Precision",
                                    "Recall",
                                    "F1-score"
                                ],
                                "biological": [
                                    "Pathway enrichment",
                                    "Network consistency",
                                    "Cell type specificity"
                                ]
                            },
                            "validation": {
                                "methods": [
                                    "Cross-validation",
                                    "Hold-out",
                                    "Bootstrap",
                                    "Time-based split"
                                ],
                                "techniques": [
                                    "Grid search",
                                    "Random search",
                                    "Bayesian optimization",
                                    "Hyperband"
                                ]
                            }
                        }
                    }
                },
                tools={
                    "python_packages": [
                        "torch",
                        "pytorch_lightning",
                        "wandb",
                        "optuna",
                        "scikit-learn",
                        "numpy",
                        "pandas"
                    ],
                    "training_components": {
                        "optimizers": [
                            "torch.optim.Adam",
                            "torch.optim.AdamW",
                            "torch.optim.SGD"
                        ],
                        "schedulers": [
                            "torch.optim.lr_scheduler.CosineAnnealingLR",
                            "torch.optim.lr_scheduler.ReduceLROnPlateau",
                            "torch.optim.lr_scheduler.OneCycleLR"
                        ],
                        "regularization": [
                            "torch.nn.Dropout",
                            "torch.nn.BatchNorm1d",
                            "torch.nn.LayerNorm"
                        ]
                    },
                    "evaluation": {
                        "metrics": [
                            "torchmetrics.MeanSquaredError",
                            "torchmetrics.MeanAbsoluteError",
                            "torchmetrics.R2Score",
                            "torchmetrics.Accuracy"
                        ],
                        "visualization": [
                            "wandb",
                            "matplotlib.pyplot",
                            "seaborn"
                        ]
                    },
                    "optimization": {
                        "hyperparameter_tuning": [
                            "optuna",
                            "ray.tune",
                            "hyperopt"
                        ],
                        "monitoring": [
                            "wandb",
                            "tensorboard",
                            "mlflow"
                        ]
                    }
                }
            ),
            Expert(
                name="Drug Response Expert",
                domain=ExpertDomain.DRUG_RESPONSE,
                expertise_level=0.95,
                background="Senior Research Scientist with 20+ years of experience in drug response prediction and perturbation analysis, specializing in single-cell transcriptomics and drug mechanism of action",
                skills=[
                    "Drug response prediction",
                    "Perturbation analysis",
                    "Mechanism of action",
                    "Dose-response modeling",
                    "Drug combination analysis",
                    "Toxicity prediction",
                    "Drug target identification",
                    "Pathway analysis"
                ],
                specializations=[
                    "Single-cell drug response",
                    "Perturbation prediction",
                    "Drug mechanism of action",
                    "Drug combination effects",
                    "Toxicity assessment"
                ],
                constraints=[
                    "Requires drug information",
                    "Needs cell type data",
                    "Requires dose information",
                    "Needs time course data",
                    "Requires biological validation"
                ],
                knowledge_base={
                    "drug_response": {
                        "prediction": {
                            "models": [
                                "Dose-response curves",
                                "IC50 prediction",
                                "EC50 prediction",
                                "Hill equation"
                            ],
                            "features": [
                                "Drug properties",
                                "Cell type",
                                "Gene expression",
                                "Pathway activity"
                            ],
                            "evaluation": [
                                "RMSE",
                                "R2",
                                "Pearson correlation",
                                "Spearman correlation"
                            ]
                        },
                        "mechanism": {
                            "analysis": [
                                "Target identification",
                                "Pathway enrichment",
                                "Network analysis",
                                "Gene set analysis"
                            ],
                            "validation": [
                                "Knockdown experiments",
                                "Overexpression",
                                "Drug combination",
                                "Time course"
                            ]
                        }
                    }
                },
                tools={
                    "python_packages": [
                        "pandas",
                        "numpy",
                        "scipy",
                        "scikit-learn",
                        "scanpy",
                        "anndata",
                        "drugcomb"
                    ],
                    "analysis_components": {
                        "dose_response": [
                            "scipy.optimize.curve_fit",
                            "drugcomb.DrugComb",
                            "drugcomb.DrugCombMatrix"
                        ],
                        "pathway_analysis": [
                            "scanpy.tl.rank_genes_groups",
                            "scanpy.tl.score_genes",
                            "scanpy.tl.score_genes_cell_cycle"
                        ],
                        "visualization": [
                            "matplotlib.pyplot",
                            "seaborn",
                            "plotly.express"
                        ]
                    },
                    "databases": {
                        "drug_info": [
                            "ChEMBL",
                            "DrugBank",
                            "PubChem"
                        ],
                        "pathways": [
                            "KEGG",
                            "Reactome",
                            "GO"
                        ],
                        "interactions": [
                            "STRING",
                            "BioGRID",
                            "STITCH"
                        ]
                    }
                }
            ),
            Expert(
                name="Pathway Analyst",
                domain=ExpertDomain.PATHWAY,
                expertise_level=0.95,
                background="Senior Research Scientist with 20+ years of experience in pathway analysis and biological network modeling, specializing in single-cell transcriptomics and perturbation analysis",
                skills=[
                    "Pathway analysis",
                    "Network modeling",
                    "Gene set enrichment",
                    "Biological network inference",
                    "Perturbation analysis",
                    "Cell type analysis",
                    "Multi-omics integration",
                    "Biological validation"
                ],
                specializations=[
                    "Single-cell pathway analysis",
                    "Perturbation effects",
                    "Network inference",
                    "Cell type-specific pathways",
                    "Multi-omics integration"
                ],
                constraints=[
                    "Requires pathway databases",
                    "Needs cell type information",
                    "Requires gene expression data",
                    "Needs biological context",
                    "Requires validation data"
                ],
                knowledge_base={
                    "pathway_analysis": {
                        "methods": {
                            "enrichment": [
                                "GSEA",
                                "ORA",
                                "PAGE",
                                "CAMERA"
                            ],
                            "network": [
                                "PPI networks",
                                "Gene regulatory networks",
                                "Metabolic networks",
                                "Signaling networks"
                            ],
                            "inference": [
                                "WGCNA",
                                "SCENIC",
                                "GENIE3",
                                "GRNBoost2"
                            ]
                        },
                        "databases": {
                            "pathways": [
                                "KEGG",
                                "Reactome",
                                "GO",
                                "WikiPathways"
                            ],
                            "interactions": [
                                "STRING",
                                "BioGRID",
                                "STITCH",
                                "TRRUST"
                            ],
                            "cell_type": [
                                "CellMarker",
                                "PanglaoDB",
                                "CellPhoneDB",
                                "CellTypist"
                            ]
                        }
                    }
                },
                tools={
                    "python_packages": [
                        "scanpy",
                        "anndata",
                        "pandas",
                        "numpy",
                        "scipy",
                        "networkx",
                        "pyensembl"
                    ],
                    "analysis_components": {
                        "pathway_analysis": [
                            "scanpy.tl.rank_genes_groups",
                            "scanpy.tl.score_genes",
                            "scanpy.tl.score_genes_cell_cycle"
                        ],
                        "network_analysis": [
                            "networkx.Graph",
                            "networkx.DiGraph",
                            "networkx.algorithms.centrality"
                        ],
                        "visualization": [
                            "matplotlib.pyplot",
                            "seaborn",
                            "plotly.express"
                        ]
                    },
                    "databases": {
                        "pathways": [
                            "KEGG",
                            "Reactome",
                            "GO",
                            "WikiPathways"
                        ],
                        "interactions": [
                            "STRING",
                            "BioGRID",
                            "STITCH",
                            "TRRUST"
                        ],
                        "cell_type": [
                            "CellMarker",
                            "PanglaoDB",
                            "CellPhoneDB",
                            "CellTypist"
                        ]
                    }
                }
            ),
            Expert(
                name="Cell Communication Expert",
                domain=ExpertDomain.CELL_COMMUNICATION,
                expertise_level=0.95,
                background="Senior Research Scientist with 20+ years of experience in cell-cell communication analysis and signaling network modeling, specializing in single-cell transcriptomics and perturbation analysis",
                skills=[
                    "Cell-cell communication analysis",
                    "Signaling network modeling",
                    "Ligand-receptor interaction",
                    "Cell type analysis",
                    "Perturbation effects",
                    "Multi-omics integration",
                    "Network inference",
                    "Biological validation"
                ],
                specializations=[
                    "Single-cell communication",
                    "Perturbation effects",
                    "Signaling networks",
                    "Cell type interactions",
                    "Multi-omics integration"
                ],
                constraints=[
                    "Requires cell type data",
                    "Needs gene expression data",
                    "Requires interaction databases",
                    "Needs biological context",
                    "Requires validation data"
                ],
                knowledge_base={
                    "cell_communication": {
                        "analysis": {
                            "methods": [
                                "CellPhoneDB",
                                "NicheNet",
                                "CellChat",
                                "iTALK"
                            ],
                            "networks": [
                                "Ligand-receptor networks",
                                "Signaling networks",
                                "Cell type networks",
                                "Perturbation networks"
                            ],
                            "validation": [
                                "Co-expression",
                                "Spatial proximity",
                                "Functional assays",
                                "Perturbation experiments"
                            ]
                        },
                        "databases": {
                            "interactions": [
                                "CellPhoneDB",
                                "NicheNet",
                                "CellChat",
                                "iTALK"
                            ],
                            "pathways": [
                                "KEGG",
                                "Reactome",
                                "GO",
                                "WikiPathways"
                            ],
                            "cell_type": [
                                "CellMarker",
                                "PanglaoDB",
                                "CellPhoneDB",
                                "CellTypist"
                            ]
                        }
                    }
                },
                tools={
                    "python_packages": [
                        "scanpy",
                        "anndata",
                        "pandas",
                        "numpy",
                        "scipy",
                        "networkx",
                        "cellphonedb"
                    ],
                    "analysis_components": {
                        "communication_analysis": [
                            "cellphonedb.CellPhoneDB",
                            "cellphonedb.methods",
                            "cellphonedb.visualization"
                        ],
                        "network_analysis": [
                            "networkx.Graph",
                            "networkx.DiGraph",
                            "networkx.algorithms.centrality"
                        ],
                        "visualization": [
                            "matplotlib.pyplot",
                            "seaborn",
                            "plotly.express"
                        ]
                    },
                    "databases": {
                        "interactions": [
                            "CellPhoneDB",
                            "NicheNet",
                            "CellChat",
                            "iTALK"
                        ],
                        "pathways": [
                            "KEGG",
                            "Reactome",
                            "GO",
                            "WikiPathways"
                        ],
                        "cell_type": [
                            "CellMarker",
                            "PanglaoDB",
                            "CellPhoneDB",
                            "CellTypist"
                        ]
                    }
                }
            ),
            Expert(
                name="Omics Modality Expert",
                domain=ExpertDomain.OMICS,
                expertise_level=0.95,
                background="Senior Research Scientist with 20+ years of experience in multi-omics data analysis and integration, specializing in single-cell transcriptomics, proteomics, and epigenomics",
                skills=[
                    "Multi-omics integration",
                    "Single-cell analysis",
                    "Data preprocessing",
                    "Feature selection",
                    "Dimensionality reduction",
                    "Batch effect correction",
                    "Data quality control",
                    "Biological validation"
                ],
                specializations=[
                    "Single-cell transcriptomics",
                    "Proteomics",
                    "Epigenomics",
                    "Multi-omics integration",
                    "Perturbation analysis"
                ],
                constraints=[
                    "Requires quality data",
                    "Needs metadata",
                    "Requires batch information",
                    "Needs biological context",
                    "Requires validation data"
                ],
                knowledge_base={
                    "omics_analysis": {
                        "modalities": {
                            "transcriptomics": [
                                "RNA-seq",
                                "scRNA-seq",
                                "spatial transcriptomics",
                                "CITE-seq"
                            ],
                            "proteomics": [
                                "Mass spectrometry",
                                "CyTOF",
                                "CITE-seq",
                                "REAP-seq"
                            ],
                            "epigenomics": [
                                "ATAC-seq",
                                "ChIP-seq",
                                "scATAC-seq",
                                "CUT&RUN"
                            ]
                        },
                        "integration": {
                            "methods": [
                                "MOFA+",
                                "Seurat",
                                "Harmony",
                                "LIGER"
                            ],
                            "analysis": [
                                "Feature selection",
                                "Dimensionality reduction",
                                "Batch correction",
                                "Clustering"
                            ],
                            "validation": [
                                "Cross-validation",
                                "Biological validation",
                                "Technical validation",
                                "Functional assays"
                            ]
                        }
                    }
                },
                tools={
                    "python_packages": [
                        "scanpy",
                        "anndata",
                        "pandas",
                        "numpy",
                        "scipy",
                        "scikit-learn",
                        "harmony"
                    ],
                    "analysis_components": {
                        "preprocessing": [
                            "scanpy.pp.filter_cells",
                            "scanpy.pp.filter_genes",
                            "scanpy.pp.normalize_total",
                            "scanpy.pp.log1p"
                        ],
                        "integration": [
                            "harmony.harmony_timer",
                            "scanpy.pp.regress_out",
                            "scanpy.pp.scale",
                            "scanpy.pp.pca"
                        ],
                        "visualization": [
                            "matplotlib.pyplot",
                            "seaborn",
                            "plotly.express"
                        ]
                    },
                    "databases": {
                        "reference": [
                            "Ensembl",
                            "GENCODE",
                            "RefSeq",
                            "UniProt"
                        ],
                        "annotations": [
                            "GO",
                            "KEGG",
                            "Reactome",
                            "WikiPathways"
                        ],
                        "cell_type": [
                            "CellMarker",
                            "PanglaoDB",
                            "CellPhoneDB",
                            "CellTypist"
                        ]
                    }
                }
            )
        ]

    def select_experts_for_task(self, task_type: str, task_analysis: Dict[str, Any]) -> List[Expert]:
        """Select experts using LLM reasoning based on task requirements"""

        # Use LLM to select appropriate experts
        selected_experts = self._select_experts_with_llm(task_type, task_analysis)

        if not selected_experts:
            # Fallback to core experts if LLM selection fails
            selected_experts = self._get_core_experts(task_type)

        return selected_experts

    def _select_experts_with_llm(self, task_type: str, task_analysis: Dict[str, Any]) -> List[Expert]:
        """Use LLM to intelligently select experts for the task"""
        try:
            # Create expert profiles for LLM
            expert_profiles = self._create_expert_profiles()

            # Create LLM prompt for expert selection
            prompt = self._create_expert_selection_prompt(task_type, task_analysis, expert_profiles)

            # Get LLM response (simplified for now)
            selected_expert_names = self._get_llm_expert_selection(prompt)

            # Convert names to expert objects
            selected_experts = []
            for name in selected_expert_names:
                expert = self._get_expert_by_name(name)
                if expert:
                    selected_experts.append(expert)

            return selected_experts

        except Exception as e:
            print(f"LLM expert selection failed: {e}")
            return []

    def _create_expert_profiles(self) -> str:
        """Create expert profiles for LLM consideration"""
        profiles = []
        for expert in self.experts:
            profile = f"""
Expert: {expert.name}
Domain: {expert.domain.value}
Expertise Level: {expert.expertise_level}
Background: {expert.background}
Skills: {', '.join(expert.skills)}
Specializations: {', '.join(expert.specializations)}
"""
            profiles.append(profile)

        return "\n".join(profiles)

    def _create_expert_selection_prompt(self, task_type: str, task_analysis: Dict[str, Any], expert_profiles: str) -> str:
        """Create prompt for LLM expert selection"""
        task_summary = str(task_analysis)[:1000]  # Limit length

        prompt = f"""
You are an expert coordinator for a scientific research team. Based on the task requirements, select the most appropriate experts to collaborate on this project.

Task Type: {task_type}
Task Analysis: {task_summary}

Available Experts:
{expert_profiles}

Requirements:
1. Select experts based on task requirements and complexity (typically 5-7 experts)
2. MUST include these core experts:
   - Data Engineering (for data processing and pipeline design)
   - Single Cell Biology (for biological context and interpretation)
   - Deep Learning (for model architecture and training)
   - Training (for model optimization and validation)
   - Critic (for proposal evaluation and quality control)
3. Consider the task type and select additional domain-specific experts if needed
4. Ensure the team has complementary skills and can cover all aspects of the project

Please respond with only the names of the selected experts, one per line:
"""
        return prompt

    def _get_llm_expert_selection(self, prompt: str) -> List[str]:
        """Get expert selection from LLM"""
        try:
            llm_client = LLMInterface()

            # Call LLM for expert selection
            response = llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )

            # Parse response to extract expert names
            expert_names = []
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    expert_names.append(line)

            # Validate expert names exist
            valid_experts = []
            for name in expert_names:
                try:
                    self._get_expert_by_name(name)
                    valid_experts.append(name)
                except ValueError:
                    print(f"Warning: Expert '{name}' not found, skipping")

            return valid_experts if valid_experts else self._get_fallback_experts()

        except Exception as e:
            print(f"LLM expert selection failed: {e}")
            return self._get_fallback_experts()

    def _get_fallback_experts(self) -> List[str]:
        """Fallback expert selection - includes core required experts"""
        return [
            "Data Engineer",
            "Single Cell Biology Expert",
            "Deep Learning Expert",
            "Training Expert",
            "Critic Agent"
        ]

    def _get_core_experts(self, task_type: str) -> List[Expert]:
        """Fallback method to get core experts"""
        core_experts = []

        # Always include these core experts
        core_domains = [
            ExpertDomain.DATA_ENGINEERING,
            ExpertDomain.SINGLE_CELL_BIOLOGY,
            ExpertDomain.DEEP_LEARNING,
            ExpertDomain.TRAINING
        ]

        for domain in core_domains:
            domain_experts = [e for e in self.experts if e.domain == domain]
            if domain_experts:
                # Select the expert with highest expertise level
                best_expert = max(domain_experts, key=lambda e: e.expertise_level)
                core_experts.append(best_expert)

        return core_experts

    def _get_expert_by_name(self, name: str) -> Expert:
        for expert in self.experts:
            if expert.name == name:
                return expert
        raise ValueError(f"Expert not found: {name}")

    def update_expert_confidence(self, expert: Expert, new_confidence: float):
        expert.confidence = np.clip(new_confidence, 0.0, 1.0)
