from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
import torch
import random
import os
import json
from datetime import datetime

@dataclass
class RefinementConfig:
    seed: int = 42
    train_test_split: float = 0.8
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_epochs: int = 100
    early_stopping_patience: int = 15
    validation_frequency: int = 5

class RefinementFramework:
    def __init__(self, config: RefinementConfig):
        self.config = config
        self._set_seed()
        self.discussion_results = {}
        self.final_plan = {}

    def _set_seed(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)

    def integrate_discussion_results(self, discussion_results: Dict[str, Any]):
        self.discussion_results = discussion_results
        self._generate_final_plan()

    def _generate_final_plan(self):
        self.final_plan = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_architecture": self._refine_architecture(),
            "data_processing": self._refine_data_processing(),
            "training_strategy": self._refine_training_strategy(),
            "evaluation_metrics": self._refine_evaluation_metrics(),
            "implementation_details": self._refine_implementation_details(),
            "biological_validation": self._refine_biological_validation(),
            "computational_optimization": self._refine_computational_optimization(),
            "model_assumptions": self._refine_model_assumptions(),
            "extensions": self._refine_extensions(),
            "mermaid_diagrams": self._generate_mermaid_json(),
            "framework_overview": self._generate_framework_overview()
        }

    def _refine_architecture(self) -> Dict[str, Any]:
        # Handle different possible structures of discussion results
        if "model_architecture" in self.discussion_results:
            arch = self.discussion_results["model_architecture"]
        elif "architecture" in self.discussion_results:
            arch = self.discussion_results["architecture"]
        else:
            # Generate default architecture if no discussion results
            arch = self._generate_default_architecture()

        refined_arch = {}

        for component, details in arch.items():
            if isinstance(details, dict):
                refined_arch[component] = self._refine_component(details)
            else:
                refined_arch[component] = details

        return refined_arch

    def _generate_default_architecture(self) -> Dict[str, Any]:
        """Generate default architecture when no discussion results are available"""
        return {
            "encoder": {
                "type": "transformer",
                "layers": 4,
                "heads": 8,
                "dimension": 512,
                "activation": "gelu"
            },
            "perturbation_encoder": {
                "type": "mlp",
                "layers": 3,
                "dimensions": [256, 128, 64],
                "activation": "swish"
            },
            "cross_attention": {
                "heads": 8,
                "dimension": 512,
                "dropout": 0.1
            },
            "decoder": {
                "type": "mlp",
                "layers": 3,
                "dimensions": [256, 512, 1024],
                "activation": "swish"
            }
        }

    def _refine_component(self, component: Dict[str, Any]) -> Dict[str, Any]:
        refined = {}
        for key, value in component.items():
            if isinstance(value, dict):
                refined[key] = self._refine_component(value)
            elif isinstance(value, list):
                refined[key] = [self._refine_component(item) if isinstance(item, dict) else item for item in value]
            else:
                refined[key] = value
        return refined

    def _refine_data_processing(self) -> Dict[str, Any]:
        # Handle different possible structures
        if "preprocessing" in self.discussion_results:
            dp = self.discussion_results["preprocessing"]
        elif "data_processing" in self.discussion_results:
            dp = self.discussion_results["data_processing"]
        else:
            # Generate default data processing
            dp = self._generate_default_data_processing()

        refined_dp = {}

        for step, details in dp.items():
            if isinstance(details, dict):
                refined_dp[step] = self._refine_component(details)
            else:
                refined_dp[step] = details

        return refined_dp

    def _generate_default_data_processing(self) -> Dict[str, Any]:
        """Generate default data processing pipeline"""
        return {
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
                }
            },
            "normalization": {
                "method": "log1p",
                "target_sum": 10000,
                "regression_vars": ["total_counts", "pct_counts_mt"]
            },
            "batch_correction": {
                "method": "harmony",
                "parameters": {
                    "theta": 2,
                    "max_iterations": 20
                }
            }
        }

    def _refine_training_strategy(self) -> Dict[str, Any]:
        # Handle different possible structures
        if "training_strategy" in self.discussion_results:
            ts = self.discussion_results["training_strategy"]
        elif "training" in self.discussion_results:
            ts = self.discussion_results["training"]
        else:
            # Generate default training strategy
            ts = self._generate_default_training_strategy()

        refined_ts = {}

        for component, details in ts.items():
            if isinstance(details, dict):
                refined_ts[component] = self._refine_component(details)
            else:
                refined_ts[component] = details

        return refined_ts

    def _generate_default_training_strategy(self) -> Dict[str, Any]:
        """Generate default training strategy"""
        return {
            "optimizer": {
                "type": "adamw",
                "learning_rate": 3e-4,
                "weight_decay": 0.01
            },
            "scheduler": {
                "type": "cosine_annealing",
                "T_0": 10,
                "eta_min": 1e-6
            },
            "training": {
                "batch_size": 64,
                "gradient_clip": 1.0,
                "early_stopping": {
                    "patience": 15,
                    "min_delta": 1e-4
                }
            }
        }

    def _refine_evaluation_metrics(self) -> Dict[str, Any]:
        # Handle different possible structures
        if "evaluation_metrics" in self.discussion_results:
            metrics = self.discussion_results["evaluation_metrics"]
        elif "optimization_strategy" in self.discussion_results:
            opt = self.discussion_results["optimization_strategy"]
            if "loss_functions" in opt:
                metrics = {"loss_functions": opt["loss_functions"]}
            else:
                metrics = self._generate_default_evaluation_metrics()
        else:
            # Generate default evaluation metrics
            metrics = self._generate_default_evaluation_metrics()

        refined_metrics = {}

        for level, details in metrics.items():
            if isinstance(details, dict):
                refined_metrics[level] = self._refine_component(details)
            else:
                refined_metrics[level] = details

        return refined_metrics

    def _generate_default_evaluation_metrics(self) -> Dict[str, Any]:
        """Generate default evaluation metrics"""
        return {
            "loss_functions": {
                "reconstruction": "mse",
                "perturbation": "bce",
                "biological": "pathway_consistency"
            },
            "metrics": [
                "MSE",
                "Pearson correlation",
                "R²",
                "Biological pathway enrichment"
            ]
        }

    def _refine_implementation_details(self) -> Dict[str, Any]:
        # Handle different possible structures
        if "implementation_details" in self.discussion_results:
            impl = self.discussion_results["implementation_details"]
        else:
            # Generate default implementation details
            impl = self._generate_default_implementation_details()

        refined_impl = {}

        for aspect, details in impl.items():
            if isinstance(details, dict):
                refined_impl[aspect] = self._refine_component(details)
            else:
                refined_impl[aspect] = details

        return refined_impl

    def _generate_default_implementation_details(self) -> Dict[str, Any]:
        """Generate default implementation details"""
        return {
            "framework": "PyTorch",
            "hardware_requirements": {
                "gpu": "NVIDIA GPU with 8GB+ VRAM",
                "ram": "32GB+ system RAM",
                "storage": "100GB+ SSD"
            },
            "dependencies": [
                "torch>=1.12.0",
                "scanpy>=1.9.0",
                "numpy>=1.21.0",
                "scikit-learn>=1.0.0"
            ],
            "code_structure": {
                "data_loader": "Custom DataLoader for single-cell data",
                "model": "Modular architecture with configurable components",
                "training": "Training loop with early stopping and checkpointing",
                "evaluation": "Comprehensive evaluation pipeline"
            }
        }

    def _refine_biological_validation(self) -> Dict[str, Any]:
        return {
            "key_theoretical_foundations": {
                "baseline_regulatory_info": {
                    "assumption": "Baseline contains sufficient regulatory info",
                    "validation": "Ablation study (masking key TFs)",
                    "biological_basis": "Central dogma of molecular biology"
                },
                "dose_response_continuity": {
                    "assumption": "Dose-response continuity",
                    "validation": "Dose interpolation experiments",
                    "biological_basis": "Ligand-receptor binding kinetics"
                },
                "pathway_modularity": {
                    "assumption": "Pathway modularity",
                    "validation": "Pathway enrichment analysis",
                    "biological_basis": "Known biological pathway organization"
                },
                "latent_space_smoothness": {
                    "assumption": "Latent space smoothness",
                    "validation": "UMAP visualization",
                    "biological_basis": "Waddington epigenetic landscape concept"
                }
            },
            "adaptive_mechanisms": {
                "dose_specific_attention": {
                    "description": "Higher attention weights to stress pathways at toxic doses",
                    "implementation": "Focus on growth signaling at therapeutic doses"
                },
                "cell_type_gating": {
                    "description": "Cell-type specific pathway modulation",
                    "implementation": "Suppress irrelevant pathways based on cell type"
                },
                "chemical_class_priors": {
                    "description": "Initialize embeddings based on chemical class",
                    "implementation": "Set initial weights based on known mechanisms"
                }
            }
        }

    def _refine_computational_optimization(self) -> Dict[str, Any]:
        return {
            "training_techniques": {
                "progressive_dose_sampling": {
                    "description": "Gradually increase dose range during training",
                    "implementation": {
                        "epoch_0_10": "uniform(0, 1μM)",
                        "epoch_10_20": "log_normal(1μM-10μM)",
                        "epoch_20+": "full_range(0-100μM)"
                    }
                }
            },
            "regularization_methods": {
                "pathway_sparsity": {
                    "type": "l1",
                    "lambda": 0.01,
                    "target": "pathway_weights"
                },
                "dose_consistency": {
                    "type": "custom",
                    "formula": "L_consist = E[||f(x,d1) - f(f(x,d1),d2-d1)||^2]"
                }
            },
            "parallelization": {
                "gene_cluster_parallelism": {
                    "chr1": {
                        "gpu": "GPU0",
                        "genes": ["EGFR", "MYC"]
                    },
                    "chr17": {
                        "gpu": "GPU1",
                        "genes": ["TP53", "BRCA1"]
                    },
                    "others": {
                        "gpu": "GPU2",
                        "genes": "Housekeeping"
                    }
                }
            }
        }

    def _refine_model_assumptions(self) -> Dict[str, Any]:
        return {
            "data_assumptions": {
                "sparse_imbalanced": {
                    "challenge": "Sparse and imbalanced datasets",
                    "solution": "Advanced sampling and weighting techniques"
                },
                "batch_effects": {
                    "challenge": "Batch effects between experiments",
                    "solution": "Harmony integration and batch correction"
                },
                "technical_noise": {
                    "challenge": "Missing values and technical noise",
                    "solution": "Robust preprocessing and imputation"
                }
            },
            "computational_assumptions": {
                "dimensionality": {
                    "challenge": "High-dimensional data",
                    "solution": "Efficient dimensionality reduction"
                },
                "transfer_learning": {
                    "challenge": "Limited training data",
                    "solution": "Pre-trained models and transfer learning"
                },
                "efficiency": {
                    "challenge": "Computational efficiency",
                    "solution": "Optimized implementations"
                }
            }
        }

    def _refine_extensions(self) -> Dict[str, Any]:
        return {
            "drug_discovery": {
                "virtual_screening": {
                    "description": "Virtual screening of novel compounds",
                    "implementation": "Predict expression profiles for new compounds"
                },
                "drug_repurposing": {
                    "description": "Repurposing existing drugs",
                    "implementation": "Identify new indications based on expression patterns"
                },
                "side_effects": {
                    "description": "Predicting side effects",
                    "implementation": "Analyze off-target expression changes"
                }
            },
            "precision_medicine": {
                "patient_specific": {
                    "description": "Patient-specific drug response prediction",
                    "implementation": "Use baseline expression for personalized predictions"
                },
                "drug_combinations": {
                    "description": "Optimize drug combinations",
                    "implementation": "Predict synergistic effects"
                },
                "biomarkers": {
                    "description": "Biomarker discovery",
                    "implementation": "Identify sensitivity markers"
                }
            },
            "model_extensions": {
                "multi_modal": {
                    "description": "Multi-modal inputs",
                    "implementation": "Integrate proteomics and metabolomics"
                },
                "time_series": {
                    "description": "Time-series predictions",
                    "implementation": "Model temporal drug effects"
                },
                "mechanistic": {
                    "description": "Integration with mechanistic models",
                    "implementation": "Combine with pathway models"
                }
            }
        }

    def generate_mermaid_diagram(self) -> str:
        """Generate comprehensive Mermaid diagram based on actual method content"""
        architecture = self.final_plan["model_architecture"]
        data_processing = self.final_plan["data_processing"]
        training = self.final_plan["training_strategy"]
        evaluation = self.final_plan["evaluation_metrics"]

        diagram = [
            "# Research Framework Architecture",
            "",
            "## System Overview",
            "This diagram shows the complete research framework based on expert discussions and method refinement.",
            "",
            "```mermaid",
            "graph TD",
            ""
        ]

        # Generate architecture based on actual content
        diagram.extend(self._generate_architecture_from_content(architecture, data_processing, training, evaluation))

        diagram.extend([
            "```",
            "",
            "## Architecture Details",
            ""
        ])

        # Add detailed component information
        diagram.extend(self._generate_detailed_component_diagram(architecture, data_processing, training, evaluation))

        return "\n".join(diagram)

    def _generate_architecture_from_content(self, architecture, data_processing, training, evaluation):
        """Generate architecture diagram based on actual method content"""
        diagram_lines = []

        # Data Input Section
        diagram_lines.extend([
            "    %% Data Input",
            "    subgraph DataInput [\"Data Input\"]",
            "        RawData[\"Raw Single-cell Data<br/>• Expression Matrix<br/>• Cell Metadata<br/>• Perturbation Info\"]",
            "        Metadata[\"Metadata<br/>• Cell Types<br/>• Batch Info<br/>• Conditions\"]",
        ])

        # Data Processing Section - based on actual processing steps
        diagram_lines.extend([
            "    end",
            "",
            "    %% Data Processing Pipeline",
            "    subgraph DataProcessing [\"🔧 Data Processing\"]"
        ])

        # Add actual data processing steps
        for step_name, step_config in data_processing.items():
            if isinstance(step_config, dict):
                method = step_config.get("method", step_name)
                params = step_config.get("parameters", {})
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()]) if params else ""

                diagram_lines.append(f"        {step_name}[\"{method}<br/>{param_str}\"]")

        diagram_lines.append("    end")

        # Model Architecture Section - based on actual architecture
        diagram_lines.extend([
            "",
            "    %% Model Architecture",
            "    subgraph ModelArch [\"🧠 Model Architecture\"]"
        ])

        # Add actual model components
        for comp_name, comp_config in architecture.items():
            if isinstance(comp_config, dict):
                comp_type = comp_config.get("type", comp_name)

                # Add main component
                diagram_lines.append(f"        {comp_name}[\"{comp_type}\"]")

                # Add sub-components if they exist
                if "components" in comp_config:
                    for subcomp_name, subcomp_config in comp_config["components"].items():
                        if isinstance(subcomp_config, dict):
                            subcomp_type = subcomp_config.get("type", subcomp_name)
                            features = subcomp_config.get("features", [])
                            feature_str = "<br/>• " + "<br/>• ".join(features[:3]) if features else ""

                            diagram_lines.append(f"        {subcomp_name}[\"{subcomp_type}{feature_str}\"]")
                            diagram_lines.append(f"        {comp_name} --> {subcomp_name}")

        diagram_lines.append("    end")

        # Training Section - based on actual training strategy
        diagram_lines.extend([
            "",
            "    %% Training Strategy",
            "    subgraph Training [\"🎯 Training Strategy\"]"
        ])

        for train_comp, train_config in training.items():
            if isinstance(train_config, dict):
                train_type = train_config.get("type", train_comp)
                params = train_config.get("parameters", {})
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()]) if params else ""

                diagram_lines.append(f"        {train_comp}[\"{train_type}<br/>{param_str}\"]")

        diagram_lines.append("    end")

        # Evaluation Section - based on actual evaluation metrics
        diagram_lines.extend([
            "",
            "    %% Evaluation Metrics",
            "    subgraph Evaluation [\"📈 Evaluation\"]"
        ])

        for eval_level, eval_config in evaluation.items():
            if isinstance(eval_config, dict):
                metrics = eval_config.get("metrics", [])
                metric_str = "<br/>• " + "<br/>• ".join(metrics) if metrics else ""

                diagram_lines.append(f"        {eval_level}[\"{eval_level.title()}{metric_str}\"]")

        diagram_lines.append("    end")

        # Output Section
        diagram_lines.extend([
            "",
            "    %% Output",
            "    subgraph Output [\"📋 Output\"]",
            "        Predictions[\"Predictions<br/>• Expression Changes<br/>• State Transitions<br/>• Pathway Activities\"]",
            "        Insights[\"Biological Insights<br/>• Mechanism of Action<br/>• Drug Targets<br/>• Biomarkers\"]",
            "    end",
            ""
        ])

        # Data Flow Connections - based on actual pipeline
        diagram_lines.extend([
            "    %% Data Flow",
            "    RawData --> DataProcessing",
            "    Metadata --> DataProcessing"
        ])

        # Connect data processing steps
        prev_step = None
        for step_name in data_processing.keys():
            if prev_step:
                diagram_lines.append(f"    {prev_step} --> {step_name}")
            prev_step = step_name

        if prev_step:
            diagram_lines.append(f"    {prev_step} --> ModelArch")

        # Connect model components
        diagram_lines.append("    ModelArch --> Training")
        diagram_lines.append("    Training --> Evaluation")
        diagram_lines.append("    Evaluation --> Output")

        # Styling
        diagram_lines.extend([
            "",
            "    %% Styling",
            "    classDef inputStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px",
            "    classDef processingStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px",
            "    classDef modelStyle fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px",
            "    classDef trainingStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px",
            "    classDef evaluationStyle fill:#fce4ec,stroke:#880e4f,stroke-width:2px",
            "    classDef outputStyle fill:#e0f2f1,stroke:#004d40,stroke-width:2px",
            "    classDef subgraphStyle fill:#fafafa,stroke:#666,stroke-width:1px",
            "",
            "    class RawData,Metadata inputStyle",
        ])

        # Apply styles to actual components
        for step_name in data_processing.keys():
            diagram_lines.append(f"    class {step_name} processingStyle")

        for comp_name in architecture.keys():
            diagram_lines.append(f"    class {comp_name} modelStyle")

        for train_comp in training.keys():
            diagram_lines.append(f"    class {train_comp} trainingStyle")

        for eval_level in evaluation.keys():
            diagram_lines.append(f"    class {eval_level} evaluationStyle")

        diagram_lines.extend([
            "    class Predictions,Insights outputStyle",
            "    class DataInput,DataProcessing,ModelArch,Training,Evaluation,Output subgraphStyle"
        ])

        return diagram_lines

    def _generate_detailed_component_diagram(self, architecture, data_processing, training, evaluation):
        """Generate detailed component diagram with specific implementations"""
        detailed_diagram = [
            "",
            "## Detailed Component Architecture",
            "",
            "```mermaid",
            "graph LR",
            ""
        ]

        # Generate detailed components based on actual content
        detailed_diagram.extend(self._generate_detailed_components(architecture, data_processing, training, evaluation))

        detailed_diagram.extend([
            "```",
            "",
            "## Implementation Specifications",
            ""
        ])

        # Add specific implementation details
        detailed_diagram.extend(self._add_implementation_details(architecture, data_processing, training, evaluation))

        return detailed_diagram

    def _generate_detailed_components(self, architecture, data_processing, training, evaluation):
        """Generate detailed component diagram based on actual content"""
        diagram_lines = []

        # Encoder Details
        if "encoder" in architecture:
            encoder_config = architecture["encoder"]
            diagram_lines.extend([
                "    %% Encoder Details",
                "    subgraph EncoderDetails [\"🔍 Encoder Details\"]"
            ])

            if "components" in encoder_config:
                for comp_name, comp_config in encoder_config["components"].items():
                    if isinstance(comp_config, dict):
                        comp_type = comp_config.get("type", comp_name)
                        features = comp_config.get("features", [])
                        feature_str = "<br/>• " + "<br/>• ".join(features[:2]) if features else ""

                        diagram_lines.append(f"        {comp_name}[\"{comp_type}{feature_str}\"]")

            diagram_lines.append("    end")

        # Decoder Details
        if "decoder" in architecture:
            decoder_config = architecture["decoder"]
            diagram_lines.extend([
                "",
                "    %% Decoder Details",
                "    subgraph DecoderDetails [\"🎯 Decoder Details\"]"
            ])

            if "components" in decoder_config:
                for comp_name, comp_config in decoder_config["components"].items():
                    if isinstance(comp_config, dict):
                        comp_type = comp_config.get("type", comp_name)
                        features = comp_config.get("features", [])
                        feature_str = "<br/>• " + "<br/>• ".join(features[:2]) if features else ""

                        diagram_lines.append(f"        {comp_name}[\"{comp_type}{feature_str}\"]")

            diagram_lines.append("    end")

        # Training Details
        diagram_lines.extend([
            "",
            "    %% Training Details",
            "    subgraph TrainingDetails [\"⚙️ Training Details\"]"
        ])

        for train_comp, train_config in training.items():
            if isinstance(train_config, dict):
                train_type = train_config.get("type", train_comp)
                params = train_config.get("parameters", {})
                param_str = "<br/>" + ", ".join([f"{k}={v}" for k, v in params.items()]) if params else ""

                diagram_lines.append(f"        {train_comp}[\"{train_type}{param_str}\"]")

        diagram_lines.append("    end")

        # Evaluation Details
        diagram_lines.extend([
            "",
            "    %% Evaluation Details",
            "    subgraph EvaluationDetails [\"📊 Evaluation Details\"]"
        ])

        for eval_level, eval_config in evaluation.items():
            if isinstance(eval_config, dict):
                metrics = eval_config.get("metrics", [])
                metric_str = "<br/>• " + "<br/>• ".join(metrics[:3]) if metrics else ""

                diagram_lines.append(f"        {eval_level}[\"{eval_level.title()}{metric_str}\"]")

        diagram_lines.append("    end")

        # Connections based on actual relationships
        diagram_lines.extend([
            "",
            "    %% Component Connections"
        ])

        # Connect encoder components
        if "encoder" in architecture and "components" in architecture["encoder"]:
            encoder_comps = list(architecture["encoder"]["components"].keys())
            for i in range(len(encoder_comps) - 1):
                diagram_lines.append(f"    {encoder_comps[i]} --> {encoder_comps[i+1]}")

        # Connect decoder components
        if "decoder" in architecture and "components" in architecture["decoder"]:
            decoder_comps = list(architecture["decoder"]["components"].keys())
            for i in range(len(decoder_comps) - 1):
                diagram_lines.append(f"    {decoder_comps[i]} --> {decoder_comps[i+1]}")

        # Connect training components
        train_comps = list(training.keys())
        for i in range(len(train_comps) - 1):
            diagram_lines.append(f"    {train_comps[i]} --> {train_comps[i+1]}")

        # Connect evaluation components
        eval_comps = list(evaluation.keys())
        for i in range(len(eval_comps) - 1):
            diagram_lines.append(f"    {eval_comps[i]} --> {eval_comps[i+1]}")

        # Cross-connections
        if "encoder" in architecture and "decoder" in architecture:
            diagram_lines.append("    EncoderDetails --> DecoderDetails")
        diagram_lines.append("    DecoderDetails --> TrainingDetails")
        diagram_lines.append("    TrainingDetails --> EvaluationDetails")

        # Styling
        diagram_lines.extend([
            "",
            "    %% Styling",
            "    classDef encoderStyle fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px",
            "    classDef decoderStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px",
            "    classDef trainingStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px",
            "    classDef evaluationStyle fill:#fce4ec,stroke:#880e4f,stroke-width:2px",
            ""
        ])

        # Apply styles to actual components
        if "encoder" in architecture and "components" in architecture["encoder"]:
            for comp_name in architecture["encoder"]["components"].keys():
                diagram_lines.append(f"    class {comp_name} encoderStyle")

        if "decoder" in architecture and "components" in architecture["decoder"]:
            for comp_name in architecture["decoder"]["components"].keys():
                diagram_lines.append(f"    class {comp_name} decoderStyle")

        for train_comp in training.keys():
            diagram_lines.append(f"    class {train_comp} trainingStyle")

        for eval_level in evaluation.keys():
            diagram_lines.append(f"    class {eval_level} evaluationStyle")

        return diagram_lines

    def generate_plan_markdown(self) -> str:
        return f"""


Generated on: {self.final_plan['timestamp']}


{self.generate_mermaid_diagram()}


{self._format_dict(self.final_plan['model_architecture'])}


{self._format_dict(self.final_plan['data_processing'])}


{self._format_dict(self.final_plan['training_strategy'])}


{self._format_dict(self.final_plan['evaluation_metrics'])}


{self._format_dict(self.final_plan['implementation_details'])}


{self._format_dict(self.final_plan['biological_validation'])}


{self._format_dict(self.final_plan['computational_optimization'])}


{self._format_dict(self.final_plan['model_assumptions'])}


{self._format_dict(self.final_plan['extensions'])}
"""

    def _format_dict(self, d: Dict[str, Any], indent: int = 0) -> str:
        lines = []
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{'  ' * indent}- {k}:")
                lines.append(self._format_dict(v, indent + 1))
            elif isinstance(v, list):
                lines.append(f"{'  ' * indent}- {k}: {', '.join(map(str, v))}")
            else:
                lines.append(f"{'  ' * indent}- {k}: {v}")
        return "\n".join(lines)

    def save_plan(self, output_path: str, format: str = "markdown"):
        if format == "markdown":
            plan = self.generate_plan_markdown()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(plan)
        elif format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.final_plan, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def print_data_matrix_info(self, matrix: np.ndarray, name: str):
        print(f"\n{name} Matrix Information:")
        print(f"Shape: {matrix.shape}")
        print(f"Data type: {matrix.dtype}")
        print(f"Memory usage: {matrix.nbytes / 1024 / 1024:.2f} MB")
        print(f"Value range: [{matrix.min():.2f}, {matrix.max():.2f}]")
        print(f"Mean: {matrix.mean():.2f}")
        print(f"Std: {matrix.std():.2f}")
        print(f"NaN values: {np.isnan(matrix).sum()}")
        print(f"Zero values: {(matrix == 0).sum()}")

    def _generate_mermaid_json(self) -> Dict[str, Any]:
        """Generate Mermaid diagrams in JSON format with detailed explanations"""
        return {
            "system_overview": {
                "title": "Research Framework Architecture",
                "description": "Complete research framework for biological perturbation analysis using single-cell data",
                "mermaid_code": self._get_system_overview_mermaid(),
                "modules": {
                    "data_input_layer": {
                        "name": "Data Input Layer",
                        "description": "Handles raw single-cell data and metadata",
                        "components": {
                            "raw_data": {
                                "name": "Raw Single-cell Data",
                                "description": "Gene expression matrices, cell metadata, and perturbation information",
                                "format": ["h5ad", "loom", "mtx"],
                                "required_fields": ["counts", "var_names", "obs_names"]
                            },
                            "metadata": {
                                "name": "Metadata",
                                "description": "Cell types, batch information, and experimental conditions",
                                "fields": ["cell_type", "batch", "condition", "perturbation"]
                            }
                        }
                    },
                    "data_processing_layer": {
                        "name": "Data Processing Layer",
                        "description": "Quality control, normalization, and feature selection",
                        "components": {
                            "quality_control": {
                                "name": "Quality Control",
                                "description": "Cell filtering, gene filtering, and doublet detection",
                                "methods": ["scanpy.pp.filter_cells", "scanpy.pp.filter_genes", "scanpy.pp.scrublet"]
                            },
                            "normalization": {
                                "name": "Normalization",
                                "description": "Count normalization, log transformation, and batch correction",
                                "methods": ["scanpy.pp.normalize_total", "scanpy.pp.log1p", "scanpy.pp.regress_out"]
                            },
                            "feature_selection": {
                                "name": "Feature Selection",
                                "description": "Highly variable genes, dimensionality reduction, and feature engineering",
                                "methods": ["scanpy.pp.highly_variable_genes", "scanpy.pp.pca", "scanpy.pp.scale"]
                            }
                        }
                    },
                    "model_architecture_layer": {
                        "name": "Model Architecture Layer",
                        "description": "Neural network architecture for perturbation prediction",
                        "components": {
                            "encoder": {
                                "name": "Encoder",
                                "description": "Transformer/GNN for gene and cell type embedding",
                                "types": ["Transformer", "Graph Neural Network", "Autoencoder"],
                                "features": ["gene_embedding", "cell_embedding", "attention_mechanism"]
                            },
                            "decoder": {
                                "name": "Decoder",
                                "description": "Conditional generation for perturbation prediction and response modeling",
                                "types": ["Conditional Generation", "Trajectory Prediction", "Response Prediction"],
                                "outputs": ["expression_profiles", "cell_states", "pathway_activities"]
                            },
                            "constraints": {
                                "name": "Biological Constraints",
                                "description": "Integration of pathway information, network structure, and cell type specificity",
                                "types": ["gene_regulatory_networks", "pathway_interactions", "cell_type_specificity"]
                            }
                        }
                    },
                    "training_layer": {
                        "name": "Training Layer",
                        "description": "Model training and optimization",
                        "components": {
                            "optimizer": {
                                "name": "Optimizer",
                                "description": "Adam/AdamW with learning rate scheduling and gradient clipping",
                                "types": ["Adam", "AdamW", "SGD"],
                                "schedulers": ["CosineAnnealingLR", "ReduceLROnPlateau", "OneCycleLR"]
                            },
                            "loss_functions": {
                                "name": "Loss Functions",
                                "description": "Reconstruction loss, biological constraints, and regularization",
                                "types": ["MSE", "MAE", "Biological Loss", "Regularization"]
                            },
                            "validation": {
                                "name": "Validation",
                                "description": "Cross-validation, early stopping, and model checkpointing",
                                "methods": ["cross_validation", "early_stopping", "model_checkpointing"]
                            }
                        }
                    },
                    "evaluation_layer": {
                        "name": "Evaluation Layer",
                        "description": "Model evaluation and validation",
                        "components": {
                            "technical_metrics": {
                                "name": "Technical Metrics",
                                "description": "MSE, R², Pearson correlation, and AUROC",
                                "metrics": ["MSE", "R2", "Pearson", "AUROC", "Accuracy"]
                            },
                            "biological_metrics": {
                                "name": "Biological Metrics",
                                "description": "Pathway enrichment, network consistency, and cell type specificity",
                                "metrics": ["pathway_enrichment", "network_consistency", "cell_type_specificity"]
                            },
                            "biological_validation": {
                                "name": "Biological Validation",
                                "description": "Experimental validation, literature comparison, and expert assessment",
                                "methods": ["experimental_validation", "literature_comparison", "expert_assessment"]
                            }
                        }
                    },
                    "output_layer": {
                        "name": "Output Layer",
                        "description": "Predictions and biological insights",
                        "components": {
                            "predictions": {
                                "name": "Predictions",
                                "description": "Gene expression changes, cell state transitions, and pathway activities",
                                "types": ["expression_changes", "state_transitions", "pathway_activities"]
                            },
                            "insights": {
                                "name": "Biological Insights",
                                "description": "Mechanism of action, drug targets, and biomarkers",
                                "types": ["mechanism_of_action", "drug_targets", "biomarkers"]
                            },
                            "applications": {
                                "name": "Applications",
                                "description": "Drug discovery, precision medicine, and biomarker discovery",
                                "types": ["drug_discovery", "precision_medicine", "biomarker_discovery"]
                            }
                        }
                    }
                }
            },
            "detailed_architecture": {
                "title": "Detailed Component Architecture",
                "description": "Detailed view of encoder, decoder, training, and evaluation components",
                "mermaid_code": self._get_detailed_architecture_mermaid(),
                "components": {
                    "encoder_details": {
                        "name": "Encoder Details",
                        "components": {
                            "gene_embedding": {
                                "name": "Gene Embedding",
                                "description": "Positional encoding, gene ontology, pathway information",
                                "features": ["positional_encoding", "gene_ontology", "pathway_information"]
                            },
                            "cell_embedding": {
                                "name": "Cell Embedding",
                                "description": "Cell type markers, state information, batch effects",
                                "features": ["cell_type_markers", "state_information", "batch_effects"]
                            },
                            "attention": {
                                "name": "Multi-head Attention",
                                "description": "Gene-gene interactions, cell-cell relationships, perturbation effects",
                                "features": ["gene_interactions", "cell_relationships", "perturbation_effects"]
                            }
                        }
                    },
                    "decoder_details": {
                        "name": "Decoder Details",
                        "components": {
                            "condition": {
                                "name": "Conditional Input",
                                "description": "Perturbation type, dose information, time point",
                                "features": ["perturbation_type", "dose_information", "time_point"]
                            },
                            "generation": {
                                "name": "Response Generation",
                                "description": "Expression prediction, state transition, pathway activation",
                                "features": ["expression_prediction", "state_transition", "pathway_activation"]
                            },
                            "output": {
                                "name": "Output Processing",
                                "description": "Denormalization, quality control, biological validation",
                                "features": ["denormalization", "quality_control", "biological_validation"]
                            }
                        }
                    },
                    "training_details": {
                        "name": "Training Details",
                        "components": {
                            "data_loader": {
                                "name": "Data Loading",
                                "description": "Batch sampling, augmentation, balancing",
                                "features": ["batch_sampling", "augmentation", "balancing"]
                            },
                            "optimizer": {
                                "name": "Optimization",
                                "description": "Adam optimizer, learning rate decay, weight decay",
                                "features": ["adam_optimizer", "lr_decay", "weight_decay"]
                            },
                            "monitoring": {
                                "name": "Monitoring",
                                "description": "Loss tracking, gradient norms, validation metrics",
                                "features": ["loss_tracking", "gradient_norms", "validation_metrics"]
                            }
                        }
                    },
                    "evaluation_details": {
                        "name": "Evaluation Details",
                        "components": {
                            "tech_eval": {
                                "name": "Technical Evaluation",
                                "description": "Reconstruction error, prediction accuracy, generalization",
                                "features": ["reconstruction_error", "prediction_accuracy", "generalization"]
                            },
                            "bio_eval": {
                                "name": "Biological Evaluation",
                                "description": "Pathway enrichment, network analysis, literature validation",
                                "features": ["pathway_enrichment", "network_analysis", "literature_validation"]
                            },
                            "robustness": {
                                "name": "Robustness Tests",
                                "description": "Cross-validation, ablation studies, sensitivity analysis",
                                "features": ["cross_validation", "ablation_studies", "sensitivity_analysis"]
                            }
                        }
                    }
                }
            }
        }

    def _get_system_overview_mermaid(self) -> str:
        """Get the system overview Mermaid diagram code"""
        return '''graph TD
    %% Data Input Layer
    subgraph InputLayer ["📊 Data Input Layer"]
        RawData["Raw Single-cell Data<br/>• Gene expression matrices<br/>• Cell metadata<br/>• Perturbation information"]
        Metadata["Metadata<br/>• Cell types<br/>• Batch information<br/>• Experimental conditions"]
    end

    %% Data Processing Layer
    subgraph ProcessingLayer ["🔧 Data Processing Layer"]
        QC["Quality Control<br/>• Cell filtering<br/>• Gene filtering<br/>• Doublet detection"]
        Norm["Normalization<br/>• Count normalization<br/>• Log transformation<br/>• Batch correction"]
        Feature["Feature Selection<br/>• Highly variable genes<br/>• Dimensionality reduction<br/>• Feature engineering"]
    end

    %% Model Architecture Layer
    subgraph ModelLayer ["🧠 Model Architecture Layer"]
        Encoder["Encoder<br/>• Transformer/GNN<br/>• Gene embedding<br/>• Cell type embedding"]
        Decoder["Decoder<br/>• Conditional generation<br/>• Perturbation prediction<br/>• Response modeling"]
        Constraints["Biological Constraints<br/>• Pathway information<br/>• Network structure<br/>• Cell type specificity"]
    end

    %% Training Layer
    subgraph TrainingLayer ["🎯 Training Layer"]
        Optimizer["Optimizer<br/>• Adam/AdamW<br/>• Learning rate scheduling<br/>• Gradient clipping"]
        Loss["Loss Functions<br/>• Reconstruction loss<br/>• Biological constraints<br/>• Regularization"]
        Validation["Validation<br/>• Cross-validation<br/>• Early stopping<br/>• Model checkpointing"]
    end

    %% Evaluation Layer
    subgraph EvaluationLayer ["📈 Evaluation Layer"]
        TechMetrics["Technical Metrics<br/>• MSE, R²<br/>• Pearson correlation<br/>• AUROC"]
        BioMetrics["Biological Metrics<br/>• Pathway enrichment<br/>• Network consistency<br/>• Cell type specificity"]
        Validation["Biological Validation<br/>• Experimental validation<br/>• Literature comparison<br/>• Expert assessment"]
    end

    %% Output Layer
    subgraph OutputLayer ["📋 Output Layer"]
        Predictions["Predictions<br/>• Gene expression changes<br/>• Cell state transitions<br/>• Pathway activities"]
        Insights["Biological Insights<br/>• Mechanism of action<br/>• Drug targets<br/>• Biomarkers"]
        Applications["Applications<br/>• Drug discovery<br/>• Precision medicine<br/>• Biomarker discovery"]
    end

    %% Data Flow Connections
    RawData --> QC
    Metadata --> QC
    QC --> Norm
    Norm --> Feature
    Feature --> Encoder
    Encoder --> Decoder
    Constraints --> Decoder
    Decoder --> Optimizer
    Optimizer --> Loss
    Loss --> Validation
    Validation --> TechMetrics
    TechMetrics --> BioMetrics
    BioMetrics --> Validation
    Validation --> Predictions
    Predictions --> Insights
    Insights --> Applications

    %% Styling
    classDef inputStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processingStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef modelStyle fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef trainingStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef evaluationStyle fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef outputStyle fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    classDef subgraphStyle fill:#fafafa,stroke:#666,stroke-width:1px

    class RawData,Metadata inputStyle
    class QC,Norm,Feature processingStyle
    class Encoder,Decoder,Constraints modelStyle
    class Optimizer,Loss,Validation trainingStyle
    class TechMetrics,BioMetrics,Validation evaluationStyle
    class Predictions,Insights,Applications outputStyle
    class InputLayer,ProcessingLayer,ModelLayer,TrainingLayer,EvaluationLayer,OutputLayer subgraphStyle'''

    def _get_detailed_architecture_mermaid(self) -> str:
        """Get the detailed architecture Mermaid diagram code"""
        return '''graph LR
    %% Detailed Architecture Components
    subgraph EncoderDetails ["🔍 Encoder Details"]
        GeneEmbed["Gene Embedding<br/>• Positional encoding<br/>• Gene ontology<br/>• Pathway information"]
        CellEmbed["Cell Embedding<br/>• Cell type markers<br/>• State information<br/>• Batch effects"]
        Attention["Multi-head Attention<br/>• Gene-gene interactions<br/>• Cell-cell relationships<br/>• Perturbation effects"]
    end

    subgraph DecoderDetails ["🎯 Decoder Details"]
        Condition["Conditional Input<br/>• Perturbation type<br/>• Dose information<br/>• Time point"]
        Generation["Response Generation<br/>• Expression prediction<br/>• State transition<br/>• Pathway activation"]
        Output["Output Processing<br/>• Denormalization<br/>• Quality control<br/>• Biological validation"]
    end

    subgraph TrainingDetails ["⚙️ Training Details"]
        DataLoader["Data Loading<br/>• Batch sampling<br/>• Augmentation<br/>• Balancing"]
        Optimizer["Optimization<br/>• Adam optimizer<br/>• Learning rate decay<br/>• Weight decay"]
        Monitoring["Monitoring<br/>• Loss tracking<br/>• Gradient norms<br/>• Validation metrics"]
    end

    subgraph EvaluationDetails ["📊 Evaluation Details"]
        TechEval["Technical Evaluation<br/>• Reconstruction error<br/>• Prediction accuracy<br/>• Generalization"]
        BioEval["Biological Evaluation<br/>• Pathway enrichment<br/>• Network analysis<br/>• Literature validation"]
        Robustness["Robustness Tests<br/>• Cross-validation<br/>• Ablation studies<br/>• Sensitivity analysis"]
    end

    %% Connections
    GeneEmbed --> Attention
    CellEmbed --> Attention
    Attention --> Condition
    Condition --> Generation
    Generation --> Output
    Output --> DataLoader
    DataLoader --> Optimizer
    Optimizer --> Monitoring
    Monitoring --> TechEval
    TechEval --> BioEval
    BioEval --> Robustness

    %% Styling
    classDef encoderStyle fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef decoderStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef trainingStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef evaluationStyle fill:#fce4ec,stroke:#880e4f,stroke-width:2px

    class GeneEmbed,CellEmbed,Attention encoderStyle
    class Condition,Generation,Output decoderStyle
    class DataLoader,Optimizer,Monitoring trainingStyle
    class TechEval,BioEval,Robustness evaluationStyle'''

    def _generate_framework_overview(self) -> Dict[str, Any]:
        """Generate framework overview information"""
        return {
            "title": "Biological Perturbation Analysis Framework",
            "description": "A comprehensive framework for analyzing biological perturbations using single-cell data",
            "key_features": [
                "Multi-expert knowledge integration",
                "RAG-enhanced knowledge retrieval",
                "Biological constraint incorporation",
                "Comprehensive evaluation metrics",
                "Scalable architecture design"
            ],
            "supported_tasks": [
                "gene_knockout",
                "drug_perturbation",
                "cytokine_stimulation"
            ],
            "output_formats": [
                "markdown",
                "json",
                "mermaid",
                "png"
            ],
            "expert_domains": [
                "Data Engineering",
                "Single Cell Biology",
                "Deep Learning",
                "Molecular Biology",
                "Bioinformatics",
                "Statistics",
                "Drug Discovery",
                "Gene Regulation",
                "Experimental Design",
                "Validation",
                "Baseline Assessment",
                "Training",
                "Drug Response",
                "Pathway Analysis",
                "Cell Communication",
                "Omics Integration"
            ]
        }

    def _add_implementation_details(self, architecture, data_processing, training, evaluation):
        """Add specific implementation details to the diagram"""
        details = [
            "### Model Architecture Specifications",
            ""
        ]

        # Add architecture details
        for component, config in architecture.items():
            if isinstance(config, dict):
                details.append(f"#### {component.replace('_', ' ').title()}")
                for key, value in config.items():
                    if isinstance(value, dict):
                        details.append(f"- **{key}**:")
                        for subkey, subvalue in value.items():
                            details.append(f"  - {subkey}: {subvalue}")
                    else:
                        details.append(f"- **{key}**: {value}")
                details.append("")

        details.extend([
            "### Data Processing Pipeline",
            ""
        ])

        # Add data processing details
        for step, config in data_processing.items():
            if isinstance(config, dict):
                details.append(f"#### {step.replace('_', ' ').title()}")
                for key, value in config.items():
                    if isinstance(value, dict):
                        details.append(f"- **{key}**:")
                        for subkey, subvalue in value.items():
                            details.append(f"  - {subkey}: {subvalue}")
                    else:
                        details.append(f"- **{key}**: {value}")
                details.append("")

        details.extend([
            "### Training Configuration",
            ""
        ])

        # Add training details
        for component, config in training.items():
            if isinstance(config, dict):
                details.append(f"#### {component.replace('_', ' ').title()}")
                for key, value in config.items():
                    if isinstance(value, dict):
                        details.append(f"- **{key}**:")
                        for subkey, subvalue in value.items():
                            details.append(f"  - {subkey}: {subvalue}")
                    else:
                        details.append(f"- **{key}**: {value}")
                details.append("")

        details.extend([
            "### Evaluation Metrics",
            ""
        ])

        # Add evaluation details
        for level, config in evaluation.items():
            if isinstance(config, dict):
                details.append(f"#### {level.replace('_', ' ').title()}")
                for key, value in config.items():
                    if isinstance(value, dict):
                        details.append(f"- **{key}**:")
                        for subkey, subvalue in value.items():
                            details.append(f"  - {subkey}: {subvalue}")
                    else:
                        details.append(f"- **{key}**: {value}")
                details.append("")

        return details
