from typing import Dict, Any, List
from datetime import datetime
import json
from ..paths import data_path
from .data_structures import AnalysisResult
from .knowledge_base import knowledge_base
from ..llm import LLMInterface

class BaselineAssessor:
    """Expert agent for assessing baseline models and evaluation strategies in single cell perturbation prediction"""

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.llm = LLMInterface()

    def assess_baselines(self, task_description: str, dataset_info: Dict[str, Any],
                        retrieved_papers: List[Dict[str, Any]]) -> AnalysisResult:
        """
        Assess baseline models and evaluation strategies using knowledge base

        Args:
            task_description: Description of the research task
            dataset_info: Dictionary containing dataset metadata
            retrieved_papers: List of relevant papers from vector database

        Returns:
            AnalysisResult with baseline assessment
        """
        # 使用knowledge base而不是重复搜索
        knowledge_items = knowledge_base.search_both_databases(
            knowledge_type="papers",
            query=task_description,
            limit=10
        )

        # 获取评估框架信息
        evaluation_items = knowledge_base.search_both_databases(
            knowledge_type="evaluation_frameworks",
            query=task_description,
            limit=5
        )

        # 获取实现指南信息
        implementation_items = knowledge_base.search_both_databases(
            knowledge_type="implementation_guides",
            query=task_description,
            limit=5
        )

        # 合并所有论文信息
        all_papers = retrieved_papers + [
            {
                "title": item.content.get("title", ""),
                "abstract": item.content.get("content", item.content.get("snippet", "")),
                "metadata": item.metadata,
                "relevance_score": item.relevance_score
            }
            for item in knowledge_items
        ]

        # 合并评估框架信息
        evaluation_frameworks = [item.content for item in evaluation_items]

        # 合并实现指南信息
        implementation_guides = [item.content for item in implementation_items]

        # Format prompt with task information and retrieved papers
        prompt = self._format_prompt_with_evaluation_frameworks(
            task_description, dataset_info, all_papers, evaluation_frameworks, implementation_guides
        )

        # Run assessment (implementation depends on your LLM backend)
        assessment_content = self._run_llm(prompt)

        return AnalysisResult(
            content=assessment_content,
            confidence_score=1.0,  # 临时设置，后续会由其他模块计算
            timestamp=datetime.now(),
            metadata={
                "agent": "Baseline Assessor",
                "knowledge_base_usage": {
                    "papers_count": len(knowledge_items),
                    "evaluation_frameworks_count": len(evaluation_items),
                    "implementation_guides_count": len(implementation_items),
                    "total_retrieved": len(all_papers)
                }
            }
        )

    def _extract_method_info(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Automatically extract method information (e.g., scGPT, Random Forest, etc.) from retrieved papers and analyze their principle, strengths, and limitations.
        """
        method_infos = []
        known_methods = {
            "scgpt": {
                "keywords": ["scgpt", "single-cell gpt", "foundation model"],
                "principle": "Transformer-based foundation model for single-cell gene expression prediction.",
            },
            "random forest": {
                "keywords": ["random forest"],
                "principle": "Ensemble tree-based method for regression and classification tasks.",
            },
            "linear regression": {
                "keywords": ["linear regression"],
                "principle": "Linear model for regression tasks.",
            },

        }
        for paper in papers:
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()
            for method, info in known_methods.items():
                if any(kw in title or kw in abstract for kw in info["keywords"]):
                    method_infos.append({
                        "name": method,
                        "principle": info["principle"],
                        "paper_title": paper.get("title", ""),
                        "paper_abstract": paper.get("abstract", ""),
                        "limitations": self._analyze_limitations(abstract, method),
                    })
        return method_infos

    def _analyze_limitations(self, abstract: str, method: str) -> str:
        """
        Simple analysis of method limitations (can be enhanced with LLM in the future)
        """

        if method == "scgpt":
            if "interpret" in abstract or "black box" in abstract:
                return "Potential limitation in biological interpretability."
            return "May require large data and computational resources; interpretability can be limited."
        if method == "random forest":
            return "Limited in capturing complex nonlinear relationships compared to deep learning."
        if method == "linear regression":
            return "Assumes linearity; may underperform on complex biological data."
        return "See paper for details."

    def _format_prompt_with_evaluation_frameworks(self, task_description: str, dataset_info: Dict[str, Any],
                                                retrieved_papers: List[Dict[str, Any]],
                                                evaluation_frameworks: List[Dict[str, Any]],
                                                implementation_guides: List[Dict[str, Any]]) -> str:
        """Format prompt with evaluation frameworks and implementation guides"""

        # Format paper context
        papers_context = "\n".join([
            f"- {paper.get('title', 'No title')}: {paper.get('abstract', paper.get('content', paper.get('snippet', 'No content')))[:200]}..."
            for paper in retrieved_papers[:5]
        ])

        # Format evaluation framework information
        framework_context = ""
        if evaluation_frameworks:
            framework_context = f"""
EVALUATION FRAMEWORKS:
{chr(10).join([f"- {framework.get('title', 'No title')}: {framework.get('content', 'No content')[:200]}..." for framework in evaluation_frameworks[:3]])}
"""

        # Format implementation guide information
        guide_context = ""
        if implementation_guides:
            guide_context = f"""
IMPLEMENTATION GUIDES:
{chr(10).join([f"- {guide.get('title', 'No title')}: {guide.get('content', 'No content')[:200]}..." for guide in implementation_guides[:3]])}
"""

        # Extract method information
        method_infos = self._extract_method_info(retrieved_papers)
        if method_infos:
            method_analysis = "\n".join([
                f"Method: {m['name'].title()}\n  Principle: {m['principle']}\n  Paper: {m['paper_title']}\n  Limitations: {m['limitations']}"
                for m in method_infos
            ])
        else:
            method_analysis = "No major methods identified in the retrieved literature."

        return f"""You are an expert in deep learning model evaluation and benchmarking for single cell perturbation prediction, with extensive experience in assessing machine learning models for biological applications. Your task is to provide a comprehensive assessment of baseline models and evaluation strategies, focusing on model analysis, evaluation framework, performance analysis, and improvement suggestions.

{framework_context}

{guide_context}

Dynamic Analysis of Existing Methods (auto-extracted from retrieved literature):
{method_analysis}

1. Analyze Baseline Models for Single Cell Perturbation Prediction:
   - Review existing deep learning methods for perturbation prediction
   - Assess model architectures and their biological interpretability
   - Evaluate implementation details and computational requirements
   - Compare performance metrics and biological relevance
   - Identify limitations and potential improvements

2. Design Evaluation Framework for Perturbation Prediction:
   - Define evaluation metrics with biological significance
   - Plan validation strategy considering biological variability
   - Design test scenarios covering edge cases
   - Establish benchmarks with state-of-the-art methods
   - Consider biological relevance and interpretability

3. Conduct Performance Analysis:
   - Analyze model performance across different perturbation types
   - Compare with baselines and state-of-the-art methods
   - Identify strengths/weaknesses in biological context
   - Assess generalization to new cell types/conditions
   - Evaluate robustness to biological and technical variations

4. Provide Improvement Suggestions:
   - Identify weaknesses in current approaches
   - Propose enhancements based on biological insights
   - Suggest optimizations for better performance
   - Address limitations in biological interpretation
   - Plan future work with biological validation

Task Description:
{task_description}

Dataset Information:
{json.dumps(dataset_info, indent=2)}

Relevant Literature:
{papers_context}

Please provide a comprehensive assessment in the following JSON format:
{{
    "baseline_models": {{
        "literature_review": {{
            "existing_methods": {{
                "methods": ["string"],
                "strengths": ["string"],
                "limitations": ["string"],
                "biological_relevance": "string"
            }},
            "model_comparison": {{
                "comparison_criteria": ["string"],
                "performance_metrics": ["string"],
                "biological_interpretability": "string",
                "computational_requirements": "string"
            }},
            "recommendations": {{
                "model_selection": ["string"],
                "implementation_priority": ["string"],
                "biological_validation": "string"
            }}
        }},
        "evaluation_framework": {{
            "metrics": {{
                "primary_metrics": ["string"],
                "secondary_metrics": ["string"],
                "biological_metrics": ["string"],
                "validation_metrics": ["string"]
            }},
            "validation_strategy": {{
                "cross_validation": "string",
                "biological_validation": "string",
                "independent_testing": "string",
                "robustness_testing": "string"
            }},
            "benchmarks": {{
                "baseline_models": ["string"],
                "state_of_the_art": ["string"],
                "biological_benchmarks": ["string"],
                "performance_targets": ["string"]
            }}
        }},
        "performance_analysis": {{
            "model_performance": {{
                "accuracy_metrics": ["string"],
                "biological_relevance": "string",
                "generalization": "string",
                "robustness": "string"
            }},
            "comparative_analysis": {{
                "baseline_comparison": "string",
                "state_of_the_art_comparison": "string",
                "biological_comparison": "string",
                "computational_comparison": "string"
            }},
            "strengths_weaknesses": {{
                "strengths": ["string"],
                "weaknesses": ["string"],
                "biological_insights": ["string"],
                "technical_limitations": ["string"]
            }}
        }},
        "improvement_suggestions": {{
            "model_enhancements": {{
                "architectural_improvements": ["string"],
                "training_optimizations": ["string"],
                "biological_interpretability": ["string"],
                "computational_efficiency": ["string"]
            }},
            "implementation_optimizations": {{
                "code_optimizations": ["string"],
                "pipeline_improvements": ["string"],
                "validation_enhancements": ["string"],
                "deployment_considerations": "string"
            }},
            "future_work": {{
                "research_directions": ["string"],
                "biological_validation": ["string"],
                "clinical_translation": ["string"],
                "technical_advancements": ["string"]
            }}
        }}
    }}
}}"""

    def _format_prompt_with_decision_support(self, task_description: str, dataset_info: Dict[str, Any],
                                           retrieved_papers: List[Dict[str, Any]],
                                           code_implementations: List[Dict[str, Any]],
                                           decision_support: Dict[str, Any],
                                           experimental_designs: List[Dict[str, Any]],
                                           evaluation_frameworks: List[Dict[str, Any]],
                                           implementation_guides: List[Dict[str, Any]]) -> str:
        """Format prompt with decision support information"""

        # Format paper context, including decision support information
        papers_context = "\n".join([
            f"- {paper.get('title', 'No title')}: {paper.get('abstract', paper.get('content', paper.get('snippet', 'No content')))[:200]}...\n"
            f"  Perturbation Types: {', '.join(paper.get('metadata', {}).get('perturbation_type', []))}\n"
            f"  Technologies: {', '.join(paper.get('metadata', {}).get('technology', []))}\n"
            f"  Analysis Methods: {', '.join(paper.get('metadata', {}).get('analysis_method', []))}\n"
            f"  Decision Support: {json.dumps(paper.get('decision_support', {}), indent=2)}"
            for paper in retrieved_papers[:5]
        ])

        # Format code context
        code_context = "\n".join([
            f"- {code['title']}: {code['content'][:200]}...\n"
            f"  Implementation Type: {', '.join(code.get('metadata', {}).get('implementation_type', []))}\n"
            f"  Framework: {', '.join(code.get('metadata', {}).get('framework', []))}"
            for code in code_implementations[:3]
        ])

        # Format decision support information
        decision_context = ""
        if decision_support:
            decision_context = f"""
DECISION SUPPORT INFORMATION:
Model Selection: {json.dumps(decision_support.get('model_selection', {}), indent=2)}
Evaluation Strategy: {json.dumps(decision_support.get('evaluation_strategy', {}), indent=2)}
Data Preparation: {json.dumps(decision_support.get('data_preparation', {}), indent=2)}
Implementation Plan: {json.dumps(decision_support.get('implementation_plan', {}), indent=2)}
Risk Assessment: {json.dumps(decision_support.get('risk_assessment', {}), indent=2)}
"""

        # Format experimental design information
        design_context = ""
        if experimental_designs:
            design_context = f"""
EXPERIMENTAL DESIGNS:
{chr(10).join([f"- {design['title']}: {design['content'][:200]}..." for design in experimental_designs[:3]])}
"""

        # Format evaluation framework information
        framework_context = ""
        if evaluation_frameworks:
            framework_context = f"""
EVALUATION FRAMEWORKS:
{chr(10).join([f"- {framework['title']}: {framework['content'][:200]}..." for framework in evaluation_frameworks[:3]])}
"""

        # Format implementation guide information
        guide_context = ""
        if implementation_guides:
            guide_context = f"""
IMPLEMENTATION GUIDES:
{chr(10).join([f"- {guide['title']}: {guide['content'][:200]}..." for guide in implementation_guides[:3]])}
"""

        # Extract method information
        method_infos = self._extract_method_info(retrieved_papers)
        if method_infos:
            method_analysis = "\n".join([
                f"Method: {m['name'].title()}\n  Principle: {m['principle']}\n  Paper: {m['paper_title']}\n  Limitations: {m['limitations']}"
                for m in method_infos
            ])
        else:
            method_analysis = "No major methods identified in the retrieved literature."

        return f"""You are an expert in deep learning model evaluation and benchmarking for single cell perturbation prediction, with extensive experience in assessing machine learning models for biological applications. Your task is to provide a comprehensive assessment of baseline models and evaluation strategies, focusing on model analysis, evaluation framework, performance analysis, and improvement suggestions.

{decision_context}

{design_context}

{framework_context}

{guide_context}

Dynamic Analysis of Existing Methods (auto-extracted from retrieved literature):
{method_analysis}

1. Analyze Baseline Models for Single Cell Perturbation Prediction:
   - Review existing deep learning methods for perturbation prediction
   - Assess model architectures and their biological interpretability
   - Evaluate implementation details and computational requirements
   - Compare performance metrics and biological relevance
   - Identify limitations and potential improvements
   - Consider decision support recommendations for model selection

2. Design Evaluation Framework for Perturbation Prediction:
   - Define evaluation metrics with biological significance
   - Plan validation strategy considering biological variability
   - Design test scenarios covering edge cases
   - Establish benchmarks with state-of-the-art methods
   - Consider biological relevance and interpretability
   - Incorporate decision support recommendations for evaluation strategy

3. Conduct Performance Analysis:
   - Analyze model performance across different perturbation types
   - Compare with baselines and state-of-the-art methods
   - Identify strengths/weaknesses in biological context
   - Assess generalization to new cell types/conditions
   - Evaluate robustness to biological and technical variations
   - Consider data preparation recommendations

4. Provide Improvement Suggestions:
   - Identify weaknesses in current approaches
   - Propose enhancements based on biological insights
   - Suggest optimizations for better performance
   - Address limitations in biological interpretation
   - Plan future work with biological validation
   - Consider implementation plan and risk assessment

Task Description:
{task_description}

Dataset Information:
{json.dumps(dataset_info, indent=2)}

Relevant Literature:
{papers_context}

Code Implementations:
{code_context}

Please provide a comprehensive assessment in the following JSON format, incorporating decision support information:
{{
    "baseline_models": {{
        "literature_review": {{
            "existing_methods": {{
                "methods": ["string"],
                "strengths": ["string"],
                "limitations": ["string"],
                "biological_relevance": "string",
                "decision_support_integration": "string"
            }},
            "model_comparison": {{
                "comparison_criteria": ["string"],
                "performance_metrics": ["string"],
                "biological_interpretability": "string",
                "computational_requirements": "string",
                "decision_support_alignment": "string"
            }},
            "recommendations": {{
                "model_selection": ["string"],
                "implementation_priority": ["string"],
                "biological_validation": "string",
                "decision_support_considerations": "string"
            }}
        }},
        "evaluation_framework": {{
            "metrics": {{
                "primary_metrics": ["string"],
                "secondary_metrics": ["string"],
                "biological_metrics": ["string"],
                "validation_metrics": ["string"],
                "decision_support_metrics": ["string"]
            }},
            "validation_strategy": {{
                "cross_validation": "string",
                "biological_validation": "string",
                "independent_testing": "string",
                "robustness_testing": "string",
                "decision_support_validation": "string"
            }},
            "benchmarks": {{
                "baseline_models": ["string"],
                "state_of_the_art": ["string"],
                "biological_benchmarks": ["string"],
                "performance_targets": ["string"],
                "decision_support_benchmarks": "string"
            }}
        }},
        "performance_analysis": {{
            "model_performance": {{
                "accuracy_metrics": ["string"],
                "biological_relevance": "string",
                "generalization": "string",
                "robustness": "string",
                "decision_support_performance": "string"
            }},
            "comparative_analysis": {{
                "baseline_comparison": "string",
                "state_of_the_art_comparison": "string",
                "biological_comparison": "string",
                "computational_comparison": "string",
                "decision_support_comparison": "string"
            }},
            "strengths_weaknesses": {{
                "strengths": ["string"],
                "weaknesses": ["string"],
                "biological_insights": ["string"],
                "technical_limitations": ["string"],
                "decision_support_insights": "string"
            }}
        }},
        "improvement_suggestions": {{
            "model_enhancements": {{
                "architectural_improvements": ["string"],
                "training_optimizations": ["string"],
                "biological_interpretability": ["string"],
                "computational_efficiency": ["string"],
                "decision_support_enhancements": "string"
            }},
            "implementation_optimizations": {{
                "code_optimizations": ["string"],
                "pipeline_improvements": ["string"],
                "validation_enhancements": ["string"],
                "deployment_considerations": ["string"],
                "decision_support_implementation": "string"
            }},
            "future_work": {{
                "research_directions": ["string"],
                "biological_validation": ["string"],
                "clinical_translation": ["string"],
                "technical_advancements": ["string"],
                "decision_support_future": "string"
            }}
        }}
    }}
}}"""

    def generate_analysis_report(self, task_description: str, dataset_info: Dict[str, Any],
                                retrieved_papers: List[Dict[str, Any]], code_implementations: List[Dict[str, Any]]) -> str:
        """
        Generate a comprehensive analysis report in JSON format, similar to drug-5.md and gene-RNA-2-A.md.
        The report includes dataset characteristics, task requirements, baseline models analysis, and improvement suggestions.
        """

        method_infos = self._extract_method_info(retrieved_papers)


        dataset_section = {
            "source_protocol": dataset_info.get('source', 'N/A'),
            "composition": {
                "cell_lines": dataset_info.get('cell_lines', ['N/A']),
                "perturbagens": dataset_info.get('perturbagens', 'N/A'),
                "dosage": dataset_info.get('dosage', 'N/A'),
                "scale": dataset_info.get('scale', 'N/A'),
                "dimensions": dataset_info.get('dimensions', 'N/A')
            },
            "unique_technical_features": {
                "cellular_indexing": dataset_info.get('cellular_indexing', 'N/A'),
                "nuclear_hashing": dataset_info.get('nuclear_hashing', 'N/A'),
                "single_cell_resolution": dataset_info.get('single_cell_resolution', 'N/A'),
                "technical_quality": dataset_info.get('technical_quality', 'N/A'),
                "library_complexity": dataset_info.get('library_complexity', 'N/A'),
                "reproducibility": dataset_info.get('reproducibility', 'N/A')
            },
            "data_availability_format": {
                "raw_data": dataset_info.get('raw_data', 'N/A'),
                "processed_data": dataset_info.get('processed_data', 'N/A'),
                "file_formats": dataset_info.get('file_formats', 'N/A'),
                "metadata": dataset_info.get('metadata', 'N/A')
            }
        }


        task_section = {
            "core_task_definition": task_description,
            "mathematical_formulation": {
                "baseline_expression": "Let X∈ℝⁿˣᵐ be the baseline expression (n cells, m genes)",
                "perturbation_encoding": "Let P∈ℝⁿˣᵏ be the perturbation encoding (n cells, k perturbation features)",
                "task_definition": "The task is to learn function f such that: Ŷ = f(X, P) approximates true perturbed expression Y",
                "optimization_objective": "min(L(Y, Ŷ)) where L is a suitable loss function (typically MSE)"
            },
            "input_requirements": {
                "baseline_expression": [
                    "Normalized log-counts (log₂(1+CPM))",
                    "Feature selection to 2,000-5,000 highly variable genes (Townes et al., 2019)",
                    "Optional: dimensionality reduction via PCA or autoencoder"
                ],
                "perturbation_encoding": [
                    "Compound identity (one-hot or learned embeddings)",
                    "Compound structure (SMILES, fingerprints, or descriptors)",
                    "Concentration (log₁₀-transformed molarity)",
                    "Optional: prior knowledge (targets, pathways, MOA)"
                ]
            },
            "output_requirements": [
                "Gene-level expression values or log₂(fold-change)",
                "Full transcriptome or selected gene set coverage",
                "Option for uncertainty quantification per prediction"
            ],
            "evaluation_criteria": {
                "primary_metrics": [
                    "MSE across all genes",
                    "Pearson correlation (global and per-gene)",
                    "AUROC for direction of change prediction"
                ],
                "secondary_metrics": [
                    "Pathway enrichment score correlation (hallmark gene sets)",
                    "Performance on top N differentially expressed genes",
                    "Dose-response curve accuracy (EC50 estimation)",
                    "Cell-type-specific effect prediction accuracy"
                ]
            }
        }


        baseline_section = {
            "literature_based_model_candidates": [
                {
                    "model_type": method['name'].title(),
                    "representative_methods": method['principle'],
                    "key_features": method['limitations'],
                    "citations": method['paper_title']
                }
                for method in method_infos
            ]
        }


        improvement_section = {
            "recommendations": [
                {
                    "title": "Factorized Perturbation Embeddings",
                    "approach": "Learn a separate embedding e_g for each guide g. Represent a perturbation set P by a learned nonlinear composition.",
                    "benefit": "Zero-shot support for unseen guide combinations via embedding arithmetic."
                },
                {
                    "title": "Zero-Inflated Negative Binomial (ZINB) Loss",
                    "approach": "Replace MSE with a ZINB loss that models both dropout probability and overdispersion per gene.",
                    "benefit": "Accounts for scRNA-seq technical noise, improving prediction in low-UMI cells."
                },
                {
                    "title": "Learned Dynamic Graph Priors",
                    "approach": "Instead of a fixed PPI graph, learn gene–gene affinity weights from data using a Gaussian kernel on baseline coexpression, then refine during training.",
                    "benefit": "Captures UPR pathway rewiring under CRISPRi; avoids external databases."
                },
                {
                    "title": "Contrastive Pretraining",
                    "approach": "Pretrain an encoder on (x,0) vs. (x,p) pairs with an InfoNCE contrastive loss.",
                    "benefit": "Disentangles baseline state from perturbation effect; enhances generalization."
                },
                {
                    "title": "Neural ODE or OT Trajectory Module",
                    "approach": "For multi-guide dynamics, model latent drift via a neural ODE dz/dt=f(z,h_P) or optimal-transport regularization.",
                    "benefit": "Enforces smooth interpolation/extrapolation between perturbation levels."
                },
                {
                    "title": "Perturbation-Guided Attention Decoder",
                    "approach": "Use a cross-attention layer where query = latent state, key/value = perturbation embedding h_P.",
                    "benefit": "Focuses model capacity on biologically salient genes; improves interpretability."
                }
            ]
        }


        report = {
            "dataset_characteristics": dataset_section,
            "task_requirements": task_section,
            "baseline_models_analysis": baseline_section,
            "improvement_suggestions": improvement_section
        }


        results_dir = data_path("analyses", "unknown_dataset")
        results_dir.mkdir(parents=True, exist_ok=True)

        try:
            report_path = results_dir / "analysis_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            return str(report_path)
        except Exception as e:
            print(f"Warning: Could not save analysis report: {e}")
            # 如果无法保存到文件，返回None
            return None

    def _run_llm(self, prompt: str) -> Dict[str, Any]:
        """Run LLM with retry mechanism for network errors"""
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                print(f"🔄 LLM attempt {attempt + 1}/{max_retries}")

                system_prompt = "You are an expert in single-cell model evaluation. Provide your response in valid JSON format."
                response = self.llm.generate(prompt, system_prompt)
                content = response.get("content") or ""

                # Parse the response content as JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # If direct parsing fails, try to extract JSON from markdown code blocks
                    import re
                    json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(1))
                    else:
                        # Return the raw content if JSON parsing fails
                        return {"content": content, "error": "Failed to parse JSON response"}

            except Exception as e:
                error_msg = str(e)
                if "Connection broken" in error_msg or "InvalidChunkLength" in error_msg:
                    if attempt < max_retries - 1:
                        print(f"⚠️  Network error (attempt {attempt + 1}): {error_msg}")
                        print(f"🔄 Retrying in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                        continue
                    else:
                        print(f"❌ Max retries reached, using fallback response")
                        # 返回fallback响应
                        return {
                            "baseline_models": {
                                "literature_review": {
                                    "existing_methods": {
                                        "methods": ["Neural networks", "Graph neural networks"],
                                        "strengths": ["Good performance", "Scalable"],
                                        "limitations": ["Limited interpretability"],
                                        "biological_relevance": "Moderate"
                                    }
                                }
                            },
                            "evaluation_framework": {
                                "metrics": ["MSE", "Pearson correlation"],
                                "validation": "Cross-validation"
                            },
                            "error": "LLM connection failed, using fallback"
                        }
                else:
                    # 其他错误直接抛出
                    raise Exception(f"LLM generation failed: {error_msg}")

        # 如果所有重试都失败
        return {"content": "LLM generation failed after all retries", "error": "Connection issues"}
