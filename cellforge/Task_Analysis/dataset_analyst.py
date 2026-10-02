from typing import Dict, Any, List
from datetime import datetime
import json

from .data_structures import AnalysisResult
from .knowledge_base import knowledge_base
from ..llm import LLMInterface

class DatasetAnalyst:
    """Expert agent for analyzing dataset characteristics and quality"""

    def __init__(self, rag_system):
        self.rag_system = rag_system

        self.llm = LLMInterface()

    def analyze_dataset(self, task_description: str, dataset_info: Dict[str, Any],
                       retrieved_papers: List[Dict[str, Any]]) -> AnalysisResult:
        """
        Analyze dataset characteristics and quality using knowledge base

        Args:
            task_description: Description of the research task
            dataset_info: Dictionary containing dataset metadata
            retrieved_papers: List of relevant papers from vector database

        Returns:
            AnalysisResult with dataset analysis
        """
        # 使用knowledge base而不是重复搜索
        knowledge_items = knowledge_base.search_both_databases(
            knowledge_type="papers",
            query=task_description,
            limit=10
        )

        # 获取决策支持信息
        decision_support_items = knowledge_base.search_both_databases(
            knowledge_type="decision_support",
            query=task_description,
            limit=5
        )

        # 获取实验设计信息
        experimental_design_items = knowledge_base.search_both_databases(
            knowledge_type="experimental_designs",
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

        # 合并决策支持信息
        decision_support = {}
        for item in decision_support_items:
            decision_support.update(item.content)

        # 合并实验设计信息
        experimental_designs = [item.content for item in experimental_design_items]

        # Format prompt with task information and retrieved papers
        prompt = self._format_prompt_with_decision_support(
            task_description, dataset_info, all_papers, decision_support, experimental_designs
        )

        # Run analysis (implementation depends on your LLM backend)
        analysis_content = self._run_llm(prompt)

        return AnalysisResult(
            content=analysis_content,
            confidence_score=1.0,  # 临时设置，后续会由其他模块计算
            timestamp=datetime.now(),
            metadata={
                "agent": "Dataset Analyst",
                "knowledge_base_usage": {
                    "papers_count": len(knowledge_items),
                    "decision_support_count": len(decision_support_items),
                    "experimental_designs_count": len(experimental_design_items),
                    "total_retrieved": len(all_papers)
                },
                "decision_support": decision_support
            }
        )

    def _format_prompt(self, task_description: str, dataset_info: Dict[str, Any],
                      retrieved_papers: List[Dict[str, Any]]) -> str:
        """Format prompt for dataset analysis"""
        papers_context = "\n".join([
            f"- {paper.get('title', 'No title')}: {paper.get('abstract', paper.get('content', paper.get('snippet', 'No content')))[:200]}..."
            for paper in retrieved_papers[:5]
        ])

        return f"""You are an expert in single-cell dataset analysis, specializing in evaluating single-cell perturbation data. Your task is to provide a comprehensive analysis of dataset characteristics and quality, focusing on experimental design, data properties, preprocessing recommendations, and quality assessment.

1. Analyze Experimental Design:
   - Evaluate experimental setup and biological conditions
   - Assess sample size and biological replicates
   - Review perturbation design and controls
   - Analyze biological variability and batch effects
   - Consider technical and biological validation

2. Assess Data Properties:
   - Analyze data distribution and biological patterns
   - Evaluate feature space and biological relevance
   - Assess data quality and technical artifacts
   - Identify biological and technical noise
   - Consider data sparsity and missing values

3. Provide Preprocessing Recommendations:
   - Suggest normalization strategies for biological data
   - Recommend quality control procedures
   - Propose feature selection methods
   - Design batch effect correction approaches
   - Plan biological validation steps

4. Conduct Quality Assessment:
   - Evaluate data completeness and coverage
   - Assess biological relevance and interpretability
   - Analyze technical quality and reproducibility
   - Consider biological validation requirements
   - Identify potential limitations and biases

Task Description:
{task_description}

Dataset Information:
{json.dumps(dataset_info, indent=2)}

Relevant Literature:
{papers_context}

Please provide a comprehensive analysis in the following JSON format:
{{
    "experimental_design": {{
        "setup": {{
            "biological_conditions": ["string"],
            "experimental_protocol": "string",
            "controls": ["string"],
            "replicates": "string",
            "biological_validation": "string"
        }},
        "perturbation": {{
            "design": "string",
            "types": ["string"],
            "doses": ["string"],
            "timepoints": ["string"],
            "biological_controls": ["string"]
        }},
        "samples": {{
            "size": "string",
            "distribution": "string",
            "characteristics": ["string"],
            "biological_variability": "string",
            "technical_variability": "string"
        }},
        "validation": {{
            "biological": {{
                "methods": ["string"],
                "controls": ["string"],
                "metrics": ["string"],
                "results": "string",
                "interpretation": "string"
            }},
            "technical": {{
                "methods": ["string"],
                "controls": ["string"],
                "metrics": ["string"],
                "results": "string",
                "interpretation": "string"
            }}
        }}
    }},
    "data_properties": {{
        "distribution": {{
            "biological": {{
                "patterns": ["string"],
                "variability": "string",
                "outliers": ["string"],
                "interpretation": "string",
                "biological_significance": "string"
            }},
            "technical": {{
                "patterns": ["string"],
                "variability": "string",
                "artifacts": ["string"],
                "interpretation": "string",
                "mitigation": ["string"]
            }}
        }},
        "features": {{
            "biological": {{
                "types": ["string"],
                "relevance": "string",
                "selection": ["string"],
                "interpretation": "string",
                "validation": "string"
            }},
            "technical": {{
                "types": ["string"],
                "quality": "string",
                "selection": ["string"],
                "interpretation": "string",
                "validation": "string"
            }}
        }},
        "quality": {{
            "biological": {{
                "metrics": ["string"],
                "assessment": "string",
                "issues": ["string"],
                "mitigation": ["string"],
                "validation": "string"
            }},
            "technical": {{
                "metrics": ["string"],
                "assessment": "string",
                "issues": ["string"],
                "mitigation": ["string"],
                "validation": "string"
            }}
        }},
        "noise": {{
            "biological": {{
                "sources": ["string"],
                "impact": "string",
                "assessment": "string",
                "mitigation": ["string"],
                "validation": "string"
            }},
            "technical": {{
                "sources": ["string"],
                "impact": "string",
                "assessment": "string",
                "mitigation": ["string"],
                "validation": "string"
            }}
        }}
    }},
    "preprocessing": {{
        "normalization": {{
            "strategies": {{
                "biological": ["string"],
                "technical": ["string"],
                "rationale": "string",
                "implementation": "string",
                "validation": "string"
            }},
            "parameters": {{
                "biological": ["string"],
                "technical": ["string"],
                "optimization": "string",
                "validation": "string",
                "biological_validation": "string"
            }}
        }},
        "quality_control": {{
            "procedures": {{
                "biological": ["string"],
                "technical": ["string"],
                "rationale": "string",
                "implementation": "string",
                "validation": "string"
            }},
            "thresholds": {{
                "biological": ["string"],
                "technical": ["string"],
                "rationale": "string",
                "implementation": "string",
                "validation": "string"
            }}
        }},
        "feature_selection": {{
            "methods": {{
                "biological": ["string"],
                "technical": ["string"],
                "rationale": "string",
                "implementation": "string",
                "validation": "string"
            }},
            "criteria": {{
                "biological": ["string"],
                "technical": ["string"],
                "rationale": "string",
                "implementation": "string",
                "validation": "string"
            }}
        }},
        "batch_correction": {{
            "methods": {{
                "biological": ["string"],
                "technical": ["string"],
                "rationale": "string",
                "implementation": "string",
                "validation": "string"
            }},
            "parameters": {{
                "biological": ["string"],
                "technical": ["string"],
                "optimization": "string",
                "validation": "string",
                "biological_validation": "string"
            }}
        }}
    }},
    "quality_assessment": {{
        "completeness": {{
            "biological": {{
                "coverage": "string",
                "gaps": ["string"],
                "impact": "string",
                "mitigation": ["string"],
                "validation": "string"
            }},
            "technical": {{
                "coverage": "string",
                "gaps": ["string"],
                "impact": "string",
                "mitigation": ["string"],
                "validation": "string"
            }}
        }},
        "relevance": {{
            "biological": {{
                "significance": "string",
                "interpretability": "string",
                "validation": "string",
                "limitations": ["string"],
                "improvements": ["string"]
            }},
            "technical": {{
                "significance": "string",
                "interpretability": "string",
                "validation": "string",
                "limitations": ["string"],
                "improvements": ["string"]
            }}
        }},
        "reproducibility": {{
            "biological": {{
                "assessment": "string",
                "metrics": ["string"],
                "validation": "string",
                "limitations": ["string"],
                "improvements": ["string"]
            }},
            "technical": {{
                "assessment": "string",
                "metrics": ["string"],
                "validation": "string",
                "limitations": ["string"],
                "improvements": ["string"]
            }}
        }},
        "limitations": {{
            "biological": {{
                "issues": ["string"],
                "impact": "string",
                "mitigation": ["string"],
                "validation": "string",
                "improvements": ["string"]
            }},
            "technical": {{
                "issues": ["string"],
                "impact": "string",
                "mitigation": ["string"],
                "validation": "string",
                "improvements": ["string"]
            }}
        }}
    }}
}}"""

    def _format_prompt_with_decision_support(self, task_description: str, dataset_info: Dict[str, Any],
                                           papers: List[Dict[str, Any]], decision_support: Dict[str, Any],
                                           experimental_designs: List[Dict[str, Any]]) -> str:
        """Format prompt for dataset analysis with decision support"""
        papers_context = "\n".join([
            f"- {paper.get('title', 'No title')}: {paper.get('abstract', paper.get('content', paper.get('snippet', 'No content')))[:200]}..."
            for paper in papers[:5]
        ])

        # 格式化决策支持信息
        decision_context = ""
        if decision_support:
            decision_context = f"""
DECISION SUPPORT INFORMATION:
Data Preparation: {json.dumps(decision_support.get('data_preparation', {}), indent=2)}
Implementation Plan: {json.dumps(decision_support.get('implementation_plan', {}), indent=2)}
Risk Assessment: {json.dumps(decision_support.get('risk_assessment', {}), indent=2)}
"""

        # 格式化实验设计信息
        design_context = ""
        if experimental_designs:
            design_context = f"""
EXPERIMENTAL DESIGNS:
{chr(10).join([f"- {design['title']}: {design['content'][:200]}..." for design in experimental_designs[:3]])}
"""

        return f"""You are an expert in single-cell dataset analysis, specializing in evaluating single-cell perturbation data. Your task is to provide a comprehensive analysis of dataset characteristics and quality, focusing on experimental design, data properties, preprocessing recommendations, and quality assessment.

{decision_context}

{design_context}

1. Analyze Experimental Design:
   - Evaluate experimental setup and biological conditions
   - Assess sample size and biological replicates
   - Review perturbation design and controls
   - Analyze biological variability and batch effects
   - Consider technical and biological validation

2. Assess Data Properties:
   - Analyze data distribution and biological patterns
   - Evaluate feature space and biological relevance
   - Assess data quality and technical artifacts
   - Identify biological and technical noise
   - Consider data sparsity and missing values

3. Provide Preprocessing Recommendations:
   - Suggest normalization strategies for biological data
   - Recommend quality control procedures
   - Propose feature selection methods
   - Design batch effect correction approaches
   - Plan biological validation steps

4. Conduct Quality Assessment:
   - Evaluate data completeness and coverage
   - Assess biological relevance and interpretability
   - Analyze technical quality and reproducibility
   - Consider biological validation requirements
   - Identify potential limitations and biases

Task Description:
{task_description}

Dataset Information:
{json.dumps(dataset_info, indent=2)}

Relevant Literature:
{papers_context}

Please provide a comprehensive analysis in the following JSON format, incorporating decision support information:
{{
    "experimental_design": {{
        "setup": {{
            "biological_conditions": ["string"],
            "experimental_protocol": "string",
            "controls": ["string"],
            "replicates": "string",
            "biological_validation": "string",
            "decision_support_alignment": "string"
        }},
        "perturbation": {{
            "design": "string",
            "types": ["string"],
            "doses": ["string"],
            "timepoints": ["string"],
            "biological_controls": ["string"],
            "decision_support_considerations": "string"
        }},
        "samples": {{
            "size": "string",
            "distribution": "string",
            "characteristics": ["string"],
            "biological_variability": "string",
            "technical_variability": "string",
            "decision_support_requirements": "string"
        }},
        "validation": {{
            "biological": {{
                "methods": ["string"],
                "controls": ["string"],
                "metrics": ["string"],
                "results": "string",
                "interpretation": "string",
                "decision_support_validation": "string"
            }},
            "technical": {{
                "methods": ["string"],
                "controls": ["string"],
                "metrics": ["string"],
                "results": "string",
                "interpretation": "string",
                "decision_support_validation": "string"
            }}
        }}
    }},
    "data_properties": {{
        "distribution": {{
            "biological": {{
                "patterns": ["string"],
                "variability": "string",
                "outliers": ["string"],
                "interpretation": "string",
                "biological_significance": "string",
                "decision_support_implications": "string"
            }},
            "technical": {{
                "patterns": ["string"],
                "variability": "string",
                "artifacts": ["string"],
                "interpretation": "string",
                "mitigation": ["string"],
                "decision_support_considerations": "string"
            }}
        }},
        "features": {{
            "biological": {{
                "types": ["string"],
                "relevance": "string",
                "selection": ["string"],
                "interpretation": "string",
                "validation": "string",
                "decision_support_features": "string"
            }},
            "technical": {{
                "types": ["string"],
                "quality": "string",
                "selection": ["string"],
                "interpretation": "string",
                "validation": "string",
                "decision_support_technical": "string"
            }}
        }},
        "quality": {{
            "biological": {{
                "metrics": ["string"],
                "assessment": "string",
                "issues": ["string"],
                "mitigation": ["string"],
                "validation": "string",
                "decision_support_quality": "string"
            }},
            "technical": {{
                "metrics": ["string"],
                "assessment": "string",
                "issues": ["string"],
                "mitigation": ["string"],
                "validation": "string",
                "decision_support_technical_quality": "string"
            }}
        }},
        "noise": {{
            "biological": {{
                "sources": ["string"],
                "impact": "string",
                "assessment": "string",
                "mitigation": ["string"],
                "validation": "string",
                "decision_support_noise": "string"
            }},
            "technical": {{
                "sources": ["string"],
                "impact": "string",
                "assessment": "string",
                "mitigation": ["string"],
                "validation": "string",
                "decision_support_technical_noise": "string"
            }}
        }}
    }},
    "preprocessing": {{
        "normalization": {{
            "strategies": {{
                "recommendations": ["string"],
                "rationale": "string",
                "implementation": "string",
                "validation": "string",
                "biological_validation": "string",
                "decision_support_normalization": "string"
            }},
            "quality_control": {{
                "procedures": ["string"],
                "criteria": ["string"],
                "implementation": "string",
                "validation": "string",
                "biological_validation": "string",
                "decision_support_qc": "string"
            }},
            "feature_selection": {{
                "methods": ["string"],
                "criteria": ["string"],
                "implementation": "string",
                "validation": "string",
                "biological_validation": "string",
                "decision_support_features": "string"
            }},
            "batch_correction": {{
                "methods": ["string"],
                "criteria": ["string"],
                "implementation": "string",
                "validation": "string",
                "biological_validation": "string",
                "decision_support_batch": "string"
            }}
        }},
        "validation": {{
            "biological": {{
                "methods": ["string"],
                "criteria": ["string"],
                "implementation": "string",
                "validation": "string",
                "biological_validation": "string",
                "decision_support_validation": "string"
            }},
            "technical": {{
                "methods": ["string"],
                "criteria": ["string"],
                "implementation": "string",
                "validation": "string",
                "technical_validation": "string",
                "decision_support_technical_validation": "string"
            }}
        }}
    }},
    "quality_assessment": {{
        "completeness": {{
            "coverage": "string",
            "missing_data": "string",
            "assessment": "string",
            "mitigation": ["string"],
            "validation": "string",
            "decision_support_completeness": "string"
        }},
        "relevance": {{
            "biological": "string",
            "technical": "string",
            "assessment": "string",
            "validation": "string",
            "biological_validation": "string",
            "decision_support_relevance": "string"
        }},
        "reproducibility": {{
            "technical": "string",
            "biological": "string",
            "assessment": "string",
            "validation": "string",
            "biological_validation": "string",
            "decision_support_reproducibility": "string"
        }},
        "limitations": {{
            "biological": ["string"],
            "technical": ["string"],
            "assessment": "string",
            "mitigation": ["string"],
            "validation": "string",
            "decision_support_limitations": "string"
        }}
    }}
}}"""

    def _run_llm(self, prompt: str) -> Dict[str, Any]:
        """Run LLM with retry mechanism for network errors"""
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                print(f"🔄 LLM attempt {attempt + 1}/{max_retries}")

                system_prompt = "You are an expert in single-cell dataset analysis. Provide your response in valid JSON format."

                # 使用统一的LLMInterface
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
                            "dataset_characteristics": {
                                "data_quality": "Fallback: High quality data",
                                "cell_count": "10,000 cells",
                                "gene_count": "2,000 genes"
                            },
                            "quality_assessment": {
                                "quality_score": 0.8,
                                "note": "Fallback assessment due to LLM connection issues"
                            },
                            "preprocessing_recommendations": [
                                "Standard quality control",
                                "Normalization",
                                "Feature selection"
                            ],
                            "error": "LLM connection failed, using fallback"
                        }
                else:
                    # 其他错误直接抛出
                    raise Exception(f"LLM generation failed: {error_msg}")

        # 如果所有重试都失败
        return {"content": "LLM generation failed after all retries", "error": "Connection issues"}
