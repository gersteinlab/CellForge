from typing import Dict, Any, List
from datetime import datetime
import json

from .data_structures import AnalysisResult
from .knowledge_base import knowledge_base
from ..llm import LLMInterface

class ProblemInvestigator:
    """Expert agent for investigating research problems and defining analytical approaches"""

    def __init__(self, rag_system):
        self.rag_system = rag_system

        self.llm = LLMInterface()

    def _run_llm(self, prompt: str) -> Dict[str, Any]:
        """Run LLM with retry mechanism for network errors"""
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                print(f"🔄 LLM attempt {attempt + 1}/{max_retries}")

                system_prompt = "You are an expert in single-cell perturbation research. Provide your response in valid JSON format."

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
                            "research_questions": [
                                "How do perturbations affect gene expression?",
                                "What are the key regulatory mechanisms?"
                            ],
                            "analytical_approaches": [
                                "Differential expression analysis",
                                "Pathway enrichment analysis"
                            ],
                            "error": "LLM connection failed, using fallback"
                        }
                else:
                    # 其他错误直接抛出
                    raise Exception(f"LLM generation failed: {error_msg}")

        # 如果所有重试都失败
        return {"content": "LLM generation failed after all retries", "error": "Connection issues"}

    def investigate_problem(self, task_description: str, dataset_info: Dict[str, Any],
                           retrieved_papers: List[Dict[str, Any]]) -> AnalysisResult:
        """
        Investigate research problem and define analytical approaches using knowledge base

        Args:
            task_description: Description of the research task
            dataset_info: Dictionary containing dataset metadata
            retrieved_papers: List of relevant papers from vector database

        Returns:
            AnalysisResult with problem investigation
        """
        # 使用knowledge base而不是重复搜索
        knowledge_items = knowledge_base.search_both_databases(
            knowledge_type="papers",
            query=task_description,
            limit=10
        )

        # 获取实现指南信息
        implementation_items = knowledge_base.search_both_databases(
            knowledge_type="implementation_guides",
            query=task_description,
            limit=5
        )

        # 获取评估框架信息
        evaluation_items = knowledge_base.search_both_databases(
            knowledge_type="evaluation_frameworks",
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

        # 合并实现指南信息
        implementation_guides = [item.content for item in implementation_items]

        # 合并评估框架信息
        evaluation_frameworks = [item.content for item in evaluation_items]

        # Format prompt with task information and retrieved papers
        prompt = self._format_prompt_with_implementation_guides(
            task_description, dataset_info, all_papers, implementation_guides, evaluation_frameworks
        )

        # Run investigation (implementation depends on your LLM backend)
        investigation_content = self._run_llm(prompt)

        return AnalysisResult(
            content=investigation_content,
            confidence_score=1.0,  # 临时设置，后续会由其他模块计算
            timestamp=datetime.now(),
            metadata={
                "agent": "Problem Investigator",
                "knowledge_base_usage": {
                    "papers_count": len(knowledge_items),
                    "implementation_guides_count": len(implementation_items),
                    "evaluation_frameworks_count": len(evaluation_items),
                    "total_retrieved": len(all_papers)
                }
            }
        )

    def _format_prompt(self, task_description: str, dataset_info: Dict[str, Any],
                      retrieved_papers: List[Dict[str, Any]]) -> str:
        """Format prompt for problem investigation"""
        papers_context = "\n".join([
            f"- {paper.get('title', 'No title')}: {paper.get('abstract', paper.get('content', paper.get('snippet', 'No content')))[:200]}..."
            for paper in retrieved_papers[:5]
        ])

        return f"""You are an expert in biological research problem analysis with extensive experience in designing computational solutions for biological applications. Your task is to provide a comprehensive investigation of the research problem and design a solution approach, focusing on problem definition, key challenges, research questions, and analysis methods.

1. Define Research Problem:
   - Formally define the problem in biological context
   - Identify key biological variables and their relationships
   - Specify input-output mappings and biological constraints
   - Define evaluation criteria with biological significance
   - Establish success metrics with biological validation

2. Analyze Key Challenges:
   - Identify biological and technical challenges
   - Assess data quality and biological variability
   - Evaluate computational complexity and scalability
   - Consider biological interpretability requirements
   - Address validation and reproducibility concerns

3. Formulate Research Questions:
   - Define primary research questions with biological focus
   - Identify key hypotheses to be tested
   - Specify biological mechanisms to be investigated
   - Outline experimental validation requirements
   - Establish biological significance criteria

4. Design Analysis Methods:
   - Propose computational approaches with biological relevance
   - Design validation strategies with biological context
   - Specify analysis pipelines with biological interpretation
   - Plan experimental validation with biological controls
   - Establish reproducibility standards with biological validation

Task Description:
{task_description}

Dataset Information:
{json.dumps(dataset_info, indent=2)}

Relevant Literature:
{papers_context}

Please provide a comprehensive investigation in the following JSON format:
{{
    "problem_definition": {{
        "formal_definition": {{
            "biological_context": "string",
            "input_output_mapping": "string",
            "biological_constraints": ["string"],
            "evaluation_criteria": ["string"],
            "success_metrics": ["string"]
        }},
        "key_variables": {{
            "biological": ["string"],
            "technical": ["string"],
            "relationships": ["string"],
            "constraints": ["string"],
            "validation_requirements": ["string"]
        }},
        "scope": {{
            "biological_scope": "string",
            "technical_scope": "string",
            "limitations": ["string"],
            "assumptions": ["string"],
            "biological_validation": "string"
        }}
    }},
    "key_challenges": {{
        "biological": {{
            "challenges": ["string"],
            "impact": "string",
            "mitigation": ["string"],
            "validation": "string",
            "biological_considerations": "string"
        }},
        "technical": {{
            "challenges": ["string"],
            "impact": "string",
            "mitigation": ["string"],
            "validation": "string",
            "implementation_requirements": "string"
        }},
        "data_quality": {{
            "issues": ["string"],
            "impact": "string",
            "mitigation": ["string"],
            "validation": "string",
            "biological_validation": "string"
        }},
        "computational": {{
            "challenges": ["string"],
            "impact": "string",
            "mitigation": ["string"],
            "validation": "string",
            "resource_requirements": "string"
        }},
        "interpretability": {{
            "requirements": ["string"],
            "challenges": ["string"],
            "solutions": ["string"],
            "validation": "string",
            "biological_validation": "string"
        }}
    }},
    "research_questions": {{
        "primary": {{
            "questions": ["string"],
            "hypotheses": ["string"],
            "biological_significance": "string",
            "validation_approach": "string",
            "expected_outcomes": ["string"]
        }},
        "secondary": {{
            "questions": ["string"],
            "hypotheses": ["string"],
            "biological_significance": "string",
            "validation_approach": "string",
            "expected_outcomes": ["string"]
        }},
        "biological_mechanisms": {{
            "mechanisms": ["string"],
            "investigation_approach": "string",
            "validation_methods": ["string"],
            "expected_insights": ["string"],
            "biological_validation": "string"
        }},
        "experimental_validation": {{
            "requirements": ["string"],
            "methods": ["string"],
            "controls": ["string"],
            "metrics": ["string"],
            "biological_validation": "string"
        }}
    }},
    "analysis_methods": {{
        "computational_approaches": {{
            "methods": ["string"],
            "rationale": "string",
            "implementation": "string",
            "validation": "string",
            "biological_validation": "string"
        }},
        "validation_strategies": {{
            "strategies": ["string"],
            "rationale": "string",
            "implementation": "string",
            "metrics": ["string"],
            "biological_validation": "string"
        }},
        "analysis_pipelines": {{
            "pipelines": ["string"],
            "components": ["string"],
            "workflow": "string",
            "validation": "string",
            "biological_interpretation": "string"
        }},
        "experimental_validation": {{
            "methods": ["string"],
            "controls": ["string"],
            "metrics": ["string"],
            "analysis": "string",
            "biological_validation": "string"
        }},
        "reproducibility": {{
            "standards": ["string"],
            "requirements": ["string"],
            "validation": "string",
            "documentation": "string",
            "biological_validation": "string"
        }}
    }}
}}"""

    def _format_prompt_with_implementation_guides(self, task_description: str, dataset_info: Dict[str, Any],
                                            papers: List[Dict[str, Any]], implementation_guides: List[Dict[str, Any]],
                                            evaluation_frameworks: List[Dict[str, Any]]) -> str:
        """Format prompt for problem investigation with decision support"""
        papers_context = "\n".join([
            f"- {paper.get('title', 'No title')}: {paper.get('abstract', paper.get('content', paper.get('snippet', 'No content')))[:200]}..."
            for paper in papers[:5]
        ])

        return f"""You are an expert in biological research problem analysis with extensive experience in designing computational solutions for biological applications. Your task is to provide a comprehensive investigation of the research problem and design a solution approach, focusing on problem definition, key challenges, research questions, and analysis methods.

1. Define Research Problem:
   - Formally define the problem in biological context
   - Identify key biological variables and their relationships
   - Specify input-output mappings and biological constraints
   - Define evaluation criteria with biological significance
   - Establish success metrics with biological validation

2. Analyze Key Challenges:
   - Identify biological and technical challenges
   - Assess data quality and biological variability
   - Evaluate computational complexity and scalability
   - Consider biological interpretability requirements
   - Address validation and reproducibility concerns

3. Formulate Research Questions:
   - Define primary research questions with biological focus
   - Identify key hypotheses to be tested
   - Specify biological mechanisms to be investigated
   - Outline experimental validation requirements
   - Establish biological significance criteria

4. Design Analysis Methods:
   - Propose computational approaches with biological relevance
   - Design validation strategies with biological context
   - Specify analysis pipelines with biological interpretation
   - Plan experimental validation with biological controls
   - Establish reproducibility standards with biological validation

Task Description:
{task_description}

Dataset Information:
{json.dumps(dataset_info, indent=2)}

Relevant Literature:
{papers_context}

Implementation Guides:
{json.dumps(implementation_guides, indent=2)}

Evaluation Frameworks:
{json.dumps(evaluation_frameworks, indent=2)}

Please provide a comprehensive investigation in the following JSON format:
{{
    "problem_definition": {{
        "formal_definition": {{
            "biological_context": "string",
            "input_output_mapping": "string",
            "biological_constraints": ["string"],
            "evaluation_criteria": ["string"],
            "success_metrics": ["string"]
        }},
        "key_variables": {{
            "biological": ["string"],
            "technical": ["string"],
            "relationships": ["string"],
            "constraints": ["string"],
            "validation_requirements": ["string"]
        }},
        "scope": {{
            "biological_scope": "string",
            "technical_scope": "string",
            "limitations": ["string"],
            "assumptions": ["string"],
            "biological_validation": "string"
        }}
    }},
    "key_challenges": {{
        "biological": {{
            "challenges": ["string"],
            "impact": "string",
            "mitigation": ["string"],
            "validation": "string",
            "biological_considerations": "string"
        }},
        "technical": {{
            "challenges": ["string"],
            "impact": "string",
            "mitigation": ["string"],
            "validation": "string",
            "implementation_requirements": "string"
        }},
        "data_quality": {{
            "issues": ["string"],
            "impact": "string",
            "mitigation": ["string"],
            "validation": "string",
            "biological_validation": "string"
        }},
        "computational": {{
            "challenges": ["string"],
            "impact": "string",
            "mitigation": ["string"],
            "validation": "string",
            "resource_requirements": "string"
        }},
        "interpretability": {{
            "requirements": ["string"],
            "challenges": ["string"],
            "solutions": ["string"],
            "validation": "string",
            "biological_validation": "string"
        }}
    }},
    "research_questions": {{
        "primary": {{
            "questions": ["string"],
            "hypotheses": ["string"],
            "biological_significance": "string",
            "validation_approach": "string",
            "expected_outcomes": ["string"]
        }},
        "secondary": {{
            "questions": ["string"],
            "hypotheses": ["string"],
            "biological_significance": "string",
            "validation_approach": "string",
            "expected_outcomes": ["string"]
        }},
        "biological_mechanisms": {{
            "mechanisms": ["string"],
            "investigation_approach": "string",
            "validation_methods": ["string"],
            "expected_insights": ["string"],
            "biological_validation": "string"
        }},
        "experimental_validation": {{
            "requirements": ["string"],
            "methods": ["string"],
            "controls": ["string"],
            "metrics": ["string"],
            "biological_validation": "string"
        }}
    }},
    "analysis_methods": {{
        "computational_approaches": {{
            "methods": ["string"],
            "rationale": "string",
            "implementation": "string",
            "validation": "string",
            "biological_validation": "string"
        }},
        "validation_strategies": {{
            "strategies": ["string"],
            "rationale": "string",
            "implementation": "string",
            "metrics": ["string"],
            "biological_validation": "string"
        }},
        "analysis_pipelines": {{
            "pipelines": ["string"],
            "components": ["string"],
            "workflow": "string",
            "validation": "string",
            "biological_interpretation": "string"
        }},
        "experimental_validation": {{
            "methods": ["string"],
            "controls": ["string"],
            "metrics": ["string"],
            "analysis": "string",
            "biological_validation": "string"
        }},
        "reproducibility": {{
            "standards": ["string"],
            "requirements": ["string"],
            "validation": "string",
            "documentation": "string",
            "biological_validation": "string"
        }}
    }}
}}"""
