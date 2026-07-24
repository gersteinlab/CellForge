from typing import Dict, Any, List
from datetime import datetime
import os
import json

from .data_structures import AnalysisResult, TaskAnalysisReport
from ..llm import LLMInterface

class RefinementAgent:
    """Expert agent for refining and integrating analysis results"""

    def __init__(self, rag_system):
        self.rag_system = rag_system

        self.llm = LLMInterface()

        self.max_iterations = max(
            0, int(os.getenv("TASK_ANALYSIS_MAX_REFINEMENT_ROUNDS", "3"))
        )

    def refine_analysis(self, dataset_analysis: AnalysisResult,
                       problem_investigation: AnalysisResult,
                       baseline_assessment: AnalysisResult) -> TaskAnalysisReport:
        """
        Refine and integrate analysis results with decision support

        Args:
            dataset_analysis: Analysis of dataset characteristics
            problem_investigation: Investigation of research problem
            baseline_assessment: Assessment of baseline models

        Returns:
            TaskAnalysisReport with refined analysis
        """

        task_description = self._extract_task_description(dataset_analysis, problem_investigation, baseline_assessment)
        dataset_info = self._extract_dataset_info(dataset_analysis)


        decision_support = self.rag_system.get_decision_support(task_description, dataset_info)


        current_iteration = 0
        refined_dataset = dataset_analysis
        refined_problem = problem_investigation
        refined_baseline = baseline_assessment

        while current_iteration < self.max_iterations:

            comments = self._generate_refinement_comments_with_decision_support(
                refined_dataset, refined_problem, refined_baseline, decision_support
            )

            # Open-ended model providers can return valid prose, truncated JSON,
            # or a JSON object that omits one of the requested sections. Keep the
            # pipeline useful in those cases instead of failing on comments[key].
            if not isinstance(comments, dict):
                comments = {"content": str(comments)}
            fallback_comments = {
                "note": comments.get("content", "No section-specific comments returned"),
                "parse_error": comments.get("error"),
            }

            refined_dataset = self._refine_dataset_analysis(
                refined_dataset, comments.get("dataset", fallback_comments)
            )
            refined_problem = self._refine_problem_investigation(
                refined_problem, comments.get("problem", fallback_comments)
            )
            refined_baseline = self._refine_baseline_assessment(
                refined_baseline, comments.get("baseline", fallback_comments)
            )

            current_iteration += 1


        recommendations = self._generate_final_recommendations_with_decision_support(
            refined_dataset, refined_problem, refined_baseline, decision_support
        )

        return TaskAnalysisReport(
            dataset_analysis=refined_dataset,
            problem_investigation=refined_problem,
            baseline_assessment=refined_baseline,
            refinement_comments=[],
            final_recommendations=recommendations,
            timestamp=datetime.now()
        )

    def _generate_refinement_comments(self, dataset_analysis: AnalysisResult,
                                    problem_investigation: AnalysisResult,
                                    baseline_assessment: AnalysisResult) -> Dict[str, Any]:
        """Generate refinement comments for each analysis"""
        prompt = self._format_refinement_prompt(
            dataset_analysis, problem_investigation, baseline_assessment
        )


        comments = self._run_llm(prompt)

        return comments

    def _format_refinement_prompt(self, dataset_analysis: AnalysisResult,
                                problem_investigation: AnalysisResult,
                                baseline_assessment: AnalysisResult) -> str:
        """Format prompt for generating refinement comments"""
        return f"""You are an expert in scientific analysis integration with extensive experience in refining and improving research analyses. Your task is to provide comprehensive refinement comments for dataset analysis, problem investigation, and baseline assessment, focusing on integration, consistency, and improvement.

1. Review Dataset Analysis:
   - Evaluate biological relevance and interpretability
   - Assess technical quality and reproducibility
   - Check preprocessing recommendations
   - Validate quality assessment
   - Suggest improvements

2. Review Problem Investigation:
   - Evaluate problem definition and scope
   - Assess research questions and hypotheses
   - Check analysis methods and validation
   - Validate biological mechanisms
   - Suggest improvements

3. Review Baseline Assessment:
   - Evaluate model analysis and comparison
   - Assess evaluation framework
   - Check performance analysis
   - Validate improvement suggestions
   - Suggest improvements

4. Ensure Integration:
   - Check consistency across analyses
   - Validate biological relevance
   - Assess technical feasibility
   - Evaluate practical implementation
   - Suggest improvements

Dataset Analysis:
{json.dumps(dataset_analysis.content, indent=2)}

Problem Investigation:
{json.dumps(problem_investigation.content, indent=2)}

Baseline Assessment:
{json.dumps(baseline_assessment.content, indent=2)}

Please provide refinement comments in the following JSON format:
{{
    "dataset": {{
        "biological": {{
            "relevance": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string"
            }},
            "interpretability": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string"
            }}
        }},
        "technical": {{
            "quality": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "technical_validation": "string"
            }},
            "reproducibility": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "technical_validation": "string"
            }}
        }},
        "preprocessing": {{
            "recommendations": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "implementation": "string"
            }},
            "validation": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "implementation": "string"
            }}
        }},
        "quality": {{
            "assessment": {{
                "evaluation": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "implementation": "string"
            }},
            "validation": {{
                "evaluation": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "implementation": "string"
            }}
        }}
    }},
    "problem": {{
        "definition": {{
            "scope": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string"
            }},
            "clarity": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string"
            }}
        }},
        "research": {{
            "questions": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string"
            }},
            "hypotheses": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string"
            }}
        }},
        "methods": {{
            "analysis": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string"
            }},
            "validation": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string"
            }}
        }},
        "mechanisms": {{
            "biological": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string"
            }},
            "validation": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string"
            }}
        }}
    }},
    "baseline": {{
        "models": {{
            "analysis": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string"
            }},
            "comparison": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string"
            }}
        }},
        "evaluation": {{
            "framework": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string"
            }},
            "metrics": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string"
            }}
        }},
        "performance": {{
            "analysis": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string"
            }},
            "validation": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string"
            }}
        }},
        "improvements": {{
            "suggestions": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string"
            }},
            "validation": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string"
            }}
        }}
    }},
    "integration": {{
        "consistency": {{
            "assessment": "string",
            "issues": ["string"],
            "improvements": ["string"],
            "validation": "string",
            "biological_validation": "string"
        }},
        "relevance": {{
            "assessment": "string",
            "issues": ["string"],
            "improvements": ["string"],
            "validation": "string",
            "biological_validation": "string"
        }},
        "feasibility": {{
            "assessment": "string",
            "issues": ["string"],
            "improvements": ["string"],
            "validation": "string",
            "technical_validation": "string"
        }},
        "implementation": {{
            "assessment": "string",
            "issues": ["string"],
            "improvements": ["string"],
            "validation": "string",
            "technical_validation": "string"
        }}
    }}
}}"""

    def _refine_dataset_analysis(self, analysis: AnalysisResult,
                               comments: Dict[str, Any]) -> AnalysisResult:
        """Refine dataset analysis based on comments"""
        prompt = self._format_dataset_refinement_prompt(analysis.content, comments)


        refined_content = self._run_llm(prompt)

        return AnalysisResult(
            content=refined_content,
            confidence_score=1.0,
            timestamp=datetime.now(),
            metadata={"agent": "Dataset Analyst"}
        )

    def _refine_problem_investigation(self, investigation: AnalysisResult,
                                    comments: Dict[str, Any]) -> AnalysisResult:
        """Refine problem investigation based on comments"""
        prompt = self._format_problem_refinement_prompt(investigation.content, comments)


        refined_content = self._run_llm(prompt)

        return AnalysisResult(
            content=refined_content,
            confidence_score=1.0,
            timestamp=datetime.now(),
            metadata={"agent": "Problem Investigator"}
        )

    def _refine_baseline_assessment(self, assessment: AnalysisResult,
                                  comments: Dict[str, Any]) -> AnalysisResult:
        """Refine baseline assessment based on comments"""
        prompt = self._format_baseline_refinement_prompt(assessment.content, comments)


        refined_content = self._run_llm(prompt)

        return AnalysisResult(
            content=refined_content,
            confidence_score=1.0,
            timestamp=datetime.now(),
            metadata={"agent": "Baseline Assessor"}
        )

    def _format_dataset_refinement_prompt(self, analysis_content: Dict[str, Any],
                                        comments: Dict[str, Any]) -> str:
        """Build a refinement prompt for dataset analysis."""
        return self._format_single_refinement_prompt(
            section_name="Dataset Analysis",
            analysis_content=analysis_content,
            comments=comments
        )

    def _format_problem_refinement_prompt(self, investigation_content: Dict[str, Any],
                                        comments: Dict[str, Any]) -> str:
        """Build a refinement prompt for problem investigation."""
        return self._format_single_refinement_prompt(
            section_name="Problem Investigation",
            analysis_content=investigation_content,
            comments=comments
        )

    def _format_baseline_refinement_prompt(self, assessment_content: Dict[str, Any],
                                         comments: Dict[str, Any]) -> str:
        """Build a refinement prompt for baseline assessment."""
        return self._format_single_refinement_prompt(
            section_name="Baseline Assessment",
            analysis_content=assessment_content,
            comments=comments
        )

    def _format_single_refinement_prompt(self, section_name: str,
                                       analysis_content: Dict[str, Any],
                                       comments: Dict[str, Any]) -> str:
        """Format a section-specific refinement prompt and request JSON output."""
        return f"""You are refining the {section_name} for a scientific task analysis.
Apply the reviewer comments to improve accuracy, completeness, and implementation usefulness.

SECTION TO REFINE:
{json.dumps(analysis_content, indent=2)}

REVIEWER COMMENTS:
{json.dumps(comments, indent=2)}

Requirements:
1. Keep the same high-level structure when possible.
2. Resolve issues called out in the comments.
3. Add explicit assumptions where information is missing.
4. Return only valid JSON (no markdown).
"""

    def _generate_final_recommendations(self, dataset_analysis: AnalysisResult,
                                      problem_investigation: AnalysisResult,
                                      baseline_assessment: AnalysisResult) -> Dict[str, Any]:
        """Generate final recommendations based on refined analyses"""
        prompt = self._format_recommendations_prompt(
            dataset_analysis, problem_investigation, baseline_assessment
        )


        recommendations = self._run_llm(prompt)

        return recommendations

    def _format_recommendations_prompt(self, dataset_analysis: AnalysisResult,
                                     problem_investigation: AnalysisResult,
                                     baseline_assessment: AnalysisResult) -> str:
        """Format prompt for generating final recommendations"""
        return f"""You are an expert in scientific analysis integration with extensive experience in providing comprehensive research recommendations. Your task is to provide final recommendations based on the refined analyses, focusing on biological relevance, technical feasibility, and practical implementation.

1. Biological Recommendations:
   - Evaluate biological significance
   - Assess experimental validation
   - Consider biological mechanisms
   - Plan biological validation
   - Suggest improvements

2. Technical Recommendations:
   - Evaluate technical feasibility
   - Assess implementation requirements
   - Consider computational resources
   - Plan technical validation
   - Suggest improvements

3. Practical Recommendations:
   - Evaluate practical implementation
   - Assess resource requirements
   - Consider deployment strategy
   - Plan validation and monitoring
   - Suggest improvements

4. Future Work:
   - Identify research directions
   - Plan validation experiments
   - Consider collaborations
   - Assess funding needs
   - Suggest improvements

Dataset Analysis:
{json.dumps(dataset_analysis.content, indent=2)}

Problem Investigation:
{json.dumps(problem_investigation.content, indent=2)}

Baseline Assessment:
{json.dumps(baseline_assessment.content, indent=2)}

Please provide final recommendations in the following JSON format:
{{
    "biological": {{
        "significance": {{
            "assessment": "string",
            "recommendations": ["string"],
            "validation": "string",
            "improvements": ["string"],
            "biological_validation": "string"
        }},
        "experimental": {{
            "validation": {{
                "methods": ["string"],
                "controls": ["string"],
                "metrics": ["string"],
                "analysis": "string",
                "biological_validation": "string"
            }},
            "improvements": {{
                "suggestions": ["string"],
                "implementation": "string",
                "validation": "string",
                "biological_validation": "string",
                "timeline": "string"
            }}
        }},
        "mechanisms": {{
            "investigation": {{
                "methods": ["string"],
                "controls": ["string"],
                "metrics": ["string"],
                "analysis": "string",
                "biological_validation": "string"
            }},
            "improvements": {{
                "suggestions": ["string"],
                "implementation": "string",
                "validation": "string",
                "biological_validation": "string",
                "timeline": "string"
            }}
        }},
        "validation": {{
            "plan": {{
                "methods": ["string"],
                "controls": ["string"],
                "metrics": ["string"],
                "analysis": "string",
                "biological_validation": "string"
            }},
            "improvements": {{
                "suggestions": ["string"],
                "implementation": "string",
                "validation": "string",
                "biological_validation": "string",
                "timeline": "string"
            }}
        }}
    }},
    "technical": {{
        "feasibility": {{
            "assessment": "string",
            "requirements": ["string"],
            "limitations": ["string"],
            "mitigation": ["string"],
            "validation": "string"
        }},
        "implementation": {{
            "requirements": {{
                "hardware": ["string"],
                "software": ["string"],
                "resources": ["string"],
                "timeline": "string",
                "validation": "string"
            }},
            "improvements": {{
                "suggestions": ["string"],
                "implementation": "string",
                "validation": "string",
                "timeline": "string",
                "resources": ["string"]
            }}
        }},
        "computational": {{
            "resources": {{
                "requirements": ["string"],
                "optimization": ["string"],
                "scalability": "string",
                "validation": "string",
                "improvements": ["string"]
            }},
            "improvements": {{
                "suggestions": ["string"],
                "implementation": "string",
                "validation": "string",
                "timeline": "string",
                "resources": ["string"]
            }}
        }},
        "validation": {{
            "technical": {{
                "methods": ["string"],
                "metrics": ["string"],
                "analysis": "string",
                "validation": "string",
                "improvements": ["string"]
            }},
            "improvements": {{
                "suggestions": ["string"],
                "implementation": "string",
                "validation": "string",
                "timeline": "string",
                "resources": ["string"]
            }}
        }}
    }},
    "practical": {{
        "implementation": {{
            "strategy": {{
                "approach": "string",
                "requirements": ["string"],
                "timeline": "string",
                "validation": "string",
                "improvements": ["string"]
            }},
            "improvements": {{
                "suggestions": ["string"],
                "implementation": "string",
                "validation": "string",
                "timeline": "string",
                "resources": ["string"]
            }}
        }},
        "resources": {{
            "requirements": {{
                "personnel": ["string"],
                "equipment": ["string"],
                "materials": ["string"],
                "timeline": "string",
                "validation": "string"
            }},
            "improvements": {{
                "suggestions": ["string"],
                "implementation": "string",
                "validation": "string",
                "timeline": "string",
                "resources": ["string"]
            }}
        }},
        "deployment": {{
            "strategy": {{
                "approach": "string",
                "requirements": ["string"],
                "timeline": "string",
                "validation": "string",
                "improvements": ["string"]
            }},
            "improvements": {{
                "suggestions": ["string"],
                "implementation": "string",
                "validation": "string",
                "timeline": "string",
                "resources": ["string"]
            }}
        }},
        "monitoring": {{
            "strategy": {{
                "approach": "string",
                "metrics": ["string"],
                "analysis": "string",
                "validation": "string",
                "improvements": ["string"]
            }},
            "improvements": {{
                "suggestions": ["string"],
                "implementation": "string",
                "validation": "string",
                "timeline": "string",
                "resources": ["string"]
            }}
        }}
    }},
    "future_work": {{
        "research": {{
            "directions": {{
                "biological": ["string"],
                "technical": ["string"],
                "practical": ["string"],
                "timeline": "string",
                "resources": ["string"]
            }},
            "improvements": {{
                "suggestions": ["string"],
                "implementation": "string",
                "validation": "string",
                "timeline": "string",
                "resources": ["string"]
            }}
        }},
        "validation": {{
            "experiments": {{
                "biological": ["string"],
                "technical": ["string"],
                "practical": ["string"],
                "timeline": "string",
                "resources": ["string"]
            }},
            "improvements": {{
                "suggestions": ["string"],
                "implementation": "string",
                "validation": "string",
                "timeline": "string",
                "resources": ["string"]
            }}
        }},
        "collaborations": {{
            "partners": {{
                "biological": ["string"],
                "technical": ["string"],
                "practical": ["string"],
                "timeline": "string",
                "resources": ["string"]
            }},
            "improvements": {{
                "suggestions": ["string"],
                "implementation": "string",
                "validation": "string",
                "timeline": "string",
                "resources": ["string"]
            }}
        }},
        "funding": {{
            "requirements": {{
                "biological": ["string"],
                "technical": ["string"],
                "practical": ["string"],
                "timeline": "string",
                "resources": ["string"]
            }},
            "improvements": {{
                "suggestions": ["string"],
                "implementation": "string",
                "validation": "string",
                "timeline": "string",
                "resources": ["string"]
            }}
        }}
    }}
}}"""

    def _run_llm(self, prompt: str) -> Dict[str, Any]:
        system_prompt = "You are an expert in scientific analysis integration. Provide your response in valid JSON format."

        try:
            response = self.llm.generate(prompt, system_prompt)
            content = response.get("content") or ""

            try:
                return json.loads(content)
            except json.JSONDecodeError:

                import re
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                else:

                    return {"content": content, "error": "Failed to parse JSON response"}
        except Exception as e:
            return {
                "content": "",
                "error": f"LLM generation failed: {str(e)}",
                "degraded": True,
            }

    def _extract_task_description(self, dataset_analysis: AnalysisResult,
                                 problem_investigation: AnalysisResult,
                                 baseline_assessment: AnalysisResult) -> str:
        """Extract task description from problem investigation"""

        if hasattr(problem_investigation, 'content') and problem_investigation.content:
            if isinstance(problem_investigation.content, dict):
                return problem_investigation.content.get('task_description', 'single cell perturbation prediction')
        return 'single cell perturbation prediction'

    def _extract_dataset_info(self, dataset_analysis: AnalysisResult) -> Dict[str, Any]:
        """Extract dataset information from dataset analysis"""
        if hasattr(dataset_analysis, 'content') and dataset_analysis.content:
            if isinstance(dataset_analysis.content, dict):
                return dataset_analysis.content.get('dataset_info', {})
        return {}

    def _generate_refinement_comments_with_decision_support(self, dataset_analysis: AnalysisResult,
                                                          problem_investigation: AnalysisResult,
                                                          baseline_assessment: AnalysisResult,
                                                          decision_support: Dict[str, Any]) -> Dict[str, Any]:
        """Generate refinement comments with decision support"""
        prompt = self._format_refinement_prompt_with_decision_support(
            dataset_analysis, problem_investigation, baseline_assessment, decision_support
        )


        comments = self._run_llm(prompt)

        return comments

    def _generate_final_recommendations_with_decision_support(self, dataset_analysis: AnalysisResult,
                                                            problem_investigation: AnalysisResult,
                                                            baseline_assessment: AnalysisResult,
                                                            decision_support: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final recommendations with decision support"""
        prompt = self._format_recommendations_prompt_with_decision_support(
            dataset_analysis, problem_investigation, baseline_assessment, decision_support
        )


        recommendations = self._run_llm(prompt)

        return recommendations

    def _format_refinement_prompt_with_decision_support(self, dataset_analysis: AnalysisResult,
                                                      problem_investigation: AnalysisResult,
                                                      baseline_assessment: AnalysisResult,
                                                      decision_support: Dict[str, Any]) -> str:
        """Format prompt for generating refinement comments with decision support"""


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

        return f"""You are an expert in scientific analysis integration with extensive experience in refining and improving research analyses. Your task is to provide comprehensive refinement comments for dataset analysis, problem investigation, and baseline assessment, focusing on integration, consistency, and improvement with decision support.

{decision_context}

1. Review Dataset Analysis:
   - Evaluate biological relevance and interpretability
   - Assess technical quality and reproducibility
   - Check preprocessing recommendations
   - Validate quality assessment
   - Suggest improvements
   - Consider decision support alignment

2. Review Problem Investigation:
   - Evaluate problem definition and scope
   - Assess research questions and hypotheses
   - Check analysis methods and validation
   - Validate biological mechanisms
   - Suggest improvements
   - Consider decision support integration

3. Review Baseline Assessment:
   - Evaluate model analysis and comparison
   - Assess evaluation framework
   - Check performance analysis
   - Validate improvement suggestions
   - Suggest improvements
   - Consider decision support recommendations

4. Ensure Integration:
   - Check consistency across analyses
   - Validate biological relevance
   - Assess technical feasibility
   - Evaluate practical implementation
   - Suggest improvements
   - Integrate decision support insights

Dataset Analysis:
{json.dumps(dataset_analysis.content, indent=2)}

Problem Investigation:
{json.dumps(problem_investigation.content, indent=2)}

Baseline Assessment:
{json.dumps(baseline_assessment.content, indent=2)}

Please provide refinement comments in the following JSON format, incorporating decision support considerations:
{{
    "dataset": {{
        "biological": {{
            "relevance": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string",
                "decision_support_alignment": "string"
            }},
            "interpretability": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string",
                "decision_support_considerations": "string"
            }}
        }},
        "technical": {{
            "quality": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "technical_validation": "string",
                "decision_support_quality": "string"
            }},
            "reproducibility": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "technical_validation": "string",
                "decision_support_reproducibility": "string"
            }}
        }},
        "preprocessing": {{
            "recommendations": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "implementation": "string",
                "decision_support_preprocessing": "string"
            }},
            "validation": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "implementation": "string",
                "decision_support_validation": "string"
            }}
        }},
        "quality": {{
            "assessment": {{
                "evaluation": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "implementation": "string",
                "decision_support_quality_assessment": "string"
            }},
            "validation": {{
                "evaluation": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "implementation": "string",
                "decision_support_quality_validation": "string"
            }}
        }}
    }},
    "problem": {{
        "definition": {{
            "scope": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string",
                "decision_support_scope": "string"
            }},
            "clarity": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string",
                "decision_support_clarity": "string"
            }},
            "feasibility": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string",
                "decision_support_feasibility": "string"
            }}
        }},
        "challenges": {{
            "identification": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string",
                "decision_support_challenges": "string"
            }},
            "mitigation": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string",
                "decision_support_mitigation": "string"
            }}
        }},
        "questions": {{
            "formulation": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string",
                "decision_support_questions": "string"
            }},
            "hypotheses": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string",
                "decision_support_hypotheses": "string"
            }}
        }},
        "methods": {{
            "approaches": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string",
                "decision_support_approaches": "string"
            }},
            "validation": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string",
                "decision_support_method_validation": "string"
            }}
        }}
    }},
    "baseline": {{
        "models": {{
            "analysis": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string",
                "decision_support_model_analysis": "string"
            }},
            "comparison": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string",
                "decision_support_model_comparison": "string"
            }}
        }},
        "evaluation": {{
            "framework": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string",
                "decision_support_evaluation_framework": "string"
            }},
            "metrics": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string",
                "decision_support_evaluation_metrics": "string"
            }}
        }},
        "performance": {{
            "analysis": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string",
                "decision_support_performance_analysis": "string"
            }},
            "comparison": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string",
                "decision_support_performance_comparison": "string"
            }}
        }},
        "improvements": {{
            "suggestions": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string",
                "decision_support_improvement_suggestions": "string"
            }},
            "implementation": {{
                "assessment": "string",
                "issues": ["string"],
                "improvements": ["string"],
                "validation": "string",
                "biological_validation": "string",
                "decision_support_improvement_implementation": "string"
            }}
        }}
    }}
}}"""

    def _format_recommendations_prompt_with_decision_support(self, dataset_analysis: AnalysisResult,
                                                           problem_investigation: AnalysisResult,
                                                           baseline_assessment: AnalysisResult,
                                                           decision_support: Dict[str, Any]) -> str:
        """Format prompt for generating final recommendations with decision support"""


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

        return f"""You are an expert in scientific analysis integration with extensive experience in generating comprehensive recommendations for research projects. Your task is to provide final recommendations based on the integrated analysis results, incorporating decision support information.

{decision_context}

1. Model Selection Recommendations:
   - Recommend appropriate models based on analysis
   - Consider decision support model recommendations
   - Assess biological interpretability requirements
   - Evaluate computational feasibility
   - Plan implementation strategy

2. Evaluation Strategy Recommendations:
   - Design comprehensive evaluation framework
   - Incorporate decision support evaluation metrics
   - Plan biological validation approaches
   - Consider technical validation requirements
   - Establish performance benchmarks

3. Data Preparation Recommendations:
   - Suggest preprocessing strategies
   - Consider decision support data requirements
   - Plan quality control procedures
   - Design validation approaches
   - Address data limitations

4. Implementation Plan Recommendations:
   - Develop detailed implementation roadmap
   - Consider decision support implementation plan
   - Plan resource allocation
   - Design timeline and milestones
   - Address risk mitigation strategies

5. Risk Assessment and Mitigation:
   - Identify potential risks and challenges
   - Consider decision support risk assessment
   - Develop mitigation strategies
   - Plan contingency approaches
   - Design monitoring and evaluation

Dataset Analysis:
{json.dumps(dataset_analysis.content, indent=2)}

Problem Investigation:
{json.dumps(problem_investigation.content, indent=2)}

Baseline Assessment:
{json.dumps(baseline_assessment.content, indent=2)}

Please provide comprehensive final recommendations in the following JSON format, incorporating decision support insights:
{{
    "model_selection": {{
        "recommended_models": {{
            "primary": ["string"],
            "secondary": ["string"],
            "rationale": "string",
            "biological_justification": "string",
            "decision_support_alignment": "string"
        }},
        "implementation_priority": {{
            "high_priority": ["string"],
            "medium_priority": ["string"],
            "low_priority": ["string"],
            "rationale": "string",
            "decision_support_priority": "string"
        }},
        "biological_interpretability": {{
            "requirements": ["string"],
            "approaches": ["string"],
            "validation": "string",
            "decision_support_interpretability": "string"
        }}
    }},
    "evaluation_strategy": {{
        "framework": {{
            "design": "string",
            "components": ["string"],
            "metrics": ["string"],
            "decision_support_framework": "string"
        }},
        "validation": {{
            "biological": ["string"],
            "technical": ["string"],
            "cross_validation": "string",
            "decision_support_validation": "string"
        }},
        "benchmarks": {{
            "baselines": ["string"],
            "state_of_the_art": ["string"],
            "performance_targets": ["string"],
            "decision_support_benchmarks": "string"
        }}
    }},
    "data_preparation": {{
        "preprocessing": {{
            "strategies": ["string"],
            "quality_control": ["string"],
            "normalization": ["string"],
            "decision_support_preprocessing": "string"
        }},
        "requirements": {{
            "data_types": ["string"],
            "quality_standards": ["string"],
            "validation_requirements": ["string"],
            "decision_support_requirements": "string"
        }},
        "limitations": {{
            "identified": ["string"],
            "mitigation": ["string"],
            "validation": ["string"],
            "decision_support_limitations": "string"
        }}
    }},
    "implementation_plan": {{
        "roadmap": {{
            "phases": ["string"],
            "timeline": "string",
            "milestones": ["string"],
            "decision_support_roadmap": "string"
        }},
        "resources": {{
            "computational": ["string"],
            "biological": ["string"],
            "personnel": ["string"],
            "decision_support_resources": "string"
        }},
        "risk_mitigation": {{
            "risks": ["string"],
            "mitigation": ["string"],
            "contingency": ["string"],
            "decision_support_risk_mitigation": "string"
        }}
    }},
    "monitoring_evaluation": {{
        "metrics": {{
            "performance": ["string"],
            "biological": ["string"],
            "technical": ["string"],
            "decision_support_monitoring": "string"
        }},
        "timeline": {{
            "short_term": "string",
            "medium_term": "string",
            "long_term": "string",
            "decision_support_timeline": "string"
        }},
        "success_criteria": {{
            "technical": ["string"],
            "biological": ["string"],
            "clinical": ["string"],
            "decision_support_success": "string"
        }}
    }}
}}"""
