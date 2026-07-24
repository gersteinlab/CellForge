from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

@dataclass
class AnalysisResult:
    """Structured analysis result with confidence scoring"""
    content: Dict[str, Any]
    confidence_score: float
    timestamp: datetime
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "content": self.content,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AnalysisResult':
        """Create from dictionary format"""
        return cls(
            content=data["content"],
            confidence_score=data["confidence_score"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data["metadata"]
        )

@dataclass
class TaskAnalysisReport:
    """Final task analysis report combining all expert analyses"""
    dataset_analysis: AnalysisResult
    problem_investigation: AnalysisResult
    baseline_assessment: AnalysisResult
    refinement_comments: List[Dict[str, Any]]
    final_recommendations: Dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "dataset_analysis": self.dataset_analysis.to_dict(),
            "problem_investigation": self.problem_investigation.to_dict(),
            "baseline_assessment": self.baseline_assessment.to_dict(),
            "refinement_comments": self.refinement_comments,
            "final_recommendations": self.final_recommendations,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TaskAnalysisReport':
        """Create from dictionary format"""
        return cls(
            dataset_analysis=AnalysisResult.from_dict(data["dataset_analysis"]),
            problem_investigation=AnalysisResult.from_dict(data["problem_investigation"]),
            baseline_assessment=AnalysisResult.from_dict(data["baseline_assessment"]),
            refinement_comments=data["refinement_comments"],
            final_recommendations=data["final_recommendations"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )

    def to_markdown(self) -> str:
        """Convert to markdown format with enhanced structure and formatting"""
        sections = [
            "# Task Analysis Report",
            f"Generated on: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n",

            "## 1. Dataset Analysis",
            "### Experimental Design & Scale",
            self._format_section(self.dataset_analysis.content.get("experimental_design", {})),
            "\n### Data Characteristics",
            self._format_section(self.dataset_analysis.content.get("data_characteristics", {})),
            "\n### Preprocessing Considerations",
            self._format_section(self.dataset_analysis.content.get("preprocessing", {})),
            "\n### Quality Assessment",
            self._format_section(self.dataset_analysis.content.get("quality_assessment", {})),

            "\n## 2. Problem Investigation",
            "### Formal Definition",
            self._format_section(self.problem_investigation.content.get("formal_definition", {})),
            "\n### Key Challenges",
            self._format_section(self.problem_investigation.content.get("key_challenges", {})),
            "\n### Research Questions",
            self._format_section(self.problem_investigation.content.get("research_questions", {})),
            "\n### Analysis Methods",
            self._format_section(self.problem_investigation.content.get("analysis_methods", {})),

            "\n## 3. Baseline Assessment",
            "### Baseline Models Analysis",
            self._format_section(self.baseline_assessment.content.get("baseline_models", {})),
            "\n### Evaluation Framework",
            self._format_section(self.baseline_assessment.content.get("evaluation_framework", {})),
            "\n### Performance Analysis",
            self._format_section(self.baseline_assessment.content.get("performance_analysis", {})),
            "\n### Improvement Suggestions",
            self._format_section(self.baseline_assessment.content.get("improvement_suggestions", {})),

            "\n## 4. Refinement Process",
            *[f"\n### Round {i+1}\n" + self._format_refinement_round(comment)
              for i, comment in enumerate(self.refinement_comments)],

            "\n## 5. Final Recommendations",
            "### Data Processing Pipeline",
            self._format_section(self.final_recommendations.get("data_processing", {})),
            "\n### Model Architecture",
            self._format_section(self.final_recommendations.get("model_architecture", {})),
            "\n### Training Strategy",
            self._format_section(self.final_recommendations.get("training_strategy", {})),
            "\n### Evaluation Protocol",
            self._format_section(self.final_recommendations.get("evaluation_protocol", {})),
            "\n### Implementation Roadmap",
            self._format_section(self.final_recommendations.get("implementation_roadmap", {}))
        ]
        return "\n".join(sections)

    def _format_section(self, content: Dict[str, Any]) -> str:
        """Format a section with proper markdown structure"""
        if not content:
            return ""

        formatted_lines = []
        for key, value in content.items():
            if isinstance(value, list):
                formatted_lines.append(f"\n#### {key.title()}")
                for item in value:
                    formatted_lines.append(f"- {item}")
            elif isinstance(value, dict):
                formatted_lines.append(f"\n#### {key.title()}")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, list):
                        formatted_lines.append(f"\n**{subkey.title()}**:")
                        for item in subvalue:
                            formatted_lines.append(f"- {item}")
                    else:
                        formatted_lines.append(f"\n**{subkey.title()}**: {subvalue}")
            else:
                formatted_lines.append(f"\n**{key.title()}**: {value}")

        return "\n".join(formatted_lines)

    def _format_refinement_round(self, comment: Dict[str, Any]) -> str:
        """Format a refinement round with proper structure"""
        sections = []
        for analysis_type, feedback in comment.items():
            sections.append(f"\n#### {analysis_type.replace('_', ' ').title()}")
            for category, items in feedback.items():
                sections.append(f"\n**{category.title()}**:")
                if isinstance(items, list):
                    for item in items:
                        sections.append(f"- {item}")
                else:
                    sections.append(f"- {items}")
        return "\n".join(sections)

    def save_report(self, filepath: str):
        """Save report to markdown file"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())