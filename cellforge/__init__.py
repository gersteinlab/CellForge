"""
cellforge: Open-Ended Autonomous Design of Computational Methods for Single-Cell Omics via Multi-Agent Collaboration

A cutting-edge end-to-end multi-agent framework that revolutionizes single-cell data analysis
through intelligent task decomposition, automated method design, and collaborative problem-solving.
"""

from importlib import import_module

__version__ = "0.1.0"
__author__ = "cellforge Team"
__email__ = "cellforge@example.com"

_LAZY_MODULES = {
    "Task_Analysis": ".Task_Analysis",
    "Method_Design": ".Method_Design",
    "Code_Generation": ".Code_Generation",
    "RAG": ".RAG",
}


def __getattr__(name):
    """Load heavyweight workflow modules only when callers request them.

    Code-generation backends should be importable without importing Scanpy,
    plotting libraries, retrieval clients, or the full task-analysis stack.
    """

    if name in _LAZY_MODULES:
        module = import_module(_LAZY_MODULES[name], __name__)
        globals()[name] = module
        return module
    if name == "LLMInterface":
        value = import_module(".llm", __name__).LLMInterface
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Main workflow function
def run_end_to_end_workflow(task_description: str, dataset_info: dict = None):
    """
    Run the complete end-to-end workflow from task analysis to code generation.

    Args:
        task_description: Description of the analysis task
        dataset_info: Optional dataset information dictionary

    Returns:
        bool: True if workflow completed successfully, False otherwise
    """
    from .end_to_end_workflow import EndToEndWorkflow

    workflow = EndToEndWorkflow()
    return workflow.run_complete_workflow(task_description, dataset_info)

# Convenience imports
__all__ = [
    "run_end_to_end_workflow",
    "LLMInterface",
    "Task_Analysis",
    "Method_Design",
    "Code_Generation",
    "RAG"
]
