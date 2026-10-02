"""Task Analysis public API with lazy imports.

Importing the package should not initialize Scanpy, sentence transformers, or
network clients. Heavy components are loaded only when requested.
"""

from importlib import import_module


_EXPORTS = {
    "AnalysisResult": (".data_structures", "AnalysisResult"),
    "TaskAnalysisReport": (".data_structures", "TaskAnalysisReport"),
    "DatasetAnalyst": (".dataset_analyst", "DatasetAnalyst"),
    "ProblemInvestigator": (".problem_investigator", "ProblemInvestigator"),
    "BaselineAssessor": (".baseline_assessor", "BaselineAssessor"),
    "RefinementAgent": (".refinement_agent", "RefinementAgent"),
    "CollaborationSystem": (".collaboration", "CollaborationSystem"),
    "Agent": (".collaboration", "Agent"),
    "View": (".view", "View"),
    "MultiView": (".view_multi", "MultiView"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
