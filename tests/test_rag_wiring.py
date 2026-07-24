import importlib.util
import json
import sys
import types
from pathlib import Path

from cellforge.Method_Design.expert_discussion import ExpertDiscussion
from cellforge.Task_Analysis.refinement_agent import RefinementAgent
from cellforge.retrieval import LiteratureRetriever


ROOT = Path(__file__).resolve().parents[1]


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("cellforge_cli_for_test", ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SpyRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, top_k=5):
        self.queries.append((query, top_k))
        return [{"content": "evidence", "source": "fixture", "relevance_score": 1.0}]


def test_task_analysis_cli_injects_shared_retriever(tmp_path, monkeypatch):
    cli = _load_cli_module()
    spy = SpyRetriever()
    captured = {}

    fake_module = types.ModuleType("cellforge.Task_Analysis.main")

    def fake_run(task_description, dataset_info, retriever=None):
        captured["retriever"] = retriever
        return object()

    fake_module.run_task_analysis = fake_run
    monkeypatch.setitem(sys.modules, "cellforge.Task_Analysis.main", fake_module)
    monkeypatch.setattr(
        LiteratureRetriever,
        "from_env",
        classmethod(lambda cls, trace_dir=None: spy),
    )
    monkeypatch.chdir(tmp_path)

    assert cli.run_task_analysis(
        {
            "task_description": "Adamson CRISPRi Perturb-seq prediction",
            "dataset_path": "data/datasets/adamson.h5ad",
        }
    )
    assert captured["retriever"] is spy


def test_method_design_cli_injects_shared_retriever(tmp_path, monkeypatch):
    cli = _load_cli_module()
    spy = SpyRetriever()
    captured = {}

    analysis_dir = tmp_path / "data" / "analyses" / "adamson"
    analysis_dir.mkdir(parents=True)
    report = analysis_dir / "task_analysis_20260724.json"
    report.write_text(json.dumps({"task_type": "gene_knockout"}), encoding="utf-8")

    method_package = types.ModuleType("cellforge.Method_Design")

    def fake_generate_research_plan(**kwargs):
        captured.update(kwargs)
        return {"json": {}, "markdown": ""}

    method_package.generate_research_plan = fake_generate_research_plan
    method_main = types.ModuleType("cellforge.Method_Design.main")
    method_main.load_task_analysis = lambda file_path, latest=False: {
        "task_type": "gene_knockout",
        "dataset": {"name": "adamson"},
    }
    monkeypatch.setitem(sys.modules, "cellforge.Method_Design", method_package)
    monkeypatch.setitem(sys.modules, "cellforge.Method_Design.main", method_main)
    monkeypatch.setattr(
        LiteratureRetriever,
        "from_env",
        classmethod(lambda cls, trace_dir=None: spy),
    )
    monkeypatch.chdir(tmp_path)

    assert cli.run_method_design({"task_description": "Adamson"})
    assert captured["rag_retriever"] is spy


def test_cli_entrypoint_maps_phase_failure_to_exit_one(monkeypatch):
    cli = _load_cli_module()
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _path: {
            "task_description": "fixture",
            "dataset_path": "data/datasets/fixture.h5ad",
            "code_generation": {},
        },
    )
    monkeypatch.setattr(cli, "run_task_analysis", lambda _config: False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["cellforge", "--phase", "task_analysis"],
    )

    assert cli.cli_entrypoint() == 1


def test_method_design_formats_unified_retrieval_contract():
    class FakeLLM:
        def get_config_status(self):
            return {"custom_configured": True, "model_name": "fixture"}

    class UnifiedRetriever:
        def retrieve(self, query, top_k=5):
            return [{
                "title": "Adamson Perturb-seq",
                "abstract": "A traceable perturbation-screen result.",
                "score": 0.91,
                "source": "pubmed",
                "evidence_id": "cf:doi:10.1016/j.cell.2016.11.048",
            }]

    discussion = ExpertDiscussion(FakeLLM(), rag_retriever=UnifiedRetriever())
    context = discussion._get_relevant_knowledge(
        "deep_learning", "gene_knockout", "proposal"
    )

    assert "Adamson Perturb-seq" in context
    assert "A traceable perturbation-screen result." in context
    assert "cf:doi:10.1016/j.cell.2016.11.048" in context


def test_refinement_accepts_null_model_content():
    class NullContentLLM:
        def generate(self, prompt, system_prompt):
            return {"content": None}

    agent = object.__new__(RefinementAgent)
    agent.llm = NullContentLLM()

    result = agent._run_llm("return JSON")

    assert result["content"] == ""
    assert result["error"] == "Failed to parse JSON response"
