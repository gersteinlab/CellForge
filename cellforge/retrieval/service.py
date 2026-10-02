from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import os
import re
import threading

from .local_corpus import LocalCorpusProvider
from .models import PaperRecord, RetrievalTrace
from .providers import CrossrefProvider, PubMedProvider, SemanticScholarProvider
from ..paths import data_path, resolve_workspace_path


class LiteratureRetriever:
    """One retrieval service shared by Task Analysis and Method Design."""

    def __init__(
        self,
        literature_dir: Path,
        trace_dir: Optional[Path] = None,
        online: bool = True,
        providers: Optional[Iterable[Any]] = None,
    ):
        self.literature_dir = Path(literature_dir).expanduser().resolve()
        self.trace_dir = Path(trace_dir or self.literature_dir / "traces").expanduser().resolve()
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[tuple[str, int], List[Dict[str, Any]]] = {}
        self._trace_lock = threading.Lock()
        self.last_trace: Optional[RetrievalTrace] = None
        self._disabled_providers: Dict[str, str] = {}
        if providers is not None:
            self.providers = list(providers)
        else:
            self.providers = [LocalCorpusProvider(self.literature_dir)]
            if online:
                self.providers.extend(
                    [
                        PubMedProvider(
                            api_key=os.getenv("PUBMED_API_KEY"),
                            email=os.getenv("PUBMED_EMAIL"),
                            tool=os.getenv("PUBMED_TOOL", "cellforge"),
                        ),
                        CrossrefProvider(
                            email=os.getenv("CROSSREF_EMAIL") or os.getenv("PUBMED_EMAIL")
                        ),
                        SemanticScholarProvider(
                            api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY")
                        ),
                    ]
                )

    @classmethod
    def from_env(cls, trace_dir: Optional[Path] = None) -> "LiteratureRetriever":
        configured_root = os.getenv("CELLFORGE_LITERATURE_DIR", "").strip()
        root = (
            resolve_workspace_path(configured_root)
            if configured_root
            else data_path("literature")
        )
        online = os.getenv("CELLFORGE_ONLINE_RETRIEVAL", "true").lower() in {
            "1", "true", "yes",
        }
        return cls(Path(root), trace_dir=trace_dir, online=online)

    @staticmethod
    def _dedupe_key(paper: PaperRecord) -> str:
        if paper.doi:
            return f"doi:{paper.doi.lower().strip()}"
        if paper.pmid:
            return f"pmid:{paper.pmid}"
        if paper.semantic_scholar_id:
            return f"s2:{paper.semantic_scholar_id}"
        title = re.sub(r"\W+", " ", paper.title.lower()).strip()
        return f"title:{title}|{paper.year or ''}"

    def _write_trace(self, trace: RetrievalTrace) -> None:
        target = self.trace_dir / "retrieval_trace.jsonl"
        line = json.dumps(trace.to_dict(), ensure_ascii=False)
        with self._trace_lock:
            with target.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def search(
        self,
        query: str,
        collection_name: str = "papers",
        limit: int = 10,
        use_main_db: bool = True,
    ) -> List[Dict[str, Any]]:
        del collection_name, use_main_db
        cache_key = (query.strip().lower(), limit)
        if cache_key in self._cache:
            return [dict(item) for item in self._cache[cache_key]]

        statuses: Dict[str, Dict[str, Any]] = {}
        ranked: List[tuple[str, int, PaperRecord]] = []
        active_providers = []
        for provider in self.providers:
            if provider.name in self._disabled_providers:
                statuses[provider.name] = {
                    "status": "disabled",
                    "count": 0,
                    "error": self._disabled_providers[provider.name],
                }
            else:
                active_providers.append(provider)
        with ThreadPoolExecutor(max_workers=max(1, len(self.providers))) as executor:
            futures = {
                executor.submit(provider.search, query, limit): provider
                for provider in active_providers
            }
            for future in as_completed(futures):
                provider = futures[future]
                try:
                    records = future.result()
                    statuses[provider.name] = {"status": "ok", "count": len(records)}
                    ranked.extend((provider.name, rank, record) for rank, record in enumerate(records))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    statuses[provider.name] = {
                        "status": "failed",
                        "count": 0,
                        "error": error,
                    }
                    self._disabled_providers[provider.name] = error

        fused: Dict[str, PaperRecord] = {}
        scores: Dict[str, float] = {}
        for provider_name, rank, record in ranked:
            key = self._dedupe_key(record)
            reciprocal_rank = 1.0 / (60 + rank + 1)
            scores[key] = scores.get(key, 0.0) + reciprocal_rank
            if key in fused:
                fused[key].merge(record)
            else:
                fused[key] = record
            if provider_name not in fused[key].source_providers:
                fused[key].source_providers.append(provider_name)

        ordered = sorted(fused.items(), key=lambda item: scores[item[0]], reverse=True)
        results = []
        max_score = max((scores[key] for key, _ in ordered), default=1.0)
        for key, paper in ordered[:limit]:
            paper.score = scores[key] / max_score
            results.append(paper.to_result())

        trace = RetrievalTrace(
            query=query,
            requested_limit=limit,
            providers=statuses,
            results=[
                {
                    "paper_id": result["paper_id"],
                    "evidence_id": result["evidence_id"],
                    "title": result["title"],
                    "score": result["score"],
                    "source": result["source"],
                }
                for result in results
            ],
        )
        self.last_trace = trace
        self._write_trace(trace)
        self._cache[cache_key] = [dict(item) for item in results]
        return results

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.search(query, limit=top_k)

    def search_for_task(
        self,
        task_description: str,
        dataset_info: Dict[str, Any],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Build a concise literature query from task and dataset metadata.

        Long natural-language task descriptions over-constrain bibliographic
        search engines. Dataset names normally start with the first author, so
        use that stable token plus the assay family for primary discovery.
        """
        dataset_name = str(dataset_info.get("dataset_name", "")).strip()
        author_match = re.match(r"([A-Z][a-z]+)", dataset_name)
        author = author_match.group(1) if author_match else ""
        perturbation = str(dataset_info.get("perturbation_type", "")).lower()
        modality = str(dataset_info.get("data_type", "")).lower()
        if "crispr" in perturbation:
            assay = "Perturb-seq"
        elif "drug" in perturbation or "chemical" in perturbation:
            assay = "single-cell drug perturbation"
        elif "cytokine" in perturbation:
            assay = "single-cell cytokine perturbation"
        else:
            assay = "single-cell perturbation"
        query = " ".join(part for part in (author, assay) if part).strip()
        if not author:
            compact = " ".join(task_description.split())
            query = compact[:300]
        if not author and modality and "single" not in query.lower():
            query = f"{query} {modality}"
        return self.search(query, limit=limit)

    def get_decision_support(
        self, task_description: str, dataset_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        results = self.search_for_task(task_description, dataset_info, limit=10)
        return {
            "task": task_description,
            "dataset": dataset_info,
            "evidence": results,
            "evidence_ids": [item["evidence_id"] for item in results],
        }

    def search_experimental_designs(self, task_description: str) -> List[Dict[str, Any]]:
        return self.search(f"{task_description} experimental design", limit=8)

    def search_implementation_guides(self, task_description: str) -> List[Dict[str, Any]]:
        return self.search(f"{task_description} implementation method model", limit=8)

    def search_evaluation_frameworks(self, task_description: str) -> List[Dict[str, Any]]:
        return self.search(f"{task_description} benchmark evaluation metrics", limit=8)
