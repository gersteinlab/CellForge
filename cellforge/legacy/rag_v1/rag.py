"""Legacy RAG facade superseded by :mod:`cellforge.retrieval`."""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    from github import Github
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    print("⚠️  PyGithub not available. GitHub search will be disabled.")


try:
    from .dataparser import DataParser
    from .indexer import VectorIndexer
    from .search import HybridSearcher, SearchResult
    from .utils import TextProcessor
except ImportError:

    import sys
    sys.path.append(str(Path(__file__).parent))
    from dataparser import DataParser
    from indexer import VectorIndexer
    from search import HybridSearcher, SearchResult
    from utils import TextProcessor

class RAGSystem:
    """
    Advanced Retrieval-Augmented Generation system for single cell perturbation analysis.
    Implements Agentic Retrieval with BFS-DFS alternating search strategy and dual Qdrant databases.
    """
    def __init__(self, qdrant_url: str = "localhost", qdrant_port: int = 6333):
        self.qdrant_enabled = os.getenv("QDRANT_ENABLED", "false").lower() in {"1", "true", "yes"}

        with open('config.json', 'r') as f:
            config = json.load(f)

        def _db_cfg(cfg: Dict[str, Any], db_key: str) -> Dict[str, Any]:
            qcfg = cfg.get("qdrant_config", {})
            if "host" in qcfg:
                host = qcfg.get("host", "localhost")
                if isinstance(host, str) and host.startswith(("http://", "https://")):
                    url = host
                else:
                    url = f"http://{host}:{qcfg.get('port', 6333)}"
                return {"url": url, "api_key": qcfg.get("api_key")}
            return qcfg.get(db_key, qcfg.get("CelloFrge", {}))


        self.qdrant_main = None
        self.qdrant_tmp = None
        self._db_disabled = {"main": False, "tmp": False}

        if self.qdrant_enabled:
            try:
                from qdrant_client import QdrantClient
                main_cfg = _db_cfg(config, "CellForge")
                tmp_cfg = _db_cfg(config, "cellforge_tmp")
                self.qdrant_main = QdrantClient(
                    url=main_cfg.get("url", "http://localhost:6333"),
                    api_key=main_cfg.get("api_key")
                )
                self.qdrant_tmp = QdrantClient(
                    url=tmp_cfg.get("url", "http://localhost:6333"),
                    api_key=tmp_cfg.get("api_key")
                )
                print("Successfully initialized dual Qdrant clients")
            except Exception as e:
                print(f"Error initializing Qdrant clients: {e}")
        else:
            print("Qdrant disabled by QDRANT_ENABLED=false; retrieval will use local fallback.")


        self.github_token = os.getenv("GITHUB_TOKEN")
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        self.pubmed_api_key = os.getenv("PUBMED_API_KEY")
        self.pubmed_tool = os.getenv("PUBMED_TOOL", "cellforge")
        self.pubmed_email = os.getenv("PUBMED_EMAIL")


        project_root = Path(__file__).parent.parent
        results_dir = project_root / "data" / "results"

        self.text_processor = TextProcessor()
        self.paper_parser = DataParser(output_path=str(results_dir))
        self.vector_indexer = VectorIndexer()
        self.hybrid_searcher = HybridSearcher()


        if self.github_token and self.github_token != "your_github_token_here":
            self.github_client = Github(self.github_token)
        else:
            self.github_client = None
            print("GitHub token not configured, GitHub search will be disabled")


        self._load_single_cell_terms()
        self.internal_papers_dir = Path(__file__).resolve().parents[2] / "knowledge_base" / "papers"

        self._initialize_collections()

    def _search_local_papers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fallback search over local internal papers directory when Qdrant is unavailable/empty."""
        try:
            if not self.internal_papers_dir.exists():
                return []

            keywords = [k.lower() for k in query.split() if len(k) > 2]
            results: List[Dict[str, Any]] = []
            for p in sorted(self.internal_papers_dir.glob("*.pdf")):
                title = p.stem
                title_l = title.lower()
                score = 0.1
                if keywords:
                    hits = sum(1 for kw in keywords if kw in title_l)
                    score += hits / max(len(keywords), 1)
                if score > 0.1:
                    results.append({
                        "title": title,
                        "content": f"Local internal paper: {title}",
                        "url": str(p),
                        "score": min(score, 1.0),
                        "metadata": {"source_type": "local_internal_pdf"},
                        "source": "local_internal_papers"
                    })
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            return results[:limit]
        except Exception as e:
            print(f"Local paper fallback failed: {e}")
            return []

    def _initialize_collections(self):
        """Initialize collections for Task Analysis in both databases"""
        collections = [
            'papers',
            'code',
            'task_analysis_dataparser',
            'task_analysis_dataset_analyst',
            'task_analysis_problem_investigator',
            'task_analysis_baseline_assessor',
            'task_analysis_refinement',
            'task_analysis_decisions'
        ]

        for collection in collections:
            try:

                if self.qdrant_main:
                    self.qdrant_main.recreate_collection(
                        collection_name=collection,
                        vectors_config={
                            "size": 384,
                            "distance": "Cosine"
                        }
                    )
                    print(f"Initialized collection in main DB: {collection}")
            except Exception as e:
                err = str(e).lower()
                if "connection refused" in err or "[errno 111]" in err:
                    if not self._db_disabled["main"]:
                        print("⚠️  main Qdrant unreachable; disabling main DB for this run.")
                    self._db_disabled["main"] = True
                    self.qdrant_main = None
                else:
                    print(f"Error initializing collection {collection} in main DB: {str(e)}")

            try:

                if self.qdrant_tmp:
                    self.qdrant_tmp.recreate_collection(
                        collection_name=collection,
                        vectors_config={
                            "size": 384,
                            "distance": "Cosine"
                        }
                    )
                    print(f"Initialized collection in tmp DB: {collection}")
            except Exception as e:
                err = str(e).lower()
                if "connection refused" in err or "[errno 111]" in err:
                    if not self._db_disabled["tmp"]:
                        print("⚠️  tmp Qdrant unreachable; disabling tmp DB for this run.")
                    self._db_disabled["tmp"] = True
                    self.qdrant_tmp = None
                else:
                    print(f"Error initializing collection {collection} in tmp DB: {str(e)}")

    def search(self, query: str, collection_name: str = "papers", limit: int = 10,
               use_main_db: bool = True) -> List[Dict[str, Any]]:
        """
        Search in Qdrant database using Agentic Retrieval

        Args:
            query: Search query
            collection_name: Name of the collection to search
            limit: Maximum number of results
            use_main_db: Whether to use main database (True) or tmp database (False)

        Returns:
            List of search results
        """
        qdrant_client = self.qdrant_main if use_main_db else self.qdrant_tmp
        db_name = "main" if use_main_db else "tmp"

        if not qdrant_client:
            print(f"Qdrant {db_name} client not available")
            # Continue with hybrid/local fallback instead of returning early.

        try:

            results = self.hybrid_searcher.search(query, search_type="agentic", limit=limit)


            search_results = []
            for result in results:
                search_results.append({
                    "title": result.title,
                    "content": result.snippet,
                    "url": result.url,
                    "score": result.score,
                    "metadata": result.metadata or {},
                    "source": f"{db_name}_db_{result.source}"
                })
            if not search_results:
                search_results = self._search_local_papers(query, limit)

            return search_results

        except Exception as e:
            print(f"Error searching in {db_name} database: {e}")
            return []

    def store_task_analysis_result(self, result_type: str, data: Dict[str, Any],
                                  use_main_db: bool = True) -> bool:
        """
        Store task analysis result to Qdrant database

        Args:
            result_type: Type of result (e.g., 'dataparser', 'dataset_analyst')
            data: Data to store
            use_main_db: Whether to use main database (True) or tmp database (False)

        Returns:
            True if successful, False otherwise
        """
        qdrant_client = self.qdrant_main if use_main_db else self.qdrant_tmp
        db_name = "main" if use_main_db else "tmp"

        if not qdrant_client:
            print(f"Qdrant {db_name} client not available")
            return False

        try:
            collection_name = f"task_analysis_{result_type}"


            content = str(data)
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = encoder.encode(content)


            from qdrant_client.http import models
            point = models.PointStruct(
                id=hash(content),
                vector=embedding.tolist(),
                payload={
                    "content": content,
                    "data": data,
                    "result_type": result_type,
                    "timestamp": str(Path(__file__).parent.parent / "data" / "results")
                }
            )


            qdrant_client.upsert(
                collection_name=collection_name,
                points=[point]
            )

            print(f"Successfully stored {result_type} result to {db_name} database")
            return True

        except Exception as e:
            print(f"Error storing {result_type} result to {db_name} database: {e}")
            return False

    def get_task_analysis_results(self, result_type: str, query: str = "",
                                 limit: int = 10, use_main_db: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve task analysis results from Qdrant database

        Args:
            result_type: Type of result to retrieve
            query: Search query (empty for all results)
            limit: Maximum number of results
            use_main_db: Whether to use main database (True) or tmp database (False)

        Returns:
            List of task analysis results
        """
        qdrant_client = self.qdrant_main if use_main_db else self.qdrant_tmp
        db_name = "main" if use_main_db else "tmp"

        if not qdrant_client:
            print(f"Qdrant {db_name} client not available")
            return []

        try:
            collection_name = f"task_analysis_{result_type}"

            if query:

                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer('all-MiniLM-L6-v2')
                query_embedding = encoder.encode(query)

                results = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding.tolist(),
                    limit=limit
                )
            else:

                results = qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=limit
                )[0]


            search_results = []
            for result in results:
                search_results.append({
                    "content": result.payload.get("content", ""),
                    "data": result.payload.get("data", {}),
                    "result_type": result.payload.get("result_type", ""),
                    "score": getattr(result, 'score', 1.0),
                    "source": f"{db_name}_db"
                })

            return search_results

        except Exception as e:
            print(f"Error retrieving {result_type} results from {db_name} database: {e}")
            return []

    def get_knowledge_base_summary(self, query: str) -> Dict[str, Any]:
        """Get knowledge base summary for MCP integration"""
        results = self.search(query, limit=30)
        return {
            "query": query,
            "total_papers": len(results),
            "unique_papers": len(set(r.get("title", "") for r in results)),
            "average_score": sum(r.get("score", 0) for r in results) / len(results) if results else 0,
            "sources": list(set(r.get("source", "") for r in results)),
            "top_papers": [
                {
                    "title": r.get("title", ""),
                    "score": r.get("score", 0),
                    "source": r.get("source", ""),
                    "content_preview": r.get("content", "")[:200] + "..." if len(r.get("content", "")) > 200 else r.get("content", "")
                }
                for r in sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:5]
            ],
            "integration_points": [
                "Dataset Analysis",
                "Problem Investigation",
                "Baseline Assessment",
                "Research Plan Generation",
                "MCP Decision Support"
            ]
        }

    def export_knowledge_base_for_mcp(self, query: str) -> str:
        """Export knowledge base in MCP-compatible format"""
        results = self.search(query, limit=50)

        knowledge_base = {
            "query": query,
            "timestamp": str(Path(__file__).parent.parent / "data" / "results"),
            "total_documents": len(results),
            "documents": [
                {
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                    "score": r.get("score", 0),
                    "source": r.get("source", ""),
                    "metadata": r.get("metadata", {})
                }
                for r in results
            ],
            "usage": "This knowledge base is automatically integrated into all analysis components and can be accessed via MCP calls"
        }

        import json
        return json.dumps(knowledge_base, indent=2, ensure_ascii=False)

    def _load_single_cell_terms(self) -> None:
        """Load specialized terms related to single-cell perturbation analysis"""
        self.single_cell_terms = {
            "perturbation_types": [
                "gene_knockout", "gene_knockdown", "CRISPR", "RNAi",
                "drug_treatment", "small_molecule", "compound",
                "overexpression", "transfection", "transduction",
                "perturbation", "intervention", "modification"
            ],
            "technologies": [
                "scRNA-seq", "single-cell RNA sequencing", "10x Genomics",
                "Drop-seq", "Smart-seq2", "CEL-seq2", "sci-RNA-seq",
                "Perturb-seq", "CROP-seq", "Mosaic-seq",
                "single-cell ATAC-seq", "single-cell proteomics",
                "spatial transcriptomics", "multi-omics"
            ],
            "analysis_methods": [
                "differential_expression", "trajectory_analysis",
                "pseudotime", "cell_type_annotation", "clustering",
                "dimensionality_reduction", "batch_correction",
                "integration", "regulatory_network",
                "gene_regulatory_network", "pathway_analysis",
                "enrichment_analysis", "cell_state_transition"
            ],
            "cell_types": [
                "T_cell", "B_cell", "macrophage", "dendritic_cell",
                "stem_cell", "neuron", "fibroblast", "epithelial_cell",
                "cancer_cell", "immune_cell", "progenitor_cell",
                "endothelial_cell", "mesenchymal_cell"
            ],
            "perturbation_effects": [
                "cell_state_change", "differentiation", "proliferation",
                "apoptosis", "cell_cycle", "signaling_pathway",
                "transcriptional_regulation", "epigenetic_modification",
                "metabolic_change", "immune_response", "stress_response",
                "cell_fate_decision", "cellular_plasticity"
            ],
            "model_types": [
                "deep_learning", "neural_network", "transformer",
                "graph_neural_network", "variational_autoencoder",
                "generative_adversarial_network", "recurrent_neural_network",
                "convolutional_neural_network", "attention_mechanism",
                "self_supervised_learning", "transfer_learning"
            ],
            "evaluation_metrics": [
                "accuracy", "precision", "recall", "f1_score",
                "auc_roc", "mean_squared_error", "r2_score",
                "silhouette_score", "adjusted_rand_index",
                "normalized_mutual_information", "pearson_correlation",
                "spearman_correlation", "jaccard_similarity"
            ],
            "dataset_characteristics": [
                "high_dimensional", "sparse", "noisy", "imbalanced",
                "batch_effect", "dropout", "technical_variation",
                "biological_variation", "temporal_dynamics",
                "spatial_heterogeneity", "cell_type_heterogeneity",
                "perturbation_heterogeneity"
            ]
        }

    def get_decision_support(self, task_description: str, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get decision support information for task analysis

        Args:
            task_description: Description of the research task
            dataset_info: Dictionary containing dataset metadata

        Returns:
            Dictionary with decision support information
        """
        try:

            query_terms = []
            if task_description:
                query_terms.extend(task_description.split()[:10])
            if dataset_info and isinstance(dataset_info, dict):
                if 'name' in dataset_info:
                    query_terms.append(dataset_info['name'])
                if 'modality' in dataset_info:
                    query_terms.append(dataset_info['modality'])

            decision_query = " ".join(query_terms) + " single cell perturbation analysis decision support"


            results = self.search(decision_query, limit=5)


            decision_support = {
                "data_preparation": {
                    "recommended_preprocessing": [],
                    "quality_control_steps": [],
                    "normalization_strategies": []
                },
                "implementation_plan": {
                    "model_architecture": [],
                    "training_strategy": [],
                    "evaluation_approach": []
                },
                "risk_assessment": {
                    "potential_issues": [],
                    "mitigation_strategies": [],
                    "validation_requirements": []
                }
            }


            for result in results:
                content = result.get("content", "")
                if "preprocessing" in content.lower() or "normalization" in content.lower():
                    decision_support["data_preparation"]["recommended_preprocessing"].append(content[:200])
                if "model" in content.lower() or "architecture" in content.lower():
                    decision_support["implementation_plan"]["model_architecture"].append(content[:200])
                if "risk" in content.lower() or "challenge" in content.lower():
                    decision_support["risk_assessment"]["potential_issues"].append(content[:200])

            return decision_support

        except Exception as e:
            print(f"Error getting decision support: {e}")
            return {}

    def search_experimental_designs(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Search for experimental design information

        Args:
            task_description: Description of the research task

        Returns:
            List of experimental design information
        """
        try:

            query_terms = task_description.split()[:8]
            design_query = " ".join(query_terms) + " experimental design single cell perturbation"


            results = self.search(design_query, limit=3)


            designs = []
            for result in results:
                designs.append({
                    "title": result.get("title", "Experimental Design"),
                    "content": result.get("content", ""),
                    "source": result.get("source", ""),
                    "score": result.get("score", 0.0)
                })

            return designs

        except Exception as e:
            print(f"Error searching experimental designs: {e}")
            return []

    def search_implementation_guides(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Search for implementation guide information

        Args:
            task_description: Description of the research task

        Returns:
            List of implementation guide information
        """
        try:

            query_terms = task_description.split()[:8]
            guide_query = " ".join(query_terms) + " implementation guide model architecture"


            results = self.search(guide_query, limit=3)


            guides = []
            for result in results:
                guides.append({
                    "title": result.get("title", "Implementation Guide"),
                    "content": result.get("content", ""),
                    "source": result.get("source", ""),
                    "score": result.get("score", 0.0)
                })

            return guides

        except Exception as e:
            print(f"Error searching implementation guides: {e}")
            return []

    def search_evaluation_frameworks(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Search for evaluation framework information

        Args:
            task_description: Description of the research task

        Returns:
            List of evaluation framework information
        """
        try:

            query_terms = task_description.split()[:8]
            eval_query = " ".join(query_terms) + " evaluation framework metrics assessment"


            results = self.search(eval_query, limit=3)


            frameworks = []
            for result in results:
                frameworks.append({
                    "title": result.get("title", "Evaluation Framework"),
                    "content": result.get("content", ""),
                    "source": result.get("source", ""),
                    "score": result.get("score", 0.0)
                })

            return frameworks

        except Exception as e:
            print(f"Error searching evaluation frameworks: {e}")
            return []
