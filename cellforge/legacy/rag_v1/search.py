"""Legacy agentic search implementation superseded by ``cellforge.retrieval``."""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer

try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False
    print("⚠️ SerpAPI not available. Web search will be disabled.")

try:
    from github import Github
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    print("⚠️ PyGithub not available. GitHub search will be disabled.")

@dataclass
class SearchResult:
    """Search result data structure"""
    title: str
    url: str
    snippet: str
    source: str
    score: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class HybridSearcher:
    """Agentic Retrieval searcher with dual Qdrant databases and dynamic search capabilities"""

    def __init__(self):
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
                print("Successfully initialized dual Qdrant clients for Task Analysis")
            except Exception as e:
                print(f"Error initializing Qdrant clients: {e}")
                self.qdrant_main = None
                self.qdrant_tmp = None
        else:
            self.qdrant_main = None
            self.qdrant_tmp = None
            print("Qdrant disabled by QDRANT_ENABLED=false; using local retrieval fallback.")
        self._db_disabled = {"CellForge": False, "cellforge_tmp": False}

        self.logger = logging.getLogger("cellforge.task_analysis.searcher")


        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')


        # Support both key names used in this repo.
        self.serpapi_key = os.getenv("SERPAPI_KEY") or os.getenv("SERPAPI_API_KEY")
        self.github_token = os.getenv("GITHUB_TOKEN")

        self.web_searcher = None
        self.github_searcher = None
        self.internal_papers_dir = Path(__file__).resolve().parents[2] / "knowledge_base" / "papers"

        if SERPAPI_AVAILABLE and self.serpapi_key:
            self.web_searcher = WebSearcher(self.serpapi_key)

        if GITHUB_AVAILABLE and self.github_token:
            self.github_searcher = GitHubSearcher(self.github_token)

    def search(self, query: str, search_type: str = "agentic",
               limit: int = 50) -> List[SearchResult]:
        """
        Agentic Retrieval with optimized BFS-DFS alternating search strategy
        """
        print(f"Starting Agentic Retrieval for query: {query}")


        keywords = self._extract_keywords(query)
        print(f"Initial keywords: {keywords}")


        current_query = query
        all_results = []
        visited_docs = set()
        layer = 0
        max_layers = 2
        overlap_threshold = 0.7
        relevance_threshold = 0.1

        while layer < max_layers:
            print(f"\nLayer {layer + 1}: {'BFS' if layer % 2 == 0 else 'DFS'}")


            if layer % 2 == 0:
                layer_results = self._bfs_search(current_query, limit, visited_docs)
            else:
                layer_results = self._dfs_search(current_query, limit, visited_docs, all_results)

            if not layer_results:
                print(f"⚠️ No new results found in layer {layer + 1}")
                break



            relevant_results = [r for r in layer_results if r.score >= relevance_threshold]
            if not relevant_results:
                print(f"⚠️ All results below relevance threshold {relevance_threshold}")
                break


            all_results.extend(relevant_results)


            if layer > 0:
                overlap = self._calculate_query_overlap(current_query, query)
                if overlap > overlap_threshold:
                    print(f"⚠️ Query overlap {overlap:.2f} exceeds threshold {overlap_threshold}")
                    break


            current_query = self._generate_next_query(relevant_results, keywords)
            print(f"Next query: {current_query}")

            layer += 1


        fused_results = self._fuse_results(all_results, query)


        fused_results = self._sort_by_relevance_and_recency(fused_results)

        final_results = fused_results[:limit]
        if not final_results:
            final_results = self._search_local_papers(query, limit)
        elif len(final_results) < limit:
            local_extra = self._search_local_papers(query, limit - len(final_results))
            final_results.extend(local_extra)

        # Add external scientific search if available and still under limit.
        if len(final_results) < limit and self.web_searcher:
            external = self.search_scientific(query, limit=limit - len(final_results))
            if external:
                final_results.extend(external)

        # De-duplicate by title/url and keep best-scored entries.
        if final_results:
            dedup = {}
            for r in final_results:
                key = (r.title.strip().lower(), r.url.strip().lower())
                if key not in dedup or r.score > dedup[key].score:
                    dedup[key] = r
            final_results = sorted(dedup.values(), key=lambda x: x.score, reverse=True)[:limit]


        if final_results:
            self._store_to_tmp_db(final_results, query)

        print(f"\nAgentic Retrieval completed.")


        return final_results

    def _search_local_papers(self, query: str, limit: int) -> List[SearchResult]:
        """Fallback search in local internal papers directory."""
        try:
            if not self.internal_papers_dir.exists():
                return []
            keywords = [k.lower() for k in query.split() if len(k) > 2]
            results: List[SearchResult] = []
            for p in sorted(self.internal_papers_dir.glob("*.pdf")):
                title = p.stem
                title_l = title.lower()
                score = 0.1
                if keywords:
                    hits = sum(1 for kw in keywords if kw in title_l)
                    score += hits / max(len(keywords), 1)
                if score > 0.1:
                    results.append(SearchResult(
                        title=title,
                        url=str(p),
                        snippet=f"Local internal paper: {title}",
                        source="local_internal_papers",
                        score=min(score, 1.0),
                        metadata={"source_type": "local_internal_pdf"}
                    ))
            results.sort(key=lambda x: x.score, reverse=True)
            if results:
                print(f"Using local paper fallback: {len(results[:limit])} results.")
            return results[:limit]
        except Exception as e:
            print(f"Local paper fallback failed: {e}")
            return []

    def _bfs_search(self, query: str, limit: int, visited_docs: set) -> List[SearchResult]:
        """Breadth-first search layer with context similarity"""
        print(f"BFS: Searching for '{query}'")

        results = []


        keywords = self._extract_keywords(query)
        search_queries = self._generate_context_queries(query, keywords)

        print(f"Generated {len(search_queries)} context queries: {search_queries}")


        for search_query in search_queries:

            if self.qdrant_main:
                try:

                    query_limit = max(10, limit // len(search_queries))
                    main_results = self._search_qdrant(search_query, self.qdrant_main, "CellForge", query_limit)
                    results.extend(main_results)
                except Exception as e:
                    print(f"  ❌ Main DB BFS failed for '{search_query}': {e}")

            if self.qdrant_tmp:
                try:

                    query_limit = max(10, limit // len(search_queries))
                    tmp_results = self._search_qdrant(search_query, self.qdrant_tmp, "cellforge_tmp", query_limit)
                    results.extend(tmp_results)
                except Exception as e:
                    print(f"  ❌ Tmp DB BFS failed for '{search_query}': {e}")


        new_results = []
        for r in results:

            doc_id = hash(f"{r.title}_{r.snippet[:100]}")
            if doc_id not in visited_docs:
                visited_docs.add(doc_id)
                new_results.append(r)

        print(f" BFS found {len(new_results)} new results")
        return new_results

    def _generate_context_queries(self, original_query: str, keywords: List[str]) -> List[str]:
        """Generate multiple search queries for context similarity"""
        queries = [original_query]


        if keywords:

            for keyword in keywords[:3]:
                if len(keyword) > 3:
                    queries.append(keyword)


            if len(keywords) >= 2:
                queries.append(" ".join(keywords[:2]))
                if len(keywords) >= 3:
                    queries.append(" ".join(keywords[:3]))


        context_queries = self._get_context_queries(original_query, keywords)
        queries.extend(context_queries[:3])


        unique_queries = list(dict.fromkeys(queries))
        return unique_queries[:5]

    def _get_context_queries(self, original_query: str, keywords: List[str]) -> List[str]:
        """Generate context-specific queries based on domain knowledge"""
        context_queries = []


        context_mappings = {
            'single': ['single cell', 'single-cell', 'single cell analysis', 'single cell technology'],
            'cell': ['cell perturbation', 'cellular response', 'cell state', 'cell type'],
            'perturbation': ['perturbation analysis', 'perturbation study', 'perturbation experiment', 'perturbation modeling'],
            'crispr': ['crispr cas9', 'crispr interference', 'gene editing', 'genetic perturbation'],
            'gene': ['gene expression', 'gene regulation', 'gene network', 'transcriptional regulation'],
            'expression': ['gene expression', 'transcriptional response', 'expression analysis'],
            'model': ['predictive model', 'deep learning model', 'machine learning model', 'neural network'],
            'analysis': ['single cell analysis', 'perturbation analysis', 'expression analysis', 'data analysis'],
            'deep': ['deep learning', 'neural network', 'transformer model', 'attention mechanism'],
            'learning': ['machine learning', 'deep learning', 'self supervised learning', 'transfer learning']
        }


        for keyword in keywords:
            if keyword in context_mappings:
                context_queries.extend(context_mappings[keyword])

        return context_queries

    def _dfs_search(self, query: str, limit: int, visited_docs: set, previous_results: List[SearchResult]) -> List[SearchResult]:
        """Depth-first search layer - follows highest-scoring paths"""
        print(f" DFS: Following high-scoring paths from '{query}'")

        if not previous_results:
            return []


        top_docs = sorted(previous_results, key=lambda x: x.score, reverse=True)[:3]

        results = []
        for doc in top_docs:

            doc_keywords = self._extract_keywords(doc.snippet)
            if doc_keywords:

                focused_query = " ".join(doc_keywords[:3])
                print(f" DFS following: {focused_query}")



                focused_limit = max(10, limit // 2)
                focused_results = self._bfs_search(focused_query, focused_limit, visited_docs)
                results.extend(focused_results)


        print(f" DFS trying original query with different focus")

        original_limit = max(10, limit // 2)
        original_focused_results = self._bfs_search(query, original_limit, visited_docs)
        results.extend(original_focused_results)

        print(f" DFS found {len(results)} new results")
        return results

    def _calculate_query_overlap(self, current_query: str, original_query: str) -> float:
        """Calculate overlap between current and original query"""
        current_words = set(current_query.lower().split())
        original_words = set(original_query.lower().split())

        if not original_words:
            return 0.0

        overlap = len(current_words.intersection(original_words)) / len(original_words)
        return overlap

    def _generate_next_query(self, results: List[SearchResult], original_keywords: List[str]) -> str:
        """Generate next query based on current results and original keywords"""
        if not results:
            return " ".join(original_keywords[:3])


        top_results = sorted(results, key=lambda x: x.score, reverse=True)[:3]
        all_text = " ".join([r.snippet for r in top_results])


        result_keywords = self._extract_keywords(all_text)


        domain_terms = [
            'single', 'cell', 'perturbation', 'crispr', 'gene', 'expression',
            'transcriptomics', 'rna', 'sequencing', 'deep', 'learning', 'model',
            'prediction', 'analysis', 'experimental', 'design', 'genomics',
            'proteomics', 'multiomics', 'knockout', 'knockdown', 'overexpression',
            'differential', 'trajectory', 'pseudotime', 'clustering', 'regulatory',
            'network', 'pathway', 'enrichment', 'annotation', 'state', 'transition',
            'differentiation', 'proliferation', 'apoptosis', 'cycle', 'signaling',
            'transcriptional', 'epigenetic', 'metabolic', 'immune', 'response',
            'stress', 'fate', 'plasticity', 'neural', 'transformer', 'graph',
            'variational', 'autoencoder', 'generative', 'adversarial', 'recurrent',
            'convolutional', 'attention', 'self-supervised', 'transfer'
        ]


        filtered_keywords = [kw for kw in result_keywords if kw in domain_terms]


        combined_keywords = filtered_keywords[:3] + original_keywords[:2]
        combined_keywords = list(dict.fromkeys(combined_keywords))


        if len(combined_keywords) < 2:
            combined_keywords = original_keywords[:3]

        return " ".join(combined_keywords)

    def _search_qdrant(self, query: str, qdrant_client, db_name: str, limit: int) -> List[SearchResult]:
        """Search in Qdrant database"""
        try:

            limit = max(1, limit)
            print(f"Encoding query: {query[:50]}...")

            query_embedding = self.encoder.encode(query)


            try:
                collections = qdrant_client.get_collections()
                collection_names = [col.name for col in collections.collections]
                print(f"Available collections in {db_name}: {collection_names}")
            except Exception as e:
                self._handle_qdrant_error(db_name, f"Failed to get collections: {e}")
                return []

            results = []


            if "CellForge" in collection_names:
                try:
                    print(f"Searching in CellForge collection...")

                    cellforge_limit = min(limit * 3, 100)
                    cellforge_results = self._query_qdrant(
                        qdrant_client=qdrant_client,
                        collection_name="CellForge",
                        query_vector=query_embedding.tolist(),
                        limit=cellforge_limit
                    )


                    for result in cellforge_results:

                        title = result.payload.get("source", result.payload.get("title", "CellForge Document"))
                        text_content = result.payload.get("text", result.payload.get("content", result.payload.get("abstract", "")))

                        results.append(SearchResult(
                            title=title,
                            url=result.payload.get("url", ""),
                            snippet=text_content[:500] if text_content else "",
                            source=f"{db_name}_CellForge",
                            score=result.score,
                            metadata=result.payload
                        ))
                except Exception as e:
                    self._handle_qdrant_error(db_name, f"CellForge search failed: {e}")
            else:
                print(f"CellForge collection not found in {db_name}")


            if "papers" in collection_names:
                try:
                    print(f"Searching in papers collection...")

                    papers_limit = max(10, limit // 2)
                    papers_results = self._query_qdrant(
                        qdrant_client=qdrant_client,
                        collection_name="papers",
                        query_vector=query_embedding.tolist(),
                        limit=papers_limit
                    )


                    for result in papers_results:

                        title = result.payload.get("title", result.payload.get("source", "Paper"))
                        content = result.payload.get("content", result.payload.get("abstract", result.payload.get("text", "")))

                        results.append(SearchResult(
                            title=title,
                            url=result.payload.get("url", ""),
                            snippet=content[:500] if content else "",
                            source=f"{db_name}_papers",
                            score=result.score,
                            metadata=result.payload
                        ))
                except Exception as e:
                    self._handle_qdrant_error(db_name, f"Papers search failed: {e}")


            if "code" in collection_names:
                try:
                    print(f" Searching in code collection...")

                    code_limit = max(10, limit // 2)
                    code_results = self._query_qdrant(
                        qdrant_client=qdrant_client,
                        collection_name="code",
                        query_vector=query_embedding.tolist(),
                        limit=code_limit
                    )


                    for result in code_results:

                        title = result.payload.get("title", result.payload.get("source", "Code"))
                        content = result.payload.get("content", result.payload.get("text", result.payload.get("code", "")))

                        results.append(SearchResult(
                            title=title,
                            url=result.payload.get("url", ""),
                            snippet=content[:500] if content else "",
                            source=f"{db_name}_code",
                            score=result.score,
                            metadata=result.payload
                        ))
                except Exception as e:
                    self._handle_qdrant_error(db_name, f"Code search failed: {e}")
            return results

        except Exception as e:
            self._handle_qdrant_error(db_name, f"Qdrant search failed: {e}")
            self.logger.error(f"Qdrant search failed for {db_name}: {e}")
            return []

    def _query_qdrant(self, qdrant_client, collection_name: str, query_vector: List[float], limit: int):
        """Compatibility wrapper for qdrant-client versions."""
        if hasattr(qdrant_client, "search"):
            return qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit
            )
        if hasattr(qdrant_client, "query_points"):
            res = qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit
            )
            if hasattr(res, "points"):
                return res.points
            if isinstance(res, list):
                return res
            return []
        raise AttributeError("Qdrant client has neither 'search' nor 'query_points'")

    def _handle_qdrant_error(self, db_name: str, message: str) -> None:
        """Disable repeated calls when DB is unavailable."""
        err = message.lower()
        if "connection refused" in err or "[errno 111]" in err:
            if not self._db_disabled.get(db_name, False):
                print(f"  ⚠️ {db_name} unreachable; disabling for this run.")
            self._db_disabled[db_name] = True
            if db_name == "CellForge":
                self.qdrant_main = None
            else:
                self.qdrant_tmp = None
        else:
            print(f"  ❌ {message}")

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords and phrases from query for enhanced search"""
        import re


        domain_phrases = [

            'single cell perturbation', 'single-cell perturbation', 'single cell rna sequencing', 'single-cell rna sequencing',
            'gene expression prediction', 'predictive model', 'experimental design', 'deep learning model',
            'machine learning model', 'neural network', 'graph neural network', 'transformer model',
            'attention mechanism', 'self supervised learning', 'transfer learning', 'multi omics',
            'differential expression', 'trajectory analysis', 'pseudotime analysis', 'cell type annotation',
            'dimensionality reduction', 'batch correction', 'regulatory network', 'gene regulatory network',
            'pathway analysis', 'enrichment analysis', 'cell state transition', 'cell fate decision',
            'cellular plasticity', 'transcriptional regulation', 'epigenetic modification', 'metabolic change',
            'immune response', 'stress response', 'cell cycle', 'signaling pathway', 'variational autoencoder',
            'generative adversarial network', 'recurrent neural network', 'convolutional neural network',
            'mean squared error', 'pearson correlation', 'spearman correlation', 'jaccard similarity',
            'high dimensional', 'technical variation', 'biological variation', 'temporal dynamics',
            'spatial heterogeneity', 'cell type heterogeneity', 'perturbation heterogeneity',
            'crispr interference', 'crispr cas9', 'gene knockout', 'gene knockdown', 'gene overexpression',
            'drug treatment', 'small molecule', 'compound treatment', 'transfection', 'transduction',
            'scRNA-seq', '10x genomics', 'drop-seq', 'smart-seq2', 'cel-seq2', 'sci-rna-seq',
            'perturb-seq', 'crop-seq', 'mosaic-seq', 'single cell atac-seq', 'spatial transcriptomics',

            'single cell analysis', 'cell perturbation', 'gene perturbation', 'cellular response',
            'transcriptional response', 'gene regulation', 'cell state', 'cell type', 'cell population',
            'gene network', 'regulatory network', 'signaling network', 'metabolic network',
            'cell differentiation', 'cell proliferation', 'cell death', 'cell survival',
            'drug response', 'treatment response', 'perturbation response', 'cellular perturbation',
            'genetic perturbation', 'chemical perturbation', 'environmental perturbation',
            'single cell technology', 'single cell method', 'single cell approach', 'single cell technique',
            'perturbation study', 'perturbation experiment', 'perturbation analysis', 'perturbation modeling'
        ]


        important_terms = [
            'crispr', 'perturbation', 'single', 'cell', 'gene', 'expression', 'model', 'prediction', 'analysis',
            'transcriptomics', 'genomics', 'proteomics', 'multiomics', 'knockout', 'knockdown', 'overexpression',
            'differential', 'trajectory', 'pseudotime', 'clustering', 'dimensionality', 'regulatory', 'network',
            'pathway', 'enrichment', 'annotation', 'state', 'transition', 'differentiation', 'proliferation',
            'apoptosis', 'cycle', 'signaling', 'transcriptional', 'epigenetic', 'metabolic', 'immune',
            'response', 'stress', 'fate', 'plasticity', 'deep', 'learning', 'neural', 'transformer', 'graph',
            'variational', 'autoencoder', 'generative', 'adversarial', 'recurrent', 'convolutional', 'attention',
            'self-supervised', 'transfer', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'roc', 'mse',
            'r2', 'silhouette', 'rand', 'mutual', 'pearson', 'spearman', 'jaccard', 'high-dimensional',
            'sparse', 'noisy', 'imbalanced', 'batch', 'dropout', 'technical', 'biological', 'temporal',
            'spatial', 'heterogeneity', 'experimental', 'design', 'protocol', 'methodology', 'algorithm',
            'framework', 'pipeline', 'workflow', 'optimization', 'validation', 'benchmark', 'comparison',
            'evaluation', 'assessment', 'measurement', 'quantification', 'characterization', 'profiling',
            'screening', 'discovery', 'identification', 'classification', 'clustering', 'segmentation',
            'reconstruction', 'inference', 'estimation', 'approximation', 'simulation', 'modeling'
        ]


        basic_stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'shall', 'must', 'ought',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }

        query_lower = query.lower()
        extracted_terms = []


        for phrase in sorted(domain_phrases, key=len, reverse=True):
            if phrase in query_lower and phrase not in extracted_terms:
                extracted_terms.append(phrase)

                query_lower = query_lower.replace(phrase, ' ' * len(phrase))


        words = re.findall(r'\b\w+\b', query_lower)


        for word in words:

            if len(word) <= 2 or word in basic_stop_words:
                continue


            if word in important_terms and word not in extracted_terms:
                extracted_terms.append(word)

            elif len(word) >= 4 and word not in extracted_terms:

                if word not in ['very', 'just', 'now', 'then', 'here', 'there', 'when', 'where', 'why', 'how', 'for']:
                    extracted_terms.append(word)


        for term in important_terms:
            if term in query_lower and term not in extracted_terms:
                extracted_terms.append(term)


        if len(extracted_terms) > 20:

            phrases = [term for term in extracted_terms if ' ' in term]
            words = [term for term in extracted_terms if ' ' not in term]


            max_words = 20 - len(phrases)
            extracted_terms = phrases + words[:max_words]

        return extracted_terms

    def _create_sample_data_if_empty(self, qdrant_client, db_name: str) -> bool:
        """Create sample data if database is empty for testing purposes"""
        try:

            collections = qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if "papers" not in collection_names:
                print(f" Creating papers collection in {db_name}...")
                qdrant_client.create_collection(
                    collection_name="papers",
                    vectors_config={
                        "size": 384,
                        "distance": "Cosine"
                    }
                )

            if "code" not in collection_names:
                print(f" Creating code collection in {db_name}...")
                qdrant_client.create_collection(
                    collection_name="code",
                    vectors_config={
                        "size": 384,
                        "distance": "Cosine"
                    }
                )


            try:
                papers_count = qdrant_client.count(collection_name="papers").count
                code_count = qdrant_client.count(collection_name="code").count

                if papers_count == 0 and code_count == 0:
                    print(f" Creating sample data in {db_name}...")
                    self._insert_sample_data(qdrant_client, db_name)
                    return True
            except Exception as e:
                print(f"  ⚠️  Could not check data count: {e}")
                return False

        except Exception as e:
            print(f"  ❌ Failed to create sample data: {e}")
            return False

    def _insert_sample_data(self, qdrant_client, db_name: str):
        """Insert sample data for testing"""
        sample_papers = [
            {
                "title": "Single-cell RNA sequencing reveals cell type-specific responses to CRISPR perturbations",
                "content": "This study demonstrates the use of single-cell RNA sequencing to analyze gene expression changes following CRISPR-mediated gene perturbations in different cell types. The authors developed a computational pipeline for analyzing perturbation effects at single-cell resolution.",
                "url": "https://example.com/paper1",
                "keywords": ["single-cell", "CRISPR", "perturbation", "RNA-seq", "gene expression"]
            },
            {
                "title": "Predictive modeling of gene expression responses to genetic perturbations",
                "content": "We present a deep learning model that predicts gene expression changes in response to genetic perturbations. The model uses attention mechanisms to capture regulatory relationships between genes.",
                "url": "https://example.com/paper2",
                "keywords": ["predictive model", "gene expression", "genetic perturbation", "deep learning", "attention"]
            },
            {
                "title": "Experimental design for single-cell perturbation studies",
                "content": "This paper provides guidelines for designing experiments to study cellular responses to perturbations at single-cell resolution, including considerations for sample size, controls, and analysis methods.",
                "url": "https://example.com/paper3",
                "keywords": ["experimental design", "single-cell", "perturbation", "sample size", "controls"]
            }
        ]

        sample_code = [
            {
                "title": "SingleCellPerturbationAnalysis",
                "content": "Python package for analyzing single-cell perturbation data. Includes tools for preprocessing, differential expression analysis, and visualization of perturbation effects.",
                "url": "https://github.com/example/single-cell-perturbation",
                "keywords": ["python", "single-cell", "perturbation", "analysis", "differential expression"]
            },
            {
                "title": "PerturbNet",
                "content": "Deep learning framework for predicting gene expression changes following genetic perturbations. Uses graph neural networks to model gene regulatory networks.",
                "url": "https://github.com/example/perturbnet",
                "keywords": ["deep learning", "graph neural network", "gene expression", "perturbation", "prediction"]
            }
        ]


        for i, paper in enumerate(sample_papers):
            embedding = self.encoder.encode(f"{paper['title']} {paper['content']}")
            qdrant_client.upsert(
                collection_name="papers",
                points=[{
                    "id": hash(f"sample_paper_{db_name}_{i}") % 1000000,
                    "vector": embedding.tolist(),
                    "payload": {
                        "title": paper["title"],
                        "content": paper["content"],
                        "url": paper["url"],
                        "keywords": paper["keywords"]
                    }
                }]
            )


        for i, code in enumerate(sample_code):
            embedding = self.encoder.encode(f"{code['title']} {code['content']}")
            qdrant_client.upsert(
                collection_name="code",
                points=[{
                    "id": hash(f"sample_code_{db_name}_{i}") % 1000000,
                    "vector": embedding.tolist(),
                    "payload": {
                        "title": code["title"],
                        "content": code["content"],
                        "url": code["url"],
                        "keywords": code["keywords"]
                    }
                }]
            )

        print(f"  ✅ Inserted {len(sample_papers)} sample papers and {len(sample_code)} sample code items in {db_name}")

    def _store_to_tmp_db(self, results: List[SearchResult], original_query: str) -> bool:
        """Store search results to cellforge_tmp database for future reference"""
        if not self.qdrant_tmp:
            return False

        try:

            collection_name = "search_results_cache"
            try:
                self.qdrant_tmp.get_collection(collection_name)
            except:
                self.qdrant_tmp.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "size": 384,
                        "distance": "Cosine"
                    }
                )


            stored_count = 0
            for i, result in enumerate(results):
                try:

                    text_content = f"{result.title} {result.snippet}"
                    embedding = self.encoder.encode(text_content)


                    unique_id = hash(f"{result.title}_{result.url}_{i}") % 1000000


                    self.qdrant_tmp.upsert(
                        collection_name=collection_name,
                        points=[{
                            "id": unique_id,
                            "vector": embedding.tolist(),
                            "payload": {
                                "title": result.title,
                                "url": result.url,
                                "snippet": result.snippet,
                                "source": result.source,
                                "score": result.score,
                                "original_query": original_query,
                                "timestamp": str(datetime.now()),
                                "metadata": result.metadata or {}
                            }
                        }]
                    )
                    stored_count += 1
                except Exception as e:
                    print(f"  ⚠️  Failed to store result {i}: {e}")
                    continue

            print(f" Stored {stored_count}/{len(results)} results to cellforge_tmp")
            return stored_count > 0
        except Exception as e:
            print(f"  ❌ Failed to store to tmp DB: {e}")
            return False

    def _fuse_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Fuse and rank results from multiple sources with context similarity"""
        if not results:
            return []


        query_embedding = self.encoder.encode(query)


        query_keywords = self._extract_keywords(query)


        for result in results:

            base_score = result.score


            source_boost = 1.0
            if "main" in result.source:
                source_boost = 1.2
            elif "tmp" in result.source:
                source_boost = 1.0
            elif "online" in result.source:
                source_boost = 0.8


            content_relevance = 0.0
            query_lower = query.lower()
            title_lower = result.title.lower()
            snippet_lower = result.snippet.lower()


            if query_lower in title_lower:
                content_relevance += 0.3
            if query_lower in snippet_lower:
                content_relevance += 0.2


            context_similarity = 0.0
            for keyword in query_keywords:
                if keyword in title_lower:
                    context_similarity += 0.1
                if keyword in snippet_lower:
                    context_similarity += 0.05


            phrase_boost = 0.0
            for keyword in query_keywords:
                if len(keyword.split()) > 1:
                    if keyword in title_lower:
                        phrase_boost += 0.2
                    if keyword in snippet_lower:
                        phrase_boost += 0.1


            result.score = (base_score * source_boost) + content_relevance + context_similarity + phrase_boost


        results.sort(key=lambda x: x.score, reverse=True)


        unique_results = []
        seen_content = set()

        for result in results:

            content_signature = hash(f"{result.title}_{result.snippet[:200]}")
            if content_signature not in seen_content:
                seen_content.add(content_signature)
                unique_results.append(result)

        return unique_results

    def _sort_by_relevance_and_recency(self, results: List[SearchResult]) -> List[SearchResult]:
        """Sort results by relevance (score) and recency (timestamp if available)"""

        def sort_key(result):

            score = result.score


            timestamp = None
            if result.metadata:
                timestamp_str = result.metadata.get("timestamp", result.metadata.get("publication_date", ""))
                if timestamp_str:
                    try:
                        from datetime import datetime
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except:
                        pass


            if timestamp:

                from datetime import datetime, timedelta
                two_years_ago = datetime.now() - timedelta(days=730)
                if timestamp > two_years_ago:
                    score += 0.1

            return score

        return sorted(results, key=sort_key, reverse=True)

    def search_cached_results(self, query: str, limit: int = 20) -> List[SearchResult]:
        """Search cached results from cellforge_tmp database"""
        if not self.qdrant_tmp:
            return []

        try:

            query_embedding = self.encoder.encode(query)


            cached_results = self.qdrant_tmp.search(
                collection_name="search_results_cache",
                query_vector=query_embedding.tolist(),
                limit=limit,
                with_payload=True
            )

            results = []
            for result in cached_results:
                results.append(SearchResult(
                    title=result.payload.get("title", ""),
                    url=result.payload.get("url", ""),
                    snippet=result.payload.get("snippet", ""),
                    source=result.payload.get("source", "cached"),
                    score=result.score,
                    metadata=result.payload
                ))

            return results
        except Exception as e:
            print(f"❌ Failed to search cached results: {e}")
            return []

    def get_decision_recommendations(self, task_description: str, dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get decision support recommendations based on task and dataset"""
        try:

            query_terms = []
            if task_description:
                query_terms.extend(task_description.split()[:10])
            if dataset_info and 'name' in dataset_info:
                query_terms.append(dataset_info['name'])


            decision_query = " ".join(query_terms) + " single cell perturbation analysis model architecture"


            search_results = self.search(decision_query, limit=5)


            recommendations = []
            for result in search_results:
                recommendations.append({
                    "title": result.title,
                    "content": result.snippet,
                    "source": result.source,
                    "score": result.score,
                    "url": result.url,
                    "metadata": result.metadata or {}
                })

            return recommendations
        except Exception as e:
            self.logger.error(f"Failed to get decision recommendations: {e}")
            return []

    def search_scientific(self, query: str, limit: int = 30) -> List[SearchResult]:
        """Search for scientific literature"""
        if not self.web_searcher:
            return []

        try:
            return self.web_searcher.search_scientific(query, limit=limit)
        except Exception as e:
            self.logger.error(f"Scientific search failed: {e}")
            return []

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics for MCP integration"""
        return {
            "search_strategy": "BFS-DFS Alternating",
            "max_layers": 2,
            "relevance_threshold": 0.3,
            "overlap_threshold": 0.6,
            "databases": ["CellForge", "cellforge_tmp"],
            "collections": ["papers", "code", "CellForge"]
        }

    def get_knowledge_base_info(self, query: str) -> Dict[str, Any]:
        """Get knowledge base information for MCP integration"""
        results = self.search(query, limit=20)
        return {
            "query": query,
            "total_documents": len(results),
            "unique_documents": len(set(r.title for r in results)),
            "average_score": sum(r.score for r in results) / len(results) if results else 0,
            "sources": list(set(r.source for r in results)),
            "top_documents": [
                {
                    "title": r.title,
                    "score": r.score,
                    "source": r.source,
                    "snippet": r.snippet[:200] + "..." if len(r.snippet) > 200 else r.snippet
                }
                for r in sorted(results, key=lambda x: x.score, reverse=True)[:5]
            ]
        }

    def export_knowledge_base(self, query: str, format: str = "json") -> str:
        """Export knowledge base for MCP integration"""
        results = self.search(query, limit=50)

        if format == "json":
            import json
            knowledge_base = {
                "query": query,
                "timestamp": str(datetime.now()),
                "total_documents": len(results),
                "documents": [
                    {
                        "title": r.title,
                        "content": r.snippet,
                        "url": r.url,
                        "score": r.score,
                        "source": r.source,
                        "metadata": r.metadata
                    }
                    for r in results
                ]
            }
            return json.dumps(knowledge_base, indent=2, ensure_ascii=False)

        elif format == "markdown":
            markdown += f"Generated at: {datetime.now()}\n"
            markdown += f"Total documents: {len(results)}\n\n"

            for i, result in enumerate(results, 1):
                markdown += f"**Score:** {result.score:.3f}\n"
                markdown += f"**Source:** {result.source}\n"
                markdown += f"**URL:** {result.url}\n\n"
                markdown += f"{result.snippet}\n\n"
                markdown += "---\n\n"

            return markdown

        else:
            raise ValueError(f"Unsupported format: {format}")

class WebSearcher:
    """Web search using SerpAPI"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.logger = logging.getLogger("cellforge.task_analysis.web_searcher")

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Perform web search"""
        if not SERPAPI_AVAILABLE:
            return []

        try:
            search = GoogleSearch({
                "q": query,
                "api_key": self.api_key,
                "num": limit
            })
            results = search.get_dict()

            search_results = []
            if "organic_results" in results:
                for result in results["organic_results"][:limit]:
                    search_results.append(SearchResult(
                        title=result.get("title", ""),
                        url=result.get("link", ""),
                        snippet=result.get("snippet", ""),
                        source="web",
                        score=1.0,
                        metadata={"position": result.get("position", 0)}
                    ))

            return search_results
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            return []

    def search_scientific(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search for scientific papers"""
        if not SERPAPI_AVAILABLE:
            return []

        try:
            search = GoogleSearch({
                "q": f"{query} site:pubmed.ncbi.nlm.nih.gov OR site:arxiv.org OR site:scholar.google.com",
                "api_key": self.api_key,
                "num": limit
            })
            results = search.get_dict()

            search_results = []
            if "organic_results" in results:
                for result in results["organic_results"][:limit]:
                    search_results.append(SearchResult(
                        title=result.get("title", ""),
                        url=result.get("link", ""),
                        snippet=result.get("snippet", ""),
                        source="scientific",
                        score=1.0,
                        metadata={"position": result.get("position", 0)}
                    ))

            return search_results
        except Exception as e:
            self.logger.error(f"Scientific search failed: {e}")
            return []

class GitHubSearcher:
    """GitHub repository search"""

    def __init__(self, token: str):
        self.token = token
        self.github = Github(token)
        self.logger = logging.getLogger("cellforge.task_analysis.github_searcher")

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search GitHub repositories"""
        if not GITHUB_AVAILABLE:
            return []

        try:
            repositories = self.github.search_repositories(
                query=query,
                sort="stars",
                order="desc"
            )

            search_results = []
            for repo in repositories[:limit]:
                search_results.append(SearchResult(
                    title=repo.name,
                    url=repo.html_url,
                    snippet=repo.description or "",
                    source="github",
                    score=repo.stargazers_count / 1000.0,
                    metadata={
                        "stars": repo.stargazers_count,
                        "language": repo.language,
                        "full_name": repo.full_name
                    }
                ))

            return search_results
        except Exception as e:

            return []

    def search_code(self, query: str, language: str = "python", limit: int = 10) -> List[SearchResult]:
        """Search GitHub code"""
        if not GITHUB_AVAILABLE:
            return []

        try:
            code_results = self.github.search_code(
                query=f"{query} language:{language}",
                sort="indexed",
                order="desc"
            )

            search_results = []
            for code in code_results[:limit]:
                search_results.append(SearchResult(
                    title=code.name,
                    url=code.html_url,
                    snippet=f"File: {code.path} in {code.repository.full_name}",
                    source="github_code",
                    score=1.0,
                    metadata={
                        "repository": code.repository.full_name,
                        "path": code.path,
                        "language": language
                    }
                ))

            return search_results
        except Exception as e:

            return []
