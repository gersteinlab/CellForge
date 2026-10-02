"""
MCP-compatible Knowledge Base for Task Analysis
Stores and retrieves knowledge from dual Qdrant databases without repeated searches
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from ..paths import config_path

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from sentence_transformers import SentenceTransformer
    QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = Any
    models = None
    QDRANT_AVAILABLE = False
    print("⚠️  Qdrant client not available. Knowledge base will use local storage.")

@dataclass
class KnowledgeItem:
    """Knowledge item with metadata"""
    content: Dict[str, Any]
    source: str
    relevance_score: float
    timestamp: datetime
    metadata: Dict[str, Any]
    knowledge_type: str
    database: str

class MCPKnowledgeBase:
    """
    MCP-compatible Knowledge Base that stores and retrieves knowledge
    from dual Qdrant databases without repeated searches. Compatible with various LLM models.
    """

    def __init__(self):
        self.knowledge_store: Dict[str, List[KnowledgeItem]] = {}
        self.encoder = None
        self.qdrant_cellforge = None
        self.qdrant_tmp = None
        self._db_disabled = {"CellForge": False, "cellforge_tmp": False}
        self.qdrant_enabled = os.getenv("QDRANT_ENABLED", "false").lower() in {"1", "true", "yes"}


        try:
            with config_path().open('r', encoding='utf-8') as f:
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


            if QDRANT_AVAILABLE:
                try:
                    self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                    print("✅ Sentence transformer encoder initialized")
                except Exception as e:
                    print(f"⚠️  Failed to initialize encoder: {e}")


            if QDRANT_AVAILABLE and self.qdrant_enabled:
                try:
                    main_cfg = _db_cfg(config, "CellForge")
                    tmp_cfg = _db_cfg(config, "cellforge_tmp")

                    self.qdrant_cellforge = QdrantClient(
                        url=main_cfg.get("url", "http://localhost:6333"),
                        api_key=main_cfg.get("api_key")
                    )
                    print("✅ CellForge Qdrant client initialized")


                    self.qdrant_tmp = QdrantClient(
                        url=tmp_cfg.get("url", "http://localhost:6333"),
                        api_key=tmp_cfg.get("api_key")
                    )
                    print("✅ cellforge_tmp Qdrant client initialized")

                except Exception as e:
                    print(f"⚠️  Failed to initialize Qdrant clients: {e}")
            elif not self.qdrant_enabled:
                print("ℹ️  Qdrant disabled by QDRANT_ENABLED=false; using local in-memory knowledge cache.")


            self._initialize_collections()

        except Exception as e:
            print(f"⚠️  Failed to load config: {e}")

    def _initialize_collections(self):
        """Initialize collections for different knowledge types in both databases"""
        if not self.qdrant_cellforge or not self.qdrant_tmp:
            return

        collections = [
            'task_analysis_papers',
            'task_analysis_experimental_designs',
            'task_analysis_implementation_guides',
            'task_analysis_evaluation_frameworks',
            'task_analysis_decision_support'
        ]


        for collection in collections:
            try:
                self.qdrant_cellforge.get_collection(collection)
                print(f"✅ Collection {collection} exists in CellForge")
            except Exception:
                try:
                    self.qdrant_cellforge.create_collection(
                        collection_name=collection,
                        vectors_config=models.VectorParams(
                            size=384,
                            distance=models.Distance.COSINE
                        )
                    )
                    print(f"✅ Created collection {collection} in CellForge")
                except Exception as e:
                    err = str(e).lower()
                    if "connection refused" in err or "[errno 111]" in err:
                        if not self._db_disabled.get("CellForge", False):
                            print("⚠️  CellForge unreachable; disabling DB writes/search for this run.")
                        self._db_disabled["CellForge"] = True
                        self.qdrant_cellforge = None
                        break
                    print(f"⚠️  Failed to create collection {collection} in CellForge: {e}")


        for collection in collections:
            try:
                self.qdrant_tmp.get_collection(collection)
                print(f"✅ Collection {collection} exists in cellforge_tmp")
            except Exception:
                try:
                    self.qdrant_tmp.create_collection(
                        collection_name=collection,
                        vectors_config=models.VectorParams(
                            size=384,
                            distance=models.Distance.COSINE
                        )
                    )
                    print(f"✅ Created collection {collection} in cellforge_tmp")
                except Exception as e:
                    err = str(e).lower()
                    if "connection refused" in err or "[errno 111]" in err:
                        if not self._db_disabled.get("cellforge_tmp", False):
                            print("⚠️  cellforge_tmp unreachable; disabling DB writes/search for this run.")
                        self._db_disabled["cellforge_tmp"] = True
                        self.qdrant_tmp = None
                        break
                    print(f"⚠️  Failed to create collection {collection} in cellforge_tmp: {e}")

    def store_knowledge(self, knowledge_type: str, content: Dict[str, Any],
                       source: str = "unknown", relevance_score: float = 1.0,
                       metadata: Optional[Dict[str, Any]] = None,
                       use_main_db: bool = True) -> bool:
        """
        Store knowledge in the knowledge base

        Args:
            knowledge_type: Type of knowledge ('papers', 'experimental_designs', etc.)
            content: Knowledge content
            source: Source of the knowledge
            relevance_score: Relevance score (0.0 to 1.0)
            metadata: Additional metadata
            use_main_db: Whether to use CellForge (True) or cellforge_tmp (False)

        Returns:
            True if successful, False otherwise
        """
        try:

            item = KnowledgeItem(
                content=content,
                source=source,
                relevance_score=relevance_score,
                timestamp=datetime.now(),
                metadata=metadata or {},
                knowledge_type=knowledge_type,
                database="CellForge" if use_main_db else "cellforge_tmp"
            )


            if knowledge_type not in self.knowledge_store:
                self.knowledge_store[knowledge_type] = []
            self.knowledge_store[knowledge_type].append(item)


            persisted = False
            if self.encoder:
                if use_main_db and self.qdrant_cellforge:
                    persisted = self._store_in_qdrant(item, self.qdrant_cellforge, "CellForge")
                elif not use_main_db and self.qdrant_tmp:
                    persisted = self._store_in_qdrant(item, self.qdrant_tmp, "cellforge_tmp")

            db_label = 'CellForge' if use_main_db else 'cellforge_tmp'
            if persisted:
                print(f"✅ Stored {knowledge_type} knowledge from {source} in {db_label}")
            else:
                print(f"✅ Cached {knowledge_type} knowledge from {source} locally (DB unavailable: {db_label})")
            return True

        except Exception as e:
            print(f"❌ Failed to store knowledge: {e}")
            return False

    def _store_in_qdrant(self, item: KnowledgeItem, qdrant_client: QdrantClient, db_name: str) -> bool:
        """Store knowledge item in specified Qdrant database"""
        try:
            collection_name = f"task_analysis_{item.knowledge_type}"


            content_str = json.dumps(item.content, ensure_ascii=False)
            embedding = self.encoder.encode(content_str)


            point = models.PointStruct(
                id=hash(content_str),
                vector=embedding.tolist(),
                payload={
                    "content": item.content,
                    "source": item.source,
                    "relevance_score": item.relevance_score,
                    "timestamp": item.timestamp.isoformat(),
                    "metadata": item.metadata,
                    "knowledge_type": item.knowledge_type,
                    "database": db_name
                }
            )


            qdrant_client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            return True

        except Exception as e:
            err = str(e).lower()
            if "connection refused" in err or "[errno 111]" in err:
                if not self._db_disabled.get(db_name, False):
                    print(f"⚠️  {db_name} unreachable; disabling DB writes/search for this run.")
                self._db_disabled[db_name] = True
                if db_name == "CellForge":
                    self.qdrant_cellforge = None
                else:
                    self.qdrant_tmp = None
                return False
            print(f"⚠️  Failed to store in {db_name}: {e}")
            return False

    def _qdrant_search(self, qdrant_client, collection_name: str, query_vector: List[float], limit: int):
        """Compatibility wrapper for different qdrant-client search APIs."""
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

    def retrieve_knowledge(self, knowledge_type: str, query: str = "",
                          limit: int = 10, min_relevance: float = 0.5,
                          use_main_db: bool = True) -> List[KnowledgeItem]:
        """
        Retrieve knowledge from the knowledge base

        Args:
            knowledge_type: Type of knowledge to retrieve
            query: Search query (optional, for semantic search)
            limit: Maximum number of results
            min_relevance: Minimum relevance score
            use_main_db: Whether to search in CellForge (True) or cellforge_tmp (False)

        Returns:
            List of knowledge items
        """
        try:

            items = self.knowledge_store.get(knowledge_type, [])


            items = [item for item in items if item.relevance_score >= min_relevance and
                    item.database == ("CellForge" if use_main_db else "cellforge_tmp")]


            if query and self.encoder:
                semantic_items = self._semantic_search(knowledge_type, query, limit, use_main_db)
                if semantic_items:
                    items = semantic_items


            items.sort(key=lambda x: x.relevance_score, reverse=True)
            return items[:limit]

        except Exception as e:
            print(f"❌ Failed to retrieve knowledge: {e}")
            return []

    def _semantic_search(self, knowledge_type: str, query: str, limit: int, use_main_db: bool) -> List[KnowledgeItem]:
        """Perform semantic search in specified Qdrant database"""
        try:
            collection_name = f"task_analysis_{knowledge_type}"
            qdrant_client = self.qdrant_cellforge if use_main_db else self.qdrant_tmp
            db_name = "CellForge" if use_main_db else "cellforge_tmp"

            if self._db_disabled.get(db_name, False):
                return []
            if not qdrant_client:
                return []


            query_embedding = self.encoder.encode(query)


            results = self._qdrant_search(
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit
            )


            items = []
            for result in results:
                payload = getattr(result, "payload", None)
                if payload is None and isinstance(result, dict):
                    payload = result.get("payload", {})
                if payload is None:
                    payload = {}
                item = KnowledgeItem(
                    content=payload.get("content", {}),
                    source=payload.get("source", "unknown"),
                    relevance_score=getattr(result, "score", 0.0),
                    timestamp=datetime.fromisoformat(payload.get("timestamp", datetime.now().isoformat())),
                    metadata=payload.get("metadata", {}),
                    knowledge_type=payload.get("knowledge_type", knowledge_type),
                    database=db_name
                )
                items.append(item)

            return items

        except Exception as e:
            err_text = str(e).lower()
            if "connection refused" in err_text or "[errno 111]" in err_text:
                if not self._db_disabled.get(db_name, False):
                    print(f"⚠️  {db_name} unreachable; disabling semantic search for this run.")
                self._db_disabled[db_name] = True
                if use_main_db:
                    self.qdrant_cellforge = None
                else:
                    self.qdrant_tmp = None
                return []
            print(f"⚠️  Semantic search failed in {db_name}: {e}")
            return []

    def search_both_databases(self, knowledge_type: str, query: str = "",
                             limit: int = 10, min_relevance: float = 0.5) -> List[KnowledgeItem]:
        """
        Search in both databases and combine results

        Args:
            knowledge_type: Type of knowledge to retrieve
            query: Search query
            limit: Maximum number of results per database
            min_relevance: Minimum relevance score

        Returns:
            Combined list of knowledge items from both databases
        """
        try:

            cellforge_items = self.retrieve_knowledge(
                knowledge_type, query, limit, min_relevance, use_main_db=True
            )


            tmp_items = self.retrieve_knowledge(
                knowledge_type, query, limit, min_relevance, use_main_db=False
            )


            all_items = cellforge_items + tmp_items
            all_items.sort(key=lambda x: x.relevance_score, reverse=True)

            return all_items[:limit * 2]

        except Exception as e:
            print(f"❌ Failed to search both databases: {e}")
            return []

    def get_knowledge_summary(self, knowledge_type: str = None) -> Dict[str, Any]:
        """
        Get summary of stored knowledge

        Args:
            knowledge_type: Specific knowledge type (optional)

        Returns:
            Summary dictionary
        """
        try:
            if knowledge_type:
                items = self.knowledge_store.get(knowledge_type, [])
                return {
                    "knowledge_type": knowledge_type,
                    "count": len(items),
                    "sources": list(set(item.source for item in items)),
                    "databases": list(set(item.database for item in items)),
                    "avg_relevance": sum(item.relevance_score for item in items) / len(items) if items else 0.0,
                    "latest_timestamp": max(item.timestamp for item in items).isoformat() if items else None
                }
            else:
                summary = {}
                for kt, items in self.knowledge_store.items():
                    summary[kt] = {
                        "count": len(items),
                        "sources": list(set(item.source for item in items)),
                        "databases": list(set(item.database for item in items)),
                        "avg_relevance": sum(item.relevance_score for item in items) / len(items) if items else 0.0
                    }
                return summary

        except Exception as e:
            print(f"❌ Failed to get knowledge summary: {e}")
            return {}

    def export_for_mcp(self, knowledge_type: str = None) -> str:
        """
        Export knowledge in MCP-compatible format

        Args:
            knowledge_type: Specific knowledge type (optional)

        Returns:
            MCP-compatible JSON string
        """
        try:
            if knowledge_type:
                items = self.knowledge_store.get(knowledge_type, [])
            else:
                items = []
                for kt_items in self.knowledge_store.values():
                    items.extend(kt_items)


            mcp_data = {
                "knowledge_base": {
                    "timestamp": datetime.now().isoformat(),
                    "total_items": len(items),
                    "knowledge_types": list(self.knowledge_store.keys()),
                    "databases": ["CellForge", "cellforge_tmp"],
                    "items": [
                        {
                            "content": item.content,
                            "source": item.source,
                            "relevance_score": item.relevance_score,
                            "knowledge_type": item.knowledge_type,
                            "database": item.database,
                            "metadata": item.metadata
                        }
                        for item in items
                    ]
                }
            }

            return json.dumps(mcp_data, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"❌ Failed to export for MCP: {e}")
            return "{}"

    def clear_knowledge(self, knowledge_type: str = None, database: str = None):
        """
        Clear knowledge from the knowledge base

        Args:
            knowledge_type: Specific knowledge type to clear (optional, clears all if None)
            database: Specific database to clear (optional, clears all if None)
        """
        try:
            if knowledge_type:
                if knowledge_type in self.knowledge_store:
                    if database:
                        self.knowledge_store[knowledge_type] = [
                            item for item in self.knowledge_store[knowledge_type]
                            if item.database != database
                        ]
                    else:
                        del self.knowledge_store[knowledge_type]
                    print(f"✅ Cleared {knowledge_type} knowledge" + (f" from {database}" if database else ""))
            else:
                if database:
                    for kt in self.knowledge_store:
                        self.knowledge_store[kt] = [
                            item for item in self.knowledge_store[kt]
                            if item.database != database
                        ]
                    print(f"✅ Cleared all knowledge from {database}")
                else:
                    self.knowledge_store.clear()
                    print("✅ Cleared all knowledge")

        except Exception as e:
            print(f"❌ Failed to clear knowledge: {e}")


knowledge_base = MCPKnowledgeBase()
