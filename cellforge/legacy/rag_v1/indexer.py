"""Legacy dual-Qdrant vector indexing utilities."""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("⚠️  Qdrant client not available. Vector indexing will be disabled.")

class VectorIndexer:
    """Vector indexing for task analysis results with dual Qdrant databases"""

    def __init__(self):
        # Load configuration
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

        # Initialize dual Qdrant clients
        self.qdrant_main = None
        self.qdrant_tmp = None
        self.logger = logging.getLogger("cellforge.task_analysis.indexer")
        self.qdrant_enabled = os.getenv("QDRANT_ENABLED", "false").lower() in {"1", "true", "yes"}

        if QDRANT_AVAILABLE and self.qdrant_enabled:
            try:
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
                self.logger.info("Successfully connected to dual Qdrant databases")
            except Exception as e:
                self.logger.warning(f"Could not connect to Qdrant databases: {e}")
                self.qdrant_main = None
                self.qdrant_tmp = None
        elif not self.qdrant_enabled:
            self.logger.info("Qdrant disabled by QDRANT_ENABLED=false; vector indexing persistence is off.")

    def create_collection(self, collection_name: str, vector_size: int = 384) -> bool:
        """Create a new collection in both Qdrant databases"""
        if not self.qdrant_main or not self.qdrant_tmp:
            self.logger.warning("Qdrant clients not available")
            return False

        success = True

        try:
            # Create in main database
            self.qdrant_main.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            self.logger.info(f"Created collection in main DB: {collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to create collection {collection_name} in main DB: {e}")
            success = False

        try:
            # Create in tmp database
            self.qdrant_tmp.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            self.logger.info(f"Created collection in tmp DB: {collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to create collection {collection_name} in tmp DB: {e}")
            success = False

        return success

    def index_data(self, collection_name: str, data: List[Dict[str, Any]],
                   use_main_db: bool = True, use_tmp_db: bool = True) -> bool:
        """
        Index data to Qdrant databases

        Args:
            collection_name: Name of the collection
            data: List of data dictionaries to index
            use_main_db: Whether to index to main database
            use_tmp_db: Whether to index to tmp database

        Returns:
            True if successful, False otherwise
        """
        if not data:
            self.logger.warning("No data to index")
            return False

        success = True

        if use_main_db and self.qdrant_main:
            try:
                self._index_to_database(collection_name, data, self.qdrant_main, "main")
            except Exception as e:
                self.logger.error(f"Failed to index to main DB: {e}")
                success = False

        if use_tmp_db and self.qdrant_tmp:
            try:
                self._index_to_database(collection_name, data, self.qdrant_tmp, "tmp")
            except Exception as e:
                self.logger.error(f"Failed to index to tmp DB: {e}")
                success = False

        return success

    def _index_to_database(self, collection_name: str, data: List[Dict[str, Any]],
                          qdrant_client: QdrantClient, db_name: str):
        """Index data to a specific Qdrant database"""
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer('all-MiniLM-L6-v2')

            points = []
            for i, item in enumerate(data):
                # Encode content
                content = item.get('content', str(item))
                embedding = encoder.encode(content)

                # Create point
                point = PointStruct(
                    id=i,
                    vector=embedding.tolist(),
                    payload={
                        "content": content,
                        "metadata": item.get('metadata', {}),
                        "timestamp": item.get('timestamp', ''),
                        "source": item.get('source', 'task_analysis')
                    }
                )
                points.append(point)

            # Upsert to database
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )

            self.logger.info(f"Successfully indexed {len(points)} items to {db_name} DB collection: {collection_name}")

        except Exception as e:
            self.logger.error(f"Error indexing to {db_name} DB: {e}")
            raise

    def search(self, collection_name: str, query: str, limit: int = 10,
               use_main_db: bool = True) -> List[Dict[str, Any]]:
        """
        Search in Qdrant database

        Args:
            collection_name: Name of the collection to search
            query: Search query
            limit: Maximum number of results
            use_main_db: Whether to use main database (True) or tmp database (False)

        Returns:
            List of search results
        """
        qdrant_client = self.qdrant_main if use_main_db else self.qdrant_tmp
        db_name = "main" if use_main_db else "tmp"

        if not qdrant_client:
            self.logger.warning(f"Qdrant {db_name} client not available")
            return []

        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer('all-MiniLM-L6-v2')

            # Encode query
            query_embedding = encoder.encode(query)

            # Search
            results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit
            )

            # Convert to standard format
            search_results = []
            for result in results:
                search_results.append({
                    "content": result.payload.get("content", ""),
                    "metadata": result.payload.get("metadata", {}),
                    "score": result.score,
                    "source": f"{db_name}_db"
                })

            return search_results

        except Exception as e:
            self.logger.error(f"Error searching in {db_name} database: {e}")
            return []

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection from both databases

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with collection information
        """
        info = {}

        if self.qdrant_main:
            try:
                main_info = self.qdrant_main.get_collection(collection_name)
                info["main_db"] = {
                    "name": main_info.name,
                    "vectors_count": main_info.vectors_count,
                    "points_count": main_info.points_count
                }
            except Exception as e:
                info["main_db"] = {"error": str(e)}
        else:
            info["main_db"] = {"error": "Client not available"}

        if self.qdrant_tmp:
            try:
                tmp_info = self.qdrant_tmp.get_collection(collection_name)
                info["tmp_db"] = {
                    "name": tmp_info.name,
                    "vectors_count": tmp_info.vectors_count,
                    "points_count": tmp_info.points_count
                }
            except Exception as e:
                info["tmp_db"] = {"error": str(e)}
        else:
            info["tmp_db"] = {"error": "Client not available"}

        return info
