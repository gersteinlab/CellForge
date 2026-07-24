# Legacy RAG v1

This directory contains the retired Qdrant/SerpAPI/GitHub retrieval stack:

- `rag.py`: orchestration facade
- `search.py`: hybrid web/vector search
- `indexer.py`: dual-Qdrant indexing
- `dataparser.py`: dataset parsing used by that facade
- `utils.py`: text processing helpers

It is not imported by the active Task Analysis or Method Design pipeline and
receives no compatibility guarantees. New code must use
`cellforge.retrieval.LiteratureRetriever`.
