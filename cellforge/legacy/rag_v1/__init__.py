"""Legacy first-generation Qdrant/SerpAPI retrieval stack.

The active pipeline uses :mod:`cellforge.retrieval`. This package is retained
only to make the historical implementation discoverable during migration.
"""

import warnings

warnings.warn(
    "cellforge.legacy.rag_v1 is unsupported; use cellforge.retrieval",
    DeprecationWarning,
    stacklevel=2,
)
