from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, List, Optional
import re


def _clean_identifier(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value.strip().lower()


@dataclass
class PaperRecord:
    title: str
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    url: Optional[str] = None
    open_access_pdf_url: Optional[str] = None
    citation_count: Optional[int] = None
    source_providers: List[str] = field(default_factory=list)
    source_path: Optional[str] = None
    page: Optional[int] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def paper_id(self) -> str:
        doi = _clean_identifier(self.doi)
        if doi:
            return f"doi:{doi}"
        if self.pmid:
            return f"pmid:{self.pmid}"
        if self.semantic_scholar_id:
            return f"s2:{self.semantic_scholar_id}"
        normalized = re.sub(r"\W+", " ", self.title.lower()).strip()
        digest = sha256(f"{normalized}|{self.year or ''}".encode("utf-8")).hexdigest()[:16]
        return f"title:{digest}"

    @property
    def evidence_id(self) -> str:
        page = f":p{self.page}" if self.page is not None else ""
        return f"cf:{self.paper_id}{page}"

    def merge(self, other: "PaperRecord") -> "PaperRecord":
        """Merge another provider's metadata into this record."""
        for name in (
            "abstract", "journal", "doi", "pmid", "semantic_scholar_id", "url",
            "open_access_pdf_url", "citation_count", "source_path", "page",
        ):
            current = getattr(self, name)
            candidate = getattr(other, name)
            if (current is None or current == "") and candidate not in (None, ""):
                setattr(self, name, candidate)
        if not self.authors and other.authors:
            self.authors = list(other.authors)
        if self.year is None:
            self.year = other.year
        self.source_providers = sorted(set(self.source_providers + other.source_providers))
        self.metadata.update(other.metadata)
        self.score = max(self.score, other.score)
        return self

    def to_result(self) -> Dict[str, Any]:
        content = self.abstract.strip() or self.title
        return {
            "title": self.title,
            "content": content,
            "url": self.url or self.source_path or "",
            "score": self.score,
            "source": ",".join(self.source_providers) or "unknown",
            "relevance_score": self.score,
            "evidence_id": self.evidence_id,
            "paper_id": self.paper_id,
            "metadata": {
                **self.metadata,
                "doi": self.doi,
                "pmid": self.pmid,
                "semantic_scholar_id": self.semantic_scholar_id,
                "authors": self.authors,
                "year": self.year,
                "journal": self.journal,
                "source_path": self.source_path,
                "page": self.page,
                "source_providers": self.source_providers,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "paper_id": self.paper_id, "evidence_id": self.evidence_id}


@dataclass
class RetrievalTrace:
    query: str
    requested_limit: int
    providers: Dict[str, Dict[str, Any]]
    results: List[Dict[str, Any]]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
