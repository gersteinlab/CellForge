from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import List
import json
import re

from .models import PaperRecord


class LocalCorpusProvider:
    """Search PDFs and metadata in an external, configurable literature directory."""

    name = "local_corpus"

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.papers_dir = self.root / "papers"
        self.cache_dir = self.root / "cache" / "extracted_text"

    def _extract_pdf(self, path: Path, digest: str) -> List[tuple[int, str]]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{digest}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return [(int(item["page"]), item["text"]) for item in data]
        try:
            import fitz
            document = fitz.open(path)
            pages = [(index + 1, page.get_text("text")) for index, page in enumerate(document)]
            cache_file.write_text(
                json.dumps(
                    [{"page": page, "text": text} for page, text in pages],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return pages
        except Exception:
            return []

    @staticmethod
    def _score(query: str, text: str) -> float:
        terms = {
            token for token in re.findall(r"[A-Za-z0-9]+", query.lower())
            if len(token) > 2
        }
        if not terms:
            return 0.0
        haystack = text.lower()
        matched = sum(1 for term in terms if term in haystack)
        return matched / len(terms)

    def search(self, query: str, limit: int) -> List[PaperRecord]:
        if not self.papers_dir.exists():
            return []
        best_by_path = {}
        for path in sorted(self.papers_dir.glob("*.pdf")):
            digest = sha256(path.read_bytes()).hexdigest()
            title_score = self._score(query, path.stem)
            pages = self._extract_pdf(path, digest)
            for page, text in pages:
                score = max(title_score, self._score(query, text))
                if score <= 0:
                    continue
                candidate = PaperRecord(
                        title=path.stem,
                        abstract=text[:4000],
                        source_path=str(path),
                        page=page,
                        source_providers=[self.name],
                        score=score,
                        metadata={"sha256": digest},
                    )
                current = best_by_path.get(path)
                if current is None or candidate.score > current.score:
                    best_by_path[path] = candidate
        candidates: List[PaperRecord] = list(best_by_path.values())
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:limit]
