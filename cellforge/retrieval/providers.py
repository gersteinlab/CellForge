from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree
import re

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import PaperRecord


def _text(element) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _year(value: Any) -> Optional[int]:
    match = re.search(r"(19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


class LiteratureProvider(ABC):
    name = "provider"

    def __init__(self, timeout: float = 15.0, session=None):
        self.timeout = timeout
        if session is not None:
            self.session = session
        else:
            self.session = requests.Session()
            retry = Retry(
                total=3,
                connect=3,
                read=3,
                status=3,
                backoff_factor=1.0,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset({"GET"}),
                respect_retry_after_header=True,
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)

    @abstractmethod
    def search(self, query: str, limit: int) -> List[PaperRecord]:
        raise NotImplementedError


class PubMedProvider(LiteratureProvider):
    name = "pubmed"
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(
        self,
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        tool: str = "cellforge",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.email = email
        self.tool = tool

    def _common_params(self) -> Dict[str, str]:
        params = {"tool": self.tool}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def search(self, query: str, limit: int) -> List[PaperRecord]:
        params = {
            **self._common_params(),
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(limit),
            "sort": "relevance",
        }
        response = self.session.get(
            f"{self.base_url}/esearch.fcgi", params=params, timeout=self.timeout
        )
        response.raise_for_status()
        identifiers = response.json().get("esearchresult", {}).get("idlist", [])
        if not identifiers:
            return []

        fetch_params = {
            **self._common_params(),
            "db": "pubmed",
            "id": ",".join(identifiers),
            "retmode": "xml",
        }
        response = self.session.get(
            f"{self.base_url}/efetch.fcgi",
            params=fetch_params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        records: List[PaperRecord] = []
        for rank, article in enumerate(root.findall(".//PubmedArticle")):
            medline = article.find("./MedlineCitation")
            journal = article.find(".//Journal")
            article_node = article.find(".//Article")
            if medline is None or article_node is None:
                continue
            pmid = _text(medline.find("./PMID"))
            title = _text(article_node.find("./ArticleTitle"))
            abstract = " ".join(
                _text(node) for node in article_node.findall(".//AbstractText")
            ).strip()
            authors = []
            for author in article_node.findall(".//Author"):
                name = " ".join(
                    part for part in (
                        _text(author.find("./ForeName")),
                        _text(author.find("./LastName")),
                    ) if part
                )
                if name:
                    authors.append(name)
            doi = None
            for identifier in article.findall(".//ArticleId"):
                if identifier.attrib.get("IdType") == "doi":
                    doi = _text(identifier)
                    break
            pub_date = journal.find("./JournalIssue/PubDate") if journal is not None else None
            records.append(
                PaperRecord(
                    title=title or f"PubMed {pmid}",
                    abstract=abstract,
                    authors=authors,
                    year=_year(_text(pub_date)),
                    journal=_text(journal.find("./Title")) if journal is not None else None,
                    doi=doi,
                    pmid=pmid,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                    source_providers=[self.name],
                    score=1.0 / (rank + 1),
                )
            )
        return records


class CrossrefProvider(LiteratureProvider):
    name = "crossref"
    base_url = "https://api.crossref.org/v1"

    def __init__(self, email: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.email = email
        self.session.headers.update(
            {"User-Agent": f"CellForge/0.1 (mailto:{email or 'unknown'})"}
        )

    def search(self, query: str, limit: int) -> List[PaperRecord]:
        params = {
            "query.bibliographic": query,
            "rows": str(limit),
            "select": (
                "DOI,title,author,published,container-title,URL,abstract,"
                "is-referenced-by-count"
            ),
        }
        if self.email:
            params["mailto"] = self.email
        response = self.session.get(
            f"{self.base_url}/works", params=params, timeout=self.timeout
        )
        response.raise_for_status()
        items = response.json().get("message", {}).get("items", [])
        records = []
        for rank, item in enumerate(items):
            title = (item.get("title") or [""])[0]
            authors = [
                " ".join(filter(None, [a.get("given"), a.get("family")]))
                for a in item.get("author", [])
            ]
            date_parts = (item.get("published") or {}).get("date-parts") or [[]]
            year = date_parts[0][0] if date_parts and date_parts[0] else None
            records.append(
                PaperRecord(
                    title=title,
                    abstract=re.sub(r"<[^>]+>", " ", item.get("abstract", "")).strip(),
                    authors=authors,
                    year=year,
                    journal=(item.get("container-title") or [None])[0],
                    doi=item.get("DOI"),
                    url=item.get("URL"),
                    citation_count=item.get("is-referenced-by-count"),
                    source_providers=[self.name],
                    score=1.0 / (rank + 1),
                )
            )
        return records


class SemanticScholarProvider(LiteratureProvider):
    name = "semantic_scholar"
    base_url = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if api_key:
            self.session.headers.update({"x-api-key": api_key})

    def search(self, query: str, limit: int) -> List[PaperRecord]:
        fields = (
            "title,abstract,authors,year,venue,url,externalIds,citationCount,"
            "openAccessPdf"
        )
        response = self.session.get(
            f"{self.base_url}/paper/search",
            params={"query": query.replace("-", " "), "limit": str(limit), "fields": fields},
            timeout=self.timeout,
        )
        response.raise_for_status()
        records = []
        for rank, item in enumerate(response.json().get("data", [])):
            external = item.get("externalIds") or {}
            open_pdf = item.get("openAccessPdf") or {}
            records.append(
                PaperRecord(
                    title=item.get("title") or "",
                    abstract=item.get("abstract") or "",
                    authors=[a.get("name", "") for a in item.get("authors", []) if a.get("name")],
                    year=item.get("year"),
                    journal=item.get("venue") or None,
                    doi=external.get("DOI"),
                    pmid=external.get("PubMed"),
                    semantic_scholar_id=item.get("paperId"),
                    url=item.get("url"),
                    open_access_pdf_url=open_pdf.get("url"),
                    citation_count=item.get("citationCount"),
                    source_providers=[self.name],
                    score=1.0 / (rank + 1),
                )
            )
        return records
