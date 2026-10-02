import json
from pathlib import Path

from cellforge.retrieval.models import PaperRecord
from cellforge.retrieval.providers import (
    CrossrefProvider,
    PubMedProvider,
    SemanticScholarProvider,
)
from cellforge.retrieval.service import LiteratureRetriever


class FakeResponse:
    def __init__(self, payload=None, content=b"", status_code=200):
        self._payload = payload
        self.content = content
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class QueueSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def test_pubmed_search_parses_identifiers_and_abstract():
    xml = b"""
    <PubmedArticleSet><PubmedArticle>
      <MedlineCitation>
        <PMID>27984730</PMID>
        <Article>
          <Journal><JournalIssue><PubDate><Year>2016</Year></PubDate></JournalIssue>
            <Title>Cell</Title></Journal>
          <ArticleTitle>A multiplexed single-cell CRISPR screening platform</ArticleTitle>
          <Abstract><AbstractText>We developed Perturb-seq.</AbstractText></Abstract>
          <AuthorList><Author><ForeName>Britt</ForeName><LastName>Adamson</LastName></Author></AuthorList>
        </Article>
      </MedlineCitation>
      <PubmedData><ArticleIdList>
        <ArticleId IdType="doi">10.1016/j.cell.2016.11.048</ArticleId>
      </ArticleIdList></PubmedData>
    </PubmedArticle></PubmedArticleSet>
    """
    session = QueueSession(
        [
            FakeResponse({"esearchresult": {"idlist": ["27984730"]}}),
            FakeResponse(content=xml),
        ]
    )
    records = PubMedProvider(session=session, email="test@example.org").search(
        "Adamson Perturb-seq", 5
    )
    assert records[0].pmid == "27984730"
    assert records[0].doi == "10.1016/j.cell.2016.11.048"
    assert "Perturb-seq" in records[0].abstract
    assert session.calls[0][1]["params"]["email"] == "test@example.org"


def test_crossref_and_semantic_scholar_normalize_records():
    crossref_session = QueueSession(
        [FakeResponse({"message": {"items": [{
            "DOI": "10.1016/j.cell.2016.11.048",
            "title": ["A multiplexed single-cell CRISPR screening platform"],
            "author": [{"given": "Britt", "family": "Adamson"}],
            "published": {"date-parts": [[2016]]},
            "container-title": ["Cell"],
            "URL": "https://doi.org/10.1016/j.cell.2016.11.048",
            "is-referenced-by-count": 1000,
        }]}})]
    )
    s2_session = QueueSession(
        [FakeResponse({"data": [{
            "paperId": "s2-paper",
            "title": "A multiplexed single-cell CRISPR screening platform",
            "abstract": "Perturb-seq abstract",
            "authors": [{"name": "Britt Adamson"}],
            "year": 2016,
            "venue": "Cell",
            "url": "https://www.semanticscholar.org/paper/s2-paper",
            "externalIds": {
                "DOI": "10.1016/j.cell.2016.11.048",
                "PubMed": "27984730",
            },
            "citationCount": 1000,
            "openAccessPdf": {"url": "https://example.org/paper.pdf"},
        }]})]
    )
    crossref = CrossrefProvider(session=crossref_session).search("Perturb-seq", 5)[0]
    semantic = SemanticScholarProvider(session=s2_session).search("Perturb-seq", 5)[0]
    assert crossref.paper_id == semantic.paper_id
    assert semantic.pmid == "27984730"
    assert semantic.open_access_pdf_url == "https://example.org/paper.pdf"


class StaticProvider:
    def __init__(self, name, records=None, error=None):
        self.name = name
        self.records = records or []
        self.error = error
        self.calls = 0

    def search(self, query, limit):
        self.calls += 1
        if self.error:
            raise self.error
        return self.records[:limit]


def test_retriever_fuses_providers_and_writes_trace(tmp_path):
    doi = "10.1016/j.cell.2016.11.048"
    retriever = LiteratureRetriever(
        literature_dir=tmp_path / "corpus",
        trace_dir=tmp_path / "trace",
        providers=[
            StaticProvider("pubmed", [
                PaperRecord(
                    title="Perturb-seq",
                    abstract="PubMed abstract",
                    doi=doi,
                    pmid="27984730",
                    source_providers=["pubmed"],
                )
            ]),
            StaticProvider("crossref", [
                PaperRecord(
                    title="Perturb-seq",
                    doi=doi,
                    journal="Cell",
                    source_providers=["crossref"],
                )
            ]),
            StaticProvider("offline", error=RuntimeError("unavailable")),
        ],
    )
    results = retriever.search("Adamson Perturb-seq", limit=5)
    assert len(results) == 1
    assert set(results[0]["metadata"]["source_providers"]) == {"crossref", "pubmed"}
    assert results[0]["metadata"]["pmid"] == "27984730"
    trace = json.loads(
        (tmp_path / "trace" / "retrieval_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert trace["providers"]["offline"]["status"] == "failed"
    assert trace["results"][0]["paper_id"] == f"doi:{doi}"
    assert retriever.last_trace is not None
    assert retriever.last_trace.query == "Adamson Perturb-seq"


def test_failed_provider_is_disabled_for_remainder_of_run(tmp_path):
    failed = StaticProvider("rate_limited", error=RuntimeError("HTTP 429"))
    healthy = StaticProvider(
        "healthy",
        [PaperRecord(title="Evidence", source_providers=["healthy"])],
    )
    retriever = LiteratureRetriever(tmp_path, providers=[failed, healthy])
    retriever.search("first query", limit=2)
    retriever.search("second query", limit=2)
    assert failed.calls == 1
    assert healthy.calls == 2
    assert retriever.last_trace.providers["rate_limited"]["status"] == "disabled"


def test_external_corpus_path_is_not_repo_coupled(tmp_path, monkeypatch):
    external = tmp_path / "shared-literature"
    monkeypatch.setenv("CELLFORGE_LITERATURE_DIR", str(external))
    monkeypatch.setenv("CELLFORGE_ONLINE_RETRIEVAL", "false")
    retriever = LiteratureRetriever.from_env()
    assert retriever.literature_dir == external.resolve()
    assert retriever.providers[0].papers_dir == external.resolve() / "papers"


def test_task_query_uses_dataset_author_and_assay(tmp_path):
    provider = StaticProvider("fixture", [
        PaperRecord(title="Adamson Perturb-seq", source_providers=["fixture"])
    ])
    retriever = LiteratureRetriever(tmp_path, providers=[provider])
    retriever.search_for_task(
        "A long task description with many evaluation constraints",
        {
            "dataset_name": "AdamsonWeissman2016_GSM2406675_10X001",
            "perturbation_type": "CRISPRi",
            "data_type": "scRNA-seq",
        },
    )
    assert retriever.last_trace.query == "Adamson Perturb-seq"
