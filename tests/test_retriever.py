"""
Unit tests for Retrievers (PubMed, Semantic Scholar, Mock) and Workflow Configuration
"""

from unittest.mock import patch, MagicMock
from src.retriever.mock_retriever import MockPubMedRetriever
from src.retriever.pubmed import PubMedRetriever
from src.retriever.semanticscholar import SemanticScholarRetriever
from src.graph.workflow import build_medical_research_graph


def test_mock_retriever_returns_papers():
    retriever = MockPubMedRetriever()
    papers = retriever.search("SGLT2 heart failure", max_results=3)
    assert len(papers) > 0
    assert any("36049247" in p.pmid or "36049248" in p.pmid for p in papers)


def test_semantic_scholar_retriever_fallback():
    # Tests that when S2 API hits error or rate-limit, it returns papers with citation count
    retriever = SemanticScholarRetriever()
    papers = retriever.search("SGLT2 inhibitors HFpEF", max_results=3)
    assert len(papers) > 0
    first = papers[0]
    assert hasattr(first, "citation_count")
    assert hasattr(first, "open_access_pdf")
    assert first.citation_count >= 0


def test_semantic_scholar_retriever_parsing():
    retriever = SemanticScholarRetriever()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {
                "title": "Novel trial of SGLT-2 inhibitor",
                "abstract": "Double-blind clinical study.",
                "venue": "NEJM",
                "year": 2023,
                "authors": [{"name": "Smith J"}],
                "publicationTypes": ["Clinical Trial"],
                "externalIds": {"PubMed": "98765432", "DOI": "10.1056/fake"},
                "citationCount": 540,
                "openAccessPdf": {"url": "https://example.com/paper.pdf"},
            }
        ]
    }
    with patch("requests.get", return_value=mock_resp):
        papers = retriever.search("test query", max_results=1)
        assert len(papers) == 1
        assert papers[0].pmid == "98765432"
        assert papers[0].citation_count == 540
        assert papers[0].open_access_pdf == "https://example.com/paper.pdf"


def test_workflow_with_configurable_retrievers():
    app = build_medical_research_graph()

    for source in ["mock", "semantic_scholar", "pubmed"]:
        state = {
            "query": "SGLT-2抑制剂治疗心衰",
            "retriever_source": source,
            "pico": None,
            "search_terms": [],
            "retrieved_papers": [],
            "ranked_papers": [],
            "annotated_papers": [],
            "draft_review": "",
            "claims": [],
            "verification_report": None,
            "final_review": "",
            "iteration_count": 0,
            "has_hallucination": False,
            "error": None,
        }
        res = app.invoke(state)
        assert len(res["retrieved_papers"]) > 0
        assert len(res["final_review"]) > 50
