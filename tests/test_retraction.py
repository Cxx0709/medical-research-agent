"""
Tests for the retraction-check feature (NIH Linked Discoveries context).

Covers: metadata flagging, the live E-utilities batch check (mocked),
verifier surfacing, synthesis warnings, and the pipeline node.
"""

import pytest

from src.models import Paper, AtomicClaim
from src.verification.retraction import RetractionChecker
from src.verification.verifier import CitationVerifier
from src.generator.synthesis import ReviewSynthesizer
from src.graph.nodes import retraction_check_node


def _paper(pmid: str, pub_types, abstract: str = "dapagliflozin reduced heart failure hospitalization") -> Paper:
    return Paper(
        pmid=pmid,
        title=f"Study on {pmid}",
        abstract=abstract,
        journal="Test J",
        pub_year=2023,
        publication_types=list(pub_types),
    )


# --- metadata flagging ----------------------------------------------------

def test_is_retracted_case_insensitive():
    assert RetractionChecker.is_retracted(["Retracted Publication"])
    assert RetractionChecker.is_retracted(["RETRACTED PUBLICATION", "Journal Article"])
    assert RetractionChecker.is_retracted(["  retracted publication  "])
    assert not RetractionChecker.is_retracted(["Journal Article", "Randomized Controlled Trial"])
    assert not RetractionChecker.is_retracted([])
    assert not RetractionChecker.is_retracted(None)


def test_flag_papers_from_metadata():
    papers = [
        _paper("111", ["Journal Article", "Retracted Publication"]),
        _paper("222", ["Journal Article"]),
    ]
    RetractionChecker().flag_papers(papers)
    assert papers[0].retracted is True
    assert "撤稿" in papers[0].retraction_note
    assert papers[1].retracted is False
    assert papers[1].retraction_note == ""


def test_retraction_notice_itself_is_not_flagged():
    # A retraction *notice* (type "Retraction of Publication") is not the
    # retracted paper itself -- it must not be flagged as retracted.
    papers = [_paper("333", ["Retraction of Publication", "Journal Article"])]
    RetractionChecker().flag_papers(papers)
    assert papers[0].retracted is False


# --- live E-utilities batch check (mocked) --------------------------------

class _FakeResp:
    def __init__(self, ids):
        self._ids = ids

    def raise_for_status(self):
        pass

    def json(self):
        return {"esearchresult": {"idlist": self._ids}}


def test_check_batch_via_eutils(monkeypatch):
    calls = {}

    def fake_get(url, params=None, timeout=None):
        calls["term"] = params["term"]
        return _FakeResp(["111"])

    monkeypatch.setattr("src.verification.retraction.requests.get", fake_get)
    result = RetractionChecker().check_batch_via_eutils(["111", "222", "333"])
    assert result == {"111": True}
    # single batched query, not one call per PMID; numeric PMIDs unquoted
    assert "111[pmid]" in calls["term"] and "333[pmid]" in calls["term"]
    assert '"111"[pmid]' not in calls["term"]
    assert "retracted publication" in calls["term"]


def test_check_papers_live_check_adds_flag(monkeypatch):
    papers = [_paper("111", ["Journal Article"]), _paper("222", ["Journal Article"])]
    monkeypatch.setattr(
        "src.verification.retraction.requests.get",
        lambda url, params=None, timeout=None: _FakeResp(["222"]),
    )
    RetractionChecker().check_papers(papers, live_cross_check=True)
    assert papers[0].retracted is False
    assert papers[1].retracted is True
    assert "实时核验" in papers[1].retraction_note


def test_check_papers_fail_open_on_network_error(monkeypatch):
    papers = [_paper("111", ["Retracted Publication"])]  # metadata flag present

    def boom(url, params=None, timeout=None):
        raise ConnectionError("offline")

    monkeypatch.setattr("src.verification.retraction.requests.get", boom)
    # must not raise; metadata flag survives
    RetractionChecker().check_papers(papers, live_cross_check=True)
    assert papers[0].retracted is True


# --- verifier integration -------------------------------------------------

def test_verifier_surfaces_retracted_citation():
    paper = _paper("111", ["Retracted Publication"])
    paper.retracted = True
    paper.retraction_note = "PubMed 题录出版类型为 Retracted Publication（已撤稿）"
    claim = AtomicClaim(
        claim_id="C1",
        statement="dapagliflozin reduced heart failure hospitalization in the trial",
        cited_pmids=["111"],
    )
    report = CitationVerifier().verify([claim], [paper])
    assert report.retracted_pmids == ["111"]
    assert len(report.retraction_warnings) == 1
    assert "111" in report.retraction_warnings[0]
    assert "撤稿" in report.claims[0].explanation
    # retraction is a parallel warning channel: guardrail math unchanged
    assert report.claims[0].verification_status.value == "SUPPORTED"


def test_verifier_no_retraction_when_clean():
    paper = _paper("222", ["Journal Article"])
    claim = AtomicClaim(
        claim_id="C1",
        statement="dapagliflozin reduced heart failure hospitalization in the trial",
        cited_pmids=["222"],
    )
    report = CitationVerifier().verify([claim], [paper])
    assert report.retracted_pmids == []
    assert report.retraction_warnings == []


# --- synthesis warnings ---------------------------------------------------

def test_synthesis_template_warns_on_retracted():
    papers = [
        _paper("111", ["Retracted Publication"], abstract="dapagliflozin reduced heart failure hospitalization in patients"),
        _paper("222", ["Journal Article"], abstract="empagliflozin improved outcomes in heart failure"),
    ]
    papers[0].retracted = True
    # mimic the grader: synthesis groups by evidence level
    from src.models import PICOElements, EvidenceLevel
    papers[0].evidence_level = EvidenceLevel.LEVEL_2
    papers[1].evidence_level = EvidenceLevel.LEVEL_2
    pico = PICOElements(population="HFpEF", intervention="SGLT-2i")
    review, claims = ReviewSynthesizer()._synthesize_template("test query", pico, papers)
    assert "撤稿警示" in review
    assert "⚠️【已撤稿" in review
    assert "[PMID:111]" in review


# --- pipeline node --------------------------------------------------------

def test_retraction_check_node_flags_from_metadata():
    state = {
        "query": "q",
        "retriever_source": "mock",  # no live check for mock source
        "retrieved_papers": [
            _paper("111", ["Retracted Publication"]).model_dump(),
            _paper("222", ["Journal Article"]).model_dump(),
        ],
    }
    out = retraction_check_node(state)
    by_pmid = {p["pmid"]: p for p in out["retrieved_papers"]}
    assert by_pmid["111"]["retracted"] is True
    assert by_pmid["222"]["retracted"] is False
