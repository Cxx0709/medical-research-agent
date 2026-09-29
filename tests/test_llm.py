"""
Unit tests for LLM Client and Graceful Degradation
"""

import os
from unittest.mock import patch, MagicMock
from src.llm.client import LLMClient
from src.models import Paper, AtomicClaim, StudyType, EvidenceLevel, PICOElements, VerificationStatus
from src.generator.synthesis import ReviewSynthesizer
from src.verification.verifier import CitationVerifier


def test_llm_client_unavailable_by_default(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    client = LLMClient(api_key="")
    assert not client.is_available
    assert client.chat_completion([{"role": "user", "content": "hi"}]) is None


def test_llm_synthesizer_fallback_when_no_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    synthesizer = ReviewSynthesizer(llm_client=LLMClient(api_key=""))
    paper = Paper(
        pmid="36049247",
        title="SGLT2 inhibitors meta-analysis",
        abstract="Composite cardiovascular death or hospitalization significantly reduced.",
        study_type=StudyType.SYSTEMATIC_REVIEW_META_ANALYSIS,
        evidence_level=EvidenceLevel.LEVEL_1,
        journal="The Lancet",
        pub_year=2022,
    )
    pico = PICOElements(population="心衰患者", intervention="SGLT-2抑制剂")
    review, claims = synthesizer.synthesize("SGLT-2 心衰", pico, [paper])
    assert "证据概览" in review
    assert len(claims) > 0
    assert claims[0].cited_pmids == ["36049247"]
    # Extractive guarantee: every claim must quote the cited paper's own text
    assert paper.abstract[:60] in claims[0].statement or paper.title in claims[0].statement


def test_llm_verifier_fallback_when_no_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    verifier = CitationVerifier(llm_client=LLMClient(api_key=""))
    paper = Paper(
        pmid="36049247",
        title="SGLT2 inhibitors meta-analysis",
        abstract="Composite cardiovascular death or hospitalization significantly reduced.",
        study_type=StudyType.SYSTEMATIC_REVIEW_META_ANALYSIS,
        evidence_level=EvidenceLevel.LEVEL_1,
        journal="The Lancet",
        pub_year=2022,
    )
    # Extractive claim quoting the source: rule-based fallback can verify it
    claim = AtomicClaim(
        claim_id="C1",
        statement="【Systematic Review / Meta-Analysis｜《The Lancet》(2022)】SGLT2 inhibitors meta-analysis。Composite cardiovascular death or hospitalization significantly reduced. [PMID:36049247]",
        cited_pmids=["36049247"],
    )
    report = verifier.verify([claim], [paper])
    assert report.total_claims == 1
    assert report.supported_claims == 1
    assert "规则核验通过" in report.claims[0].explanation


def test_llm_verifier_crosslingual_paraphrase_is_uncertain(monkeypatch):
    """Honest limitation: without an LLM, a Chinese paraphrase of an English
    abstract cannot be verified by lexical overlap. The verifier must report
    uncertainty (NOT_MENTIONED) rather than fake confidence."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    verifier = CitationVerifier(llm_client=LLMClient(api_key=""))
    paper = Paper(
        pmid="36049247",
        title="SGLT2 inhibitors meta-analysis",
        abstract="Composite cardiovascular death or hospitalization significantly reduced.",
        study_type=StudyType.SYSTEMATIC_REVIEW_META_ANALYSIS,
        evidence_level=EvidenceLevel.LEVEL_1,
        journal="The Lancet",
        pub_year=2022,
    )
    claim = AtomicClaim(
        claim_id="C1",
        statement="SGLT-2抑制剂显著降低心衰住院风险 [PMID:36049247]。",
        cited_pmids=["36049247"],
    )
    report = verifier.verify([claim], [paper])
    assert report.total_claims == 1
    assert report.supported_claims == 0
    assert report.claims[0].verification_status == VerificationStatus.NOT_MENTIONED


def test_llm_mock_call_integration():
    mock_client = MagicMock(spec=LLMClient)
    mock_client.is_available = True
    mock_client.chat_completion.return_value = '{"status": "SUPPORTED", "explanation": "摘要中明确指出心血管死亡率下降"}'
    
    verifier = CitationVerifier(llm_client=mock_client)
    paper = Paper(
        pmid="12345",
        title="Trial of drug X",
        abstract="Drug X reduces cardiovascular events.",
        study_type=StudyType.RANDOMIZED_CONTROLLED_TRIAL,
        evidence_level=EvidenceLevel.LEVEL_2,
    )
    claim = AtomicClaim(
        claim_id="C1",
        statement="Drug X reduces mortality [PMID:12345].",
        cited_pmids=["12345"],
    )
    report = verifier.verify([claim], [paper])
    assert report.supported_claims == 1
    assert "LLM核验通过" in report.claims[0].explanation
