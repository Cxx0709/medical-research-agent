"""
Unit tests for Citation Verifier and Hallucination Guardrail
"""

import pytest
from src.models import Paper, AtomicClaim, StudyType, EvidenceLevel, VerificationStatus
from src.verification.verifier import CitationVerifier


def test_citation_verifier_supported():
    verifier = CitationVerifier()
    paper = Paper(
        pmid="36049247",
        title="SGLT2 inhibitors in heart failure with mildly reduced or preserved ejection fraction: a meta-analysis",
        abstract="SGLT2 inhibitors significantly reduced composite cardiovascular death or hospitalization (HR 0.80).",
        study_type=StudyType.SYSTEMATIC_REVIEW_META_ANALYSIS,
        evidence_level=EvidenceLevel.LEVEL_1,
    )
    claim = AtomicClaim(
        claim_id="C1",
        statement="SGLT2 inhibitors significantly reduced cardiovascular death and hospitalization in HFpEF [PMID:36049247].",
        cited_pmids=["36049247"]
    )
    report = verifier.verify([claim], [paper])
    assert report.total_claims == 1
    assert report.supported_claims == 1
    assert report.unsupported_claims == 0
    assert report.citation_precision == 1.0
    assert report.hallucination_rate == 0.0
    assert report.passed_guardrail is True


def test_citation_verifier_hallucination_detection():
    verifier = CitationVerifier()
    paper = Paper(
        pmid="36049247",
        title="SGLT2 inhibitors in heart failure",
        abstract="SGLT2 inhibitors reduced cardiovascular death.",
        study_type=StudyType.SYSTEMATIC_REVIEW_META_ANALYSIS,
        evidence_level=EvidenceLevel.LEVEL_1,
    )
    # Claim cites a fabricated PMID not in the database
    fake_claim = AtomicClaim(
        claim_id="C2",
        statement="Drug ABC cures brain cancer in 100% of patients [PMID:99999999].",
        cited_pmids=["99999999"]
    )
    report = verifier.verify([fake_claim], [paper])
    assert report.total_claims == 1
    assert report.supported_claims == 0
    assert report.unsupported_claims == 1
    assert report.citation_precision == 0.0
    assert report.hallucination_rate == 1.0
    assert report.passed_guardrail is False
    assert report.claims[0].verification_status == VerificationStatus.NOT_MENTIONED
