"""
Medical Research Agent Data Models
Defines core clinical research domain models, evidence hierarchies, and verification structures.
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class StudyType(str, Enum):
    SYSTEMATIC_REVIEW_META_ANALYSIS = "Systematic Review / Meta-Analysis"
    RANDOMIZED_CONTROLLED_TRIAL = "Randomized Controlled Trial (RCT)"
    COHORT_STUDY = "Cohort Study"
    CASE_CONTROL_STUDY = "Case-Control Study"
    CASE_REPORT = "Case Report / Case Series"
    NARRATIVE_REVIEW = "Narrative Review"
    PRECLINICAL = "In Vitro / Animal Study"
    UNKNOWN = "Unknown / Undetermined"


class EvidenceLevel(str, Enum):
    LEVEL_1 = "Level 1 (Highest: Systematic Review / Meta-Analysis of RCTs)"
    LEVEL_2 = "Level 2 (High: Individual Randomized Controlled Trial)"
    LEVEL_3 = "Level 3 (Moderate: Cohort Study / Prospective Observational)"
    LEVEL_4 = "Level 4 (Low: Case-Control Study / Retrospective Case Series)"
    LEVEL_5 = "Level 5 (Very Low: Case Report / Expert Opinion / Mechanism Study)"


class VerificationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_MENTIONED = "NOT_MENTIONED"
    UNVERIFIED = "UNVERIFIED"


class PICOElements(BaseModel):
    population: str = Field(..., description="Target patient population or clinical condition")
    intervention: str = Field(..., description="Target drug, surgical procedure, or intervention")
    comparison: str = Field("", description="Active comparator, standard care, or placebo")
    outcome: str = Field("", description="Primary and secondary clinical endpoints")


class Paper(BaseModel):
    pmid: str = Field(..., description="PubMed Unique Identifier")
    title: str = Field(..., description="Title of the paper")
    abstract: str = Field("", description="Full or structured abstract text")
    journal: str = Field("", description="Publishing journal name")
    pub_year: int = Field(0, description="Year of publication")
    authors: List[str] = Field(default_factory=list, description="Author list")
    doi: str = Field("", description="Digital Object Identifier")
    mesh_terms: List[str] = Field(default_factory=list, description="Medical Subject Headings")
    publication_types: List[str] = Field(default_factory=list, description="PubMed publication types")
    study_type: StudyType = Field(default=StudyType.UNKNOWN, description="Identified study design")
    evidence_level: Optional[EvidenceLevel] = Field(default=None, description="Oxford CEBM Evidence Level")
    evidence_rationale: str = Field("", description="Reasoning for assigned evidence level")
    relevance_score: float = Field(0.0, description="Calculated relevance score to query")
    citation_count: int = Field(0, description="Academic citation count")
    open_access_pdf: str = Field("", description="URL to Open Access PDF if available")

    @property
    def pubmed_url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


class AtomicClaim(BaseModel):
    claim_id: str = Field(..., description="Identifier for the claim (e.g. C1, C2)")
    statement: str = Field(..., description="Atomic clinical statement extracted from review")
    cited_pmids: List[str] = Field(default_factory=list, description="PMIDs cited to back this claim")
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.UNVERIFIED,
        description="Verification outcome against source literature abstracts"
    )
    supported_pmids: List[str] = Field(default_factory=list, description="PMIDs proven to support this claim")
    unsupported_pmids: List[str] = Field(default_factory=list, description="PMIDs that fail to support this claim")
    explanation: str = Field("", description="Verification explanation and matched excerpts")


class VerificationReport(BaseModel):
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    citation_precision: float = 0.0  # supported / total citations
    hallucination_rate: float = 0.0  # claims with 0 supported sources / total claims
    claims: List[AtomicClaim] = Field(default_factory=list)
    passed_guardrail: bool = True
