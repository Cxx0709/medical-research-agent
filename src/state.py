"""
Medical Research LangGraph State Definition
"""

from typing import TypedDict, List, Dict, Any, Optional
from src.models import Paper, PICOElements, AtomicClaim, VerificationReport


class MedicalResearchState(TypedDict):
    # User Inputs & Config
    query: str
    retriever_source: Optional[str]  # 'pubmed' | 'semantic_scholar' | 'mock'

    # PICO & Query Strategy
    pico: Optional[Dict[str, str]]
    search_terms: List[str]

    # Retrieval & Ranking
    retrieved_papers: List[Dict[str, Any]]
    ranked_papers: List[Dict[str, Any]]

    # Evidence Grading
    annotated_papers: List[Dict[str, Any]]

    # Review Synthesis
    draft_review: str
    claims: List[Dict[str, Any]]

    # Verification & Hallucination Guardrail
    verification_report: Optional[Dict[str, Any]]
    final_review: str
    iteration_count: int
    has_hallucination: bool

    # Final execution metadata
    error: Optional[str]
