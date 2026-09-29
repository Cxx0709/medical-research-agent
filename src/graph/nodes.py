"""
LangGraph Node Implementations for Medical Research Agent
"""

import logging
from typing import Dict, Any, List
from src.state import MedicalResearchState
from src.models import Paper, PICOElements, AtomicClaim
from src.retriever.mock_retriever import MockPubMedRetriever
from src.retriever.pubmed import PubMedRetriever
from src.retriever.semanticscholar import SemanticScholarRetriever
from src.ranking.ranker import LiteratureRanker
from src.evidence.grader import EvidenceGrader
from src.generator.synthesis import ReviewSynthesizer
from src.verification.verifier import CitationVerifier

logger = logging.getLogger(__name__)


def parse_pico_rules(query: str) -> PICOElements:
    """Rule-based extractor of clinical PICO elements from medical query."""
    q_lower = query.lower()
    
    # 1. Intervention extraction
    intervention = "药物干预"
    if any(k in q_lower for k in ["sglt-2", "sglt2", "列净", "dapagliflozin", "empagliflozin"]):
        intervention = "SGLT-2 抑制剂 (达格列净/恩格列净)"
    elif any(k in q_lower for k in ["glp-1", "glp1", "司美格鲁肽", "semaglutide"]):
        intervention = "GLP-1 受体激动剂 (司美格鲁肽)"
    elif any(k in q_lower for k in ["pd-1", "pdl1", "pd-l1", "免疫检查点", "pembrolizumab"]):
        intervention = "PD-1/PD-L1 免疫检查点抑制剂"
    elif "治疗" in query:
        parts = query.split("治疗")
        intervention = parts[0][-15:].strip()

    # 2. Population extraction
    population = "临床目标人群"
    if any(k in q_lower for k in ["hfpef", "射血分数保留", "射血分数轻度降低", "心衰"]):
        population = "心衰伴射血分数保留 (HFpEF) 或轻度降低 (HFmrEF) 患者"
    elif any(k in q_lower for k in ["肥胖", "超重", "obesity", "weight loss"]):
        population = "非糖尿病合并超重/肥胖且伴心血管疾病患者"
    elif any(k in q_lower for k in ["肺癌", "nsclc", "非小细胞肺癌"]):
        population = "晚期非小细胞肺癌 (NSCLC) 患者"
    elif any(k in q_lower for k in ["糖尿病", "t2d", "diabetes"]):
        population = "2型糖尿病患者"

    # 3. Outcome
    outcome = "主要临床结局（心血管死亡、心衰住院率、MACE或减重幅度）"

    return PICOElements(
        population=population,
        intervention=intervention,
        comparison="安慰剂或标准指南疗法",
        outcome=outcome
    )


INTERVENTION_EN = [
    (["sglt-2", "sglt2", "列净", "dapagliflozin", "empagliflozin"], "SGLT2 inhibitor"),
    (["glp-1", "glp1", "司美格鲁肽", "semaglutide"], "GLP-1 receptor agonist"),
    (["pd-1", "pdl1", "pd-l1", "免疫检查点", "pembrolizumab"], "PD-1 immune checkpoint inhibitor"),
]

POPULATION_EN = [
    (["hfpef", "射血分数保留", "射血分数轻度降低", "心衰"], "heart failure with preserved ejection fraction"),
    (["肥胖", "超重", "obesity"], "obesity"),
    (["肺癌", "nsclc", "非小细胞肺癌"], "non-small cell lung cancer"),
    (["糖尿病", "diabetes", "t2d"], "type 2 diabetes"),
]


def english_keywords(query: str, pico: PICOElements) -> tuple:
    """Map a (possibly Chinese) clinical question to English PubMed search keywords."""
    q_lower = query.lower()
    intervention_en, population_en = "", ""
    for keys, label in INTERVENTION_EN:
        if any(k in q_lower for k in keys):
            intervention_en = label
            break
    for keys, label in POPULATION_EN:
        if any(k in q_lower for k in keys):
            population_en = label
            break
    # If the query is already English but matched nothing, use it verbatim.
    if not intervention_en and not population_en and query.strip().isascii():
        return query.strip(), ""
    return intervention_en, population_en


def query_analysis_node(state: MedicalResearchState) -> Dict[str, Any]:
    """Node 1: Extract PICO and generate targeted search keywords."""
    query = state.get("query", "")
    pico = parse_pico_rules(query)

    intervention_en, population_en = english_keywords(query, pico)
    if intervention_en and population_en:
        search_terms = [
            f"{intervention_en} {population_en}",
            f"{intervention_en} {population_en} randomized controlled trial",
            f"{intervention_en} {population_en} meta-analysis",
        ]
    elif intervention_en or population_en:
        core = f"{intervention_en} {population_en}".strip()
        search_terms = [core, f"{core} randomized controlled trial", f"{core} meta-analysis"]
    else:
        # Unknown/unsupported phrasing: let the retriever try the raw query;
        # PubMedRetriever falls back to mock on zero results.
        search_terms = [query]

    return {
        "pico": pico.model_dump(),
        "search_terms": search_terms,
        "iteration_count": 0,
        "error": None,
    }


def retrieval_node(state: MedicalResearchState) -> Dict[str, Any]:
    """Node 2: Retrieve literature using configured source (pubmed / semantic_scholar / mock).

    Fans out over all search terms and dedupes by PMID so the evidence base
    covers both general and design-specific (RCT / meta-analysis) hits.
    """
    query = state.get("query", "")
    search_terms = state.get("search_terms", [query]) or [query]
    source = (state.get("retriever_source") or "pubmed").lower().strip()

    if source in ["semantic_scholar", "semanticscholar", "s2"]:
        retriever = SemanticScholarRetriever()
    elif source == "mock":
        retriever = MockPubMedRetriever()
    else:
        retriever = PubMedRetriever()

    seen_pmids = set()
    merged: List[Paper] = []
    per_term = max(2, 8 // max(1, len(search_terms)))
    for term in search_terms:
        try:
            batch = retriever.search(query=term, max_results=per_term)
        except Exception as exc:
            logger.warning(f"Retrieval failed for term {term!r}: {exc}")
            continue
        for paper in batch:
            if paper.pmid and paper.pmid not in seen_pmids:
                seen_pmids.add(paper.pmid)
                merged.append(paper)
        if len(merged) >= 8:
            break

    return {
        "retrieved_papers": [p.model_dump() for p in merged[:8]]
    }


def annotation_node(state: MedicalResearchState) -> Dict[str, Any]:
    """Node 3: Annotate study design and assign Oxford CEBM evidence level."""
    retrieved = state.get("retrieved_papers", [])
    papers = [Paper(**p) for p in retrieved]
    
    grader = EvidenceGrader()
    annotated = grader.grade_batch(papers)
    
    return {
        "annotated_papers": [p.model_dump() for p in annotated]
    }


def ranking_node(state: MedicalResearchState) -> Dict[str, Any]:
    """Node 4: Rank papers based on relevance, evidence level, and recency."""
    annotated = state.get("annotated_papers", [])
    papers = [Paper(**p) for p in annotated]
    query = state.get("query", "")
    
    ranker = LiteratureRanker()
    ranked = ranker.rank(papers, query)
    
    return {
        "ranked_papers": [p.model_dump() for p in ranked]
    }


def review_synthesis_node(state: MedicalResearchState) -> Dict[str, Any]:
    """Node 5: Synthesize review with mandatory citation anchoring [PMID:xxx]."""
    ranked = state.get("ranked_papers", [])
    papers = [Paper(**p) for p in ranked]
    pico_dict = state.get("pico", {})
    pico = PICOElements(**pico_dict) if pico_dict else parse_pico_rules(state.get("query", ""))
    
    synthesizer = ReviewSynthesizer()
    draft, claims = synthesizer.synthesize(
        query=state.get("query", ""),
        pico=pico,
        papers=papers
    )
    
    return {
        "draft_review": draft,
        "claims": [c.model_dump() for c in claims]
    }


def citation_verification_node(state: MedicalResearchState) -> Dict[str, Any]:
    """Node 6: Hallucination Guardrail - Validate citations against source abstracts."""
    claims_data = state.get("claims", [])
    claims = [AtomicClaim(**c) for c in claims_data]
    
    papers_data = state.get("ranked_papers", [])
    papers = [Paper(**p) for p in papers_data]
    
    verifier = CitationVerifier()
    report = verifier.verify(claims, papers)
    
    has_hallucination = not report.passed_guardrail
    
    return {
        "verification_report": report.model_dump(),
        "has_hallucination": has_hallucination,
    }


def finalize_node(state: MedicalResearchState) -> Dict[str, Any]:
    """Node 7: Assemble verified final review with quality badge and metrics."""
    draft = state.get("draft_review", "")
    report_dict = state.get("verification_report", {})
    
    precision = report_dict.get("citation_precision", 1.0) * 100
    hallucination_rate = report_dict.get("hallucination_rate", 0.0) * 100
    total_claims = report_dict.get("total_claims", 0)
    supported_claims = report_dict.get("supported_claims", 0)

    quality_banner = (
        "\n---\n"
        "### 🛡️ 循证质量与引用溯源核验报告\n\n"
        f"- **引用准确率 (Citation Precision)**: `{precision:.1f}%`\n"
        f"- **幻觉率 (Hallucination Rate)**: `{hallucination_rate:.1f}%`\n"
        f"- **核验论断数**: 共 `{total_claims}` 条核心论断，其中 `{supported_claims}` 条通过原文摘要严格溯源核验。\n"
        f"- **质检状态**: `{'✅ 已通过「无来源不生成」防幻觉门禁' if report_dict.get('passed_guardrail', True) else '⚠️ 存在部分未核验引用'}`\n"
    )

    final = draft + quality_banner

    return {
        "final_review": final
    }
