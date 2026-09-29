"""
Citation Verifier and Hallucination Guardrail
Validates claim-level citations against source literature abstracts to ensure 'No Citation, No Claim'.
Supports LLM Natural Language Inference (NLI) with automatic rule-based fallback.
"""

import re
import json
import logging
from typing import List, Dict, Tuple, Optional
from src.models import Paper, AtomicClaim, VerificationReport, VerificationStatus
from src.llm.client import LLMClient

logger = logging.getLogger(__name__)


class CitationVerifier:
    """Verifies that atomic claims in the synthesized review are strictly supported by cited papers."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def verify(self, claims: List[AtomicClaim], papers: List[Paper]) -> VerificationReport:
        paper_dict: Dict[str, Paper] = {p.pmid: p for p in papers}
        verified_claims: List[AtomicClaim] = []

        total_citations = 0
        supported_citations = 0

        for claim in claims:
            v_claim = claim.model_copy()
            v_claim.supported_pmids = []
            v_claim.unsupported_pmids = []

            for pmid in claim.cited_pmids:
                total_citations += 1
                paper = paper_dict.get(pmid)
                if not paper:
                    # Cited an unretrieved or hallucinated PMID
                    v_claim.unsupported_pmids.append(pmid)
                    continue

                # Check support via LLM NLI or rule-based fallback
                is_supported, explanation = self._check_support(claim.statement, paper)
                if is_supported:
                    v_claim.supported_pmids.append(pmid)
                    supported_citations += 1
                    v_claim.explanation = explanation
                else:
                    v_claim.unsupported_pmids.append(pmid)
                    if explanation:
                        v_claim.explanation = explanation

            if len(v_claim.supported_pmids) == len(v_claim.cited_pmids) and len(v_claim.cited_pmids) > 0:
                v_claim.verification_status = VerificationStatus.SUPPORTED
            elif len(v_claim.supported_pmids) > 0:
                v_claim.verification_status = VerificationStatus.SUPPORTED
            else:
                v_claim.verification_status = VerificationStatus.NOT_MENTIONED
                if not v_claim.explanation:
                    v_claim.explanation = "Citation abstract does not verify claim."

            verified_claims.append(v_claim)

        supported_count = sum(1 for c in verified_claims if c.verification_status == VerificationStatus.SUPPORTED)
        unsupported_count = len(verified_claims) - supported_count

        precision = (supported_citations / total_citations) if total_citations > 0 else 0.0
        hallucination_rate = (unsupported_count / len(verified_claims)) if verified_claims else 0.0

        return VerificationReport(
            total_claims=len(verified_claims),
            supported_claims=supported_count,
            unsupported_claims=unsupported_count,
            citation_precision=round(precision, 4),
            hallucination_rate=round(hallucination_rate, 4),
            claims=verified_claims,
            passed_guardrail=(hallucination_rate == 0.0 and precision >= 0.95),
        )

    def _check_support(self, claim_text: str, paper: Paper) -> Tuple[bool, str]:
        """Verify if the paper abstract corroborates the statement (LLM or Rule)."""
        if self.llm_client.is_available:
            llm_res = self._check_support_with_llm(claim_text, paper)
            if llm_res is not None:
                return llm_res

        return self._check_support_rule(claim_text, paper)

    def _check_support_with_llm(self, claim_text: str, paper: Paper) -> Optional[Tuple[bool, str]]:
        """LLM NLI entailment inference."""
        system_prompt = (
            "You are an expert in medical natural language inference (NLI) and evidence verification.\n"
            "Determine whether the literature abstract provided strictly entails and supports the clinical statement.\n"
            "Respond ONLY with a JSON object in this format:\n"
            '{"status": "SUPPORTED" | "CONTRADICTED" | "NOT_MENTIONED", "explanation": "<short rationale in Chinese>"}'
        )

        user_content = (
            f"【文献题目与摘要 (PMID: {paper.pmid})】：\n"
            f"Title: {paper.title}\n"
            f"Abstract: {paper.abstract}\n\n"
            f"【待核验论断】：\n{claim_text}\n\n"
            "JSON response:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        raw = self.llm_client.chat_completion(messages, temperature=0.0, max_tokens=300)
        if not raw:
            return None

        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group(0))
            status = data.get("status", "").upper()
            explanation = data.get("explanation", "")
            if status == "SUPPORTED":
                return True, f"LLM核验通过：{explanation}"
            elif status in ["CONTRADICTED", "NOT_MENTIONED"]:
                return False, f"LLM核验未通过 ({status})：{explanation}"
            return None
        except Exception as e:
            logger.warning(f"Error parsing LLM NLI response: {e}")
            return None

    def _check_support_rule(self, claim_text: str, paper: Paper) -> Tuple[bool, str]:
        """Deterministic content-overlap rule.

        A claim is supported only if a substantial share of its meaningful
        semantic units actually appear in the cited paper's title/abstract.
        No auto-pass on study design: a claim about trial X is not supported
        by merely citing any RCT.
        """
        source_text = f"{paper.title} {paper.abstract}".lower()

        tokens = [t for t in re.findall(r"[\w\u4e00-\u9fa5]+", claim_text.lower()) if len(t) > 2]
        stopwords = {
            "根据", "发表", "研究", "表明", "显示", "证实", "临床", "患者", "治疗", "使用",
            "可以", "进行", "通过", "以及", "pmid", "the", "and", "for", "with",
        }
        meaningful = [t for t in tokens if t not in stopwords]
        if not meaningful:
            return False, "规则核验未通过：论断中无有效语义单元可供核验"

        matched = {t for t in meaningful if t in source_text}
        ratio = len(matched) / len(set(meaningful))

        if ratio >= 0.35 and len(matched) >= 3:
            return True, f"规则核验通过：{len(matched)} 个关键语义单元在 PMID {paper.pmid} 题录/摘要中找到对应"
        return False, f"规则核验未通过：PMID {paper.pmid} 摘要缺乏充分支持依据（语义匹配度 {ratio:.0%}）"
