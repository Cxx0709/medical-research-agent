"""
Relevance and Evidence Quality Ranker for Medical Papers
Combines semantic text matching, clinical evidence hierarchy weights, and publication recency.
"""

from typing import List
from src.models import Paper, EvidenceLevel


class LiteratureRanker:
    """Ranks retrieved literature based on clinical query relevance and evidence hierarchy."""

    LEVEL_WEIGHTS = {
        EvidenceLevel.LEVEL_1: 1.0,
        EvidenceLevel.LEVEL_2: 0.88,
        EvidenceLevel.LEVEL_3: 0.68,
        EvidenceLevel.LEVEL_4: 0.48,
        EvidenceLevel.LEVEL_5: 0.28,
        None: 0.4,
    }

    def __init__(self, w_rel: float = 0.50, w_level: float = 0.35, w_recency: float = 0.15):
        self.w_rel = w_rel
        self.w_level = w_level
        self.w_recency = w_recency

    def rank(self, papers: List[Paper], query: str) -> List[Paper]:
        """Rank papers using composite scoring."""
        scored = []
        current_year = 2026

        query_terms = [t.lower().strip() for t in query.replace(",", " ").split() if len(t) > 1]

        for p in papers:
            # 1. Relevance Score
            text_corpus = f"{p.title} {p.abstract} {' '.join(p.mesh_terms)}".lower()
            overlap = sum(1 for term in query_terms if term in text_corpus)
            rel_score = min(1.0, overlap / (len(query_terms) + 1e-4) + (p.relevance_score * 0.5))

            # 2. Evidence Hierarchy Weight
            level_weight = self.LEVEL_WEIGHTS.get(p.evidence_level, 0.4)

            # 3. Recency Decay
            age = max(0, current_year - p.pub_year) if p.pub_year > 0 else 10
            recency_score = max(0.2, 1.0 - (age * 0.05))

            composite_score = (
                self.w_rel * rel_score +
                self.w_level * level_weight +
                self.w_recency * recency_score
            )

            p_ranked = p.model_copy()
            p_ranked.relevance_score = round(composite_score, 4)
            scored.append(p_ranked)

        # Sort descending by composite score
        scored.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored
