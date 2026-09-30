"""
Retraction Checker for the Medical Research Agent.

Detects retracted publications so the pipeline never silently cites a paper
that PubMed later retracted -- a failure mode the abstract-support check
alone cannot catch.

Two complementary signals, both via NCBI E-utilities (no new dependencies):

1. Metadata flags (offline, deterministic): a paper whose PubMed
   ``publication_types`` contain "Retracted Publication" is the retracted
   paper itself. This data is already fetched by PubMedRetriever
   (ESummary + EFetch Medline types), so flagging costs zero extra calls.

2. Live batch cross-check (online, fail-open): a single ESearch query of the
   form ``("pmid1"[pmid] OR "pmid2"[pmid]) AND "retracted publication"[pt]``
   returns the subset of a result batch that PubMed currently marks as
   retracted. Used by the pipeline for the ``pubmed`` source to catch
   anything the cached metadata missed; network failure never breaks the
   pipeline -- metadata flags are kept.

Context: NIH/NLM launched the PubMed tool "Linked Discoveries" (Sept 24,
2026), which explicitly surfaces retractions as core literature context.
Linked Discoveries has no public API (web UI only), so this module goes
through E-utilities instead -- same underlying PubMed data.
"""

import logging
from typing import Dict, List

import requests

from src.models import Paper
from src.retriever.pubmed import EUTILS_BASE

logger = logging.getLogger(__name__)

#: Publication types (lowercased) that mark the paper itself as retracted.
RETRACTED_PUBLICATION_TYPES = {"retracted publication"}

#: Publication types (lowercased) that mark a retraction *notice* rather than
#: the retracted paper itself. Tracked for completeness, not flagged.
RETRACTION_NOTICE_TYPES = {"retraction of publication"}


class RetractionChecker:
    """Flags retracted papers in a retrieved batch."""

    def __init__(self, email: str = "medical_agent@example.com",
                 api_key: str | None = None, timeout: float = 8.0):
        self.email = email
        self.api_key = api_key
        self.timeout = timeout

    @staticmethod
    def is_retracted(publication_types: List[str]) -> bool:
        """True if any publication type marks the paper as retracted."""
        return any(
            (pt or "").strip().lower() in RETRACTED_PUBLICATION_TYPES
            for pt in (publication_types or [])
        )

    def flag_papers(self, papers: List[Paper]) -> List[Paper]:
        """Flag retracted papers from already-fetched metadata (in place).

        Never fabricates: only sets the flag when PubMed's own publication
        types say so.
        """
        for paper in papers:
            if self.is_retracted(paper.publication_types):
                paper.retracted = True
                if not paper.retraction_note:
                    paper.retraction_note = (
                        "PubMed 题录出版类型为 Retracted Publication（已撤稿）"
                    )
        return papers

    def check_batch_via_eutils(self, pmids: List[str]) -> Dict[str, bool]:
        """Live cross-check: which of these PMIDs does PubMed mark retracted?

        Single ESearch call for the whole batch. Returns {pmid: True} for
        the retracted subset; PMIDs absent from the dict were not confirmed
        retracted (or the check could not run -- callers must fail open).
        """
        pmids = [p for p in dict.fromkeys(pmids) if p]
        if not pmids:
            return {}

        # NOTE: numeric PMIDs must NOT be quoted in the [pmid] field --
        # '"9500320"[pmid]' matches nothing, '9500320[pmid]' works.
        or_clause = " OR ".join(f"{pmid}[pmid]" for pmid in pmids)
        term = f'({or_clause}) AND "retracted publication"[pt]'
        params = {
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retmax": len(pmids),
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=self.timeout)
        resp.raise_for_status()
        id_list = resp.json().get("esearchresult", {}).get("idlist", [])
        return {pmid: True for pmid in id_list if pmid in set(pmids)}

    def check_papers(self, papers: List[Paper], live_cross_check: bool = True) -> List[Paper]:
        """Full check: metadata flags first, then optional live cross-check.

        The live check only ever *adds* flags, never clears metadata flags,
        and any network error is swallowed with a warning (fail-open).
        """
        self.flag_papers(papers)

        if live_cross_check:
            try:
                live = self.check_batch_via_eutils([p.pmid for p in papers if p.pmid])
                for paper in papers:
                    if live.get(paper.pmid) and not paper.retracted:
                        paper.retracted = True
                        paper.retraction_note = (
                            "经 E-utilities 实时核验：PubMed 将其标记为 Retracted Publication（已撤稿）"
                        )
            except Exception as exc:
                logger.warning(f"Retraction live cross-check failed ({exc}); keeping metadata flags.")

        return papers
