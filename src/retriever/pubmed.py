"""
PubMed Literature Retriever using NCBI E-Utilities API
Uses ESearch + ESummary + EFetch: real abstracts and Medline publication types.
Falls back to the local clinical mock database only when the network fails
or returns zero results. Never fabricates abstracts.
"""

import logging
import time
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Tuple
import requests
from src.models import Paper
from src.retriever.base import BaseRetriever
from src.retriever.mock_retriever import MockPubMedRetriever

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def parse_efetch_xml(xml_text: str) -> Dict[str, Tuple[str, List[str], List[str]]]:
    """Parse EFetch XML into {pmid: (abstract, publication_types, mesh_terms)}."""
    result: Dict[str, Tuple[str, List[str], List[str]]] = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning(f"EFetch XML parse failed: {exc}")
        return result

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = (pmid_el.text or "").strip() if pmid_el is not None else ""
        if not pmid:
            continue

        abstract_parts = [
            (el.text or "").strip()
            for el in article.findall(".//AbstractText")
            if el.text and el.text.strip()
        ]
        abstract = " ".join(abstract_parts)

        pub_types = [
            (el.text or "").strip()
            for el in article.findall(".//PublicationType")
            if el.text and el.text.strip()
        ]

        mesh_terms = [
            (el.findtext("DescriptorName") or "").strip()
            for el in article.findall(".//MeshHeading")
            if el.findtext("DescriptorName")
        ]

        result[pmid] = (abstract, pub_types, mesh_terms)
    return result


class PubMedRetriever(BaseRetriever):
    """Retriever for PubMed using official E-utilities REST endpoints."""

    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None, timeout: float = 8.0):
        self.api_key = api_key
        self.email = email or "medical_agent@example.com"
        self.timeout = timeout
        self.fallback_retriever = MockPubMedRetriever()

    def search(self, query: str, max_results: int = 5) -> List[Paper]:
        """Search PubMed synchronously via E-Utilities."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            # Step 1: ESearch to get PMIDs
            esearch_url = f"{EUTILS_BASE}/esearch.fcgi"
            search_resp = requests.get(esearch_url, params=params, timeout=self.timeout)
            search_resp.raise_for_status()
            search_data = search_resp.json()

            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                logger.info("PubMed ESearch returned 0 results, falling back to mock.")
                return self.fallback_retriever.search(query, max_results)

            # Step 2: ESummary to get paper titles, authors, journals
            summary_url = f"{EUTILS_BASE}/esummary.fcgi"
            sum_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "json",
                "email": self.email,
            }
            if self.api_key:
                sum_params["api_key"] = self.api_key

            sum_resp = requests.get(summary_url, params=sum_params, timeout=self.timeout)
            sum_resp.raise_for_status()
            sum_data = sum_resp.json().get("result", {})

            papers = []
            for pmid in id_list:
                item = sum_data.get(pmid, {})
                if not item:
                    continue

                title = item.get("title", "").strip().rstrip(".")
                journal = item.get("source", "")
                pubdate = item.get("pubdate", "")
                year = 2023
                if pubdate:
                    parts = pubdate.split()
                    if parts and parts[0].isdigit():
                        year = int(parts[0])

                authors = [a.get("name", "") for a in item.get("authors", []) if "name" in a]
                pub_types = item.get("pubtypes", []) or item.get("pubtype", [])
                doi = ""
                for article_id in item.get("articleids", []):
                    if article_id.get("idtype") == "doi":
                        doi = article_id.get("value", "")
                        break

                papers.append(
                    Paper(
                        pmid=pmid,
                        title=title,
                        abstract="",  # filled by EFetch below; never fabricated
                        journal=journal,
                        pub_year=year,
                        authors=authors,
                        doi=doi,
                        publication_types=pub_types,
                    )
                )

            if papers:
                self._enrich_with_efetch(papers)
                return papers
            return self.fallback_retriever.search(query, max_results)

        except Exception as exc:
            logger.warning(f"PubMed online retrieval encountered error ({exc}). Falling back to local clinical mock database.")
            return self.fallback_retriever.search(query, max_results)

    def _enrich_with_efetch(self, papers: List[Paper]) -> None:
        """Fetch real abstracts + Medline publication types via EFetch (in place)."""
        pmids = [p.pmid for p in papers]
        if not pmids:
            return
        try:
            time.sleep(0.4)  # respect NCBI rate limits (3 req/s without API key)
            params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "rettype": "abstract",
                "email": self.email,
            }
            if self.api_key:
                params["api_key"] = self.api_key
            resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=self.timeout + 10)
            resp.raise_for_status()

            details = parse_efetch_xml(resp.text)
            for paper in papers:
                abstract, medline_pub_types, mesh_terms = details.get(paper.pmid, ("", [], []))
                if abstract:
                    paper.abstract = abstract
                if medline_pub_types:
                    # Medline publication types are authoritative; merge with esummary ones
                    merged = list(dict.fromkeys(list(medline_pub_types) + list(paper.publication_types)))
                    paper.publication_types = merged
                if mesh_terms:
                    paper.mesh_terms = mesh_terms
        except Exception as exc:
            logger.warning(f"EFetch enrichment failed ({exc}); papers keep esummary metadata.")

    async def asearch(self, query: str, max_results: int = 5) -> List[Paper]:
        """Asynchronous wrapper for search."""
        return self.search(query, max_results)
