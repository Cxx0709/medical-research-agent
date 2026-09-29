"""
End-to-End Pipeline Tests for Medical Research Agent
"""

import pytest
from src.graph.workflow import build_medical_research_graph


def test_full_pipeline_run():
    app = build_medical_research_graph()
    
    initial_state = {
        "query": "SGLT-2抑制剂对心衰伴射血分数保留 (HFpEF) 患者的预后改善证据如何？",
        "pico": None,
        "search_terms": [],
        "retrieved_papers": [],
        "ranked_papers": [],
        "annotated_papers": [],
        "draft_review": "",
        "claims": [],
        "verification_report": None,
        "final_review": "",
        "iteration_count": 0,
        "has_hallucination": False,
        "error": None,
    }

    result = app.invoke(initial_state)

    assert result["pico"] is not None
    assert "population" in result["pico"]
    assert "intervention" in result["pico"]

    assert len(result["retrieved_papers"]) > 0
    assert len(result["annotated_papers"]) > 0
    assert len(result["ranked_papers"]) > 0

    assert len(result["draft_review"]) > 50
    assert "PMID:" in result["draft_review"]

    assert result["verification_report"] is not None
    assert "citation_precision" in result["verification_report"]
    assert "final_review" in result
    assert "循证质量与引用溯源核验报告" in result["final_review"]
