"""
LangGraph Workflow Assembly for Medical Research Agent
"""

from langgraph.graph import StateGraph, START, END
from src.state import MedicalResearchState
from src.graph.nodes import (
    query_analysis_node,
    retrieval_node,
    retraction_check_node,
    annotation_node,
    ranking_node,
    review_synthesis_node,
    citation_verification_node,
    finalize_node,
)


def build_medical_research_graph():
    """Builds and compiles the full LangGraph pipeline."""
    graph = StateGraph(MedicalResearchState)

    # 1. Register Nodes
    graph.add_node("query_analysis", query_analysis_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("retraction_check", retraction_check_node)
    graph.add_node("annotation", annotation_node)
    graph.add_node("ranking", ranking_node)
    graph.add_node("review_synthesis", review_synthesis_node)
    graph.add_node("citation_verification", citation_verification_node)
    graph.add_node("finalize", finalize_node)

    # 2. Add Flow Edges
    graph.add_edge(START, "query_analysis")
    graph.add_edge("query_analysis", "retrieval")
    graph.add_edge("retrieval", "retraction_check")
    graph.add_edge("retraction_check", "annotation")
    graph.add_edge("annotation", "ranking")
    graph.add_edge("ranking", "review_synthesis")
    graph.add_edge("review_synthesis", "citation_verification")
    graph.add_edge("citation_verification", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
