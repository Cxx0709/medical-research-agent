"""
CLI Entrypoint for Medical Research Agent
Allows doctors and researchers to ask clinical questions and receive evidence-graded,
citation-traceable reviews with automated hallucination checks.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from src.graph.workflow import build_medical_research_graph


def run_medical_agent(query: str, retriever: str = "pubmed"):
    print("=" * 70)
    print("🩺 医学文献研究 Agent (Medical Research Agent)")
    print(f"原则：无来源不生成 | Oxford CEBM 证据等级标注 | 检索源: {retriever}")
    print("=" * 70)
    print(f"\n[1/5] 📥 接收临床提问: {query}\n")

    app = build_medical_research_graph()

    initial_state = {
        "query": query,
        "retriever_source": retriever,
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

    print("[2/5] 🔍 执行 LangGraph 工作流 (PICO解析 -> 文献检索 -> 证据分级 -> 重排 -> 综述生成 -> 引用核验)...")
    final_output = app.invoke(initial_state)

    print("\n[3/5] 📄 生成最终循证综述：\n")
    print(final_output["final_review"])
    print("\n" + "=" * 70)
    print("✅ 全链路执行完毕！")


def main():
    parser = argparse.ArgumentParser(description="Medical Research Agent CLI")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="SGLT-2抑制剂对心衰伴射血分数保留 (HFpEF) 患者的预后改善证据如何？",
        help="Doctor's clinical question in natural language",
    )
    parser.add_argument(
        "--retriever",
        "-r",
        type=str,
        default="pubmed",
        choices=["pubmed", "semantic_scholar", "mock"],
        help="Literature retrieval source (pubmed / semantic_scholar / mock)",
    )
    args = parser.parse_args()
    run_medical_agent(args.query, args.retriever)


if __name__ == "__main__":
    main()
