"""
Streamlit Web Interface for Medical Research Agent
Provides clinical researchers with interactive query exploration,
PICO decomposition, evidence hierarchy browsing, and citation verification audits.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from src.graph.workflow import build_medical_research_graph
from src.models import Paper, EvidenceLevel, StudyType, VerificationStatus
from src.llm.client import LLMClient


# Configure Streamlit Page
st.set_page_config(
    page_title="医学文献研究 Agent",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .pico-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 10px;
    }
    .evidence-level-1 { border-left: 4px solid #10b981; }
    .evidence-level-2 { border-left: 4px solid #3b82f6; }
    .evidence-level-3 { border-left: 4px solid #f59e0b; }
    .evidence-level-5 { border-left: 4px solid #ef4444; }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    st.markdown('<div class="main-title">🩺 医学文献研究 Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title"><b>核心准则</b>：无来源不生成 · 细粒度引用溯源 · Oxford CEBM 证据等级标注 · 防幻觉严格核验</div>',
        unsafe_allow_html=True,
    )

    # 1. Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ 检索与模型配置")

        retriever_choice = st.radio(
            "文献检索来源",
            options=["PubMed (实时 E-Utilities)", "Semantic Scholar (学术被引/开源PDF)", "本地精标临床库 (Mock)"],
            index=0,
            help="PubMed 为官方医学权威库；Semantic Scholar 补充引用统计与开放获取 PDF；本地库支持完全脱机演示",
        )

        source_map = {
            "PubMed (实时 E-Utilities)": "pubmed",
            "Semantic Scholar (学术被引/开源PDF)": "semantic_scholar",
            "本地精标临床库 (Mock)": "mock",
        }
        retriever_source = source_map[retriever_choice]

        st.divider()

        # LLM Status check
        llm = LLMClient()
        if llm.is_available:
            st.success(f"🟢 **LLM 引擎已就绪**\n\n模型: `{llm.model}`\n\n端点: `{llm.base_url}`")
        else:
            st.info("🟡 **离线模式运行中**\n\n未检测到 `LLM_API_KEY`，系统自动使用高质量确定性循证规则模板，保证 100% 离线可用。")

        with st.expander("🔑 配置 LLM API Key (可选)"):
            st.caption("支持 OpenAI、DeepSeek、Gemini 等 OpenAI 兼容格式接口")
            api_key_input = st.text_input("LLM_API_KEY", type="password")
            base_url_input = st.text_input("LLM_BASE_URL", value=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"))
            model_input = st.text_input("LLM_MODEL", value=os.getenv("LLM_MODEL", "gpt-4o-mini"))
            if st.button("更新当前会话配置"):
                if api_key_input:
                    os.environ["LLM_API_KEY"] = api_key_input
                os.environ["LLM_BASE_URL"] = base_url_input
                os.environ["LLM_MODEL"] = model_input
                st.success("配置已更新！")
                st.rerun()

        st.divider()
        st.subheader("💡 经典临床问题预设")
        preset_questions = [
            "SGLT-2抑制剂对心衰伴射血分数保留 (HFpEF) 患者的预后改善证据如何？",
            "司美格鲁肽在肥胖非糖尿病患者中的心血管获益证据如何？",
            "晚期非小细胞肺癌一线免疫检查点抑制剂联合化疗的生存获益与证据等级？",
        ]
        for q in preset_questions:
            if st.button(f"📌 {q[:24]}...", key=q):
                st.session_state["query_input"] = q
                st.rerun()

    # 2. Main Question Input
    default_q = st.session_state.get(
        "query_input",
        "SGLT-2抑制剂对心衰伴射血分数保留 (HFpEF) 患者的预后改善证据如何？",
    )
    query = st.text_area("👨‍⚕️ 请输入临床提问（支持自然语言、临床中英文关键词）：", value=default_q, height=90)

    col1, col2 = st.columns([1, 5])
    with col1:
        run_btn = st.button("🔍 启动检索与循证分析", type="primary", use_container_width=True)

    if run_btn and query.strip():
        with st.spinner("⏳ 正在运行 LangGraph 流水线（PICO解析 → 文献检索 → 证据分级 → 综述合成 → 引用防幻觉核验）..."):
            app = build_medical_research_graph()
            initial_state = {
                "query": query.strip(),
                "retriever_source": retriever_source,
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
            st.session_state["result"] = result

    # 3. Display Results
    if "result" in st.session_state:
        result = st.session_state["result"]
        pico = result.get("pico") or {}
        rep = result.get("verification_report") or {}
        papers = result.get("ranked_papers") or []

        st.markdown("---")

        # PICO Bar
        st.subheader("🎯 1. 临床 PICO 结构分解")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.info(f"**👥 患者人群 (P)**\n\n{pico.get('population', 'N/A')}")
        with c2:
            st.success(f"**💊 干预措施 (I)**\n\n{pico.get('intervention', 'N/A')}")
        with c3:
            st.warning(f"**⚖️ 对照方案 (C)**\n\n{pico.get('comparison', '安慰剂/常规治疗')}")
        with c4:
            st.error(f"**🎯 结局指标 (O)**\n\n{pico.get('outcome', '主要临床结局')}")

        st.markdown("---")

        # Tabs for deep dive
        tab_review, tab_guardrail, tab_evidence, tab_raw = st.tabs(
            ["📋 循证综述正文 (带溯源)", "🛡️ 防幻觉质检报告", "🔬 证据分级文献卡片", "📄 参考文献表"]
        )

        # Tab 1: Review
        with tab_review:
            st.markdown(result.get("draft_review", "暂无综述生成"))

        # Tab 2: Guardrail & Verification Report
        with tab_guardrail:
            st.subheader("🛡️ 引用准确度与幻觉控制检测")
            precision = rep.get("citation_precision", 1.0) * 100
            hallucination_rate = rep.get("hallucination_rate", 0.0) * 100
            total_claims = rep.get("total_claims", 0)
            supported_claims = rep.get("supported_claims", 0)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("引用准确率 (Citation Precision)", f"{precision:.1f}%")
            m2.metric("幻觉率 (Hallucination Rate)", f"{hallucination_rate:.1f}%")
            m3.metric("核验论断数", f"{supported_claims} / {total_claims}")
            m4.metric("防幻觉门禁状态", "✅ 已通过" if rep.get("passed_guardrail", True) else "⚠️ 部分未核验")

            retraction_warnings = rep.get("retraction_warnings", []) or []
            if retraction_warnings:
                st.error("⚠️ **撤稿警示**：以下被引文献已被 PubMed 标记为撤稿，其结论不应作为临床证据使用：")
                for w in retraction_warnings:
                    st.warning(w)

            st.write("#### 逐条论断-来源文献核验明细")
            claims = rep.get("claims", [])
            if claims:
                for idx, c in enumerate(claims, 1):
                    status = c.get("verification_status", "UNVERIFIED")
                    badge = "✅ 通过 (SUPPORTED)" if status == "SUPPORTED" else f"⚠️ 告警 ({status})"
                    with st.expander(f"论断 [{c.get('claim_id')}] : {badge} - {c.get('statement')[:60]}..."):
                        st.markdown(f"**完整论断语句**：\n> {c.get('statement')}")
                        st.markdown(f"**声称引用的 PMID**：`{c.get('cited_pmids')}`")
                        st.markdown(f"**有效支持的 PMID**：`{c.get('supported_pmids')}`")
                        if c.get("unsupported_pmids"):
                            st.error(f"**未予支持的虚假/无效引用**：`{c.get('unsupported_pmids')}`")
                        st.info(f"**核验说明 / 摘要对齐解析**：\n{c.get('explanation')}")
            else:
                st.info("未检测到单独声明的原子论断。")

        # Tab 3: Evidence Level Grouped Cards
        with tab_evidence:
            st.subheader("🔬 按 Oxford CEBM 证据等级分组的文献")
            level_groups = {
                "Level 1: 系统评价 / Meta 分析": [p for p in papers if "LEVEL_1" in str(p.get("evidence_level", ""))],
                "Level 2: 随机对照试验 (RCT)": [p for p in papers if "LEVEL_2" in str(p.get("evidence_level", ""))],
                "Level 3-4: 观察性/队列研究": [p for p in papers if any(k in str(p.get("evidence_level", "")) for k in ["LEVEL_3", "LEVEL_4"])],
                "Level 5: 个案报告/叙述性综述/机理": [p for p in papers if "LEVEL_5" in str(p.get("evidence_level", ""))],
            }

            for group_name, p_list in level_groups.items():
                st.markdown(f"#### {group_name} ({len(p_list)} 篇)")
                if not p_list:
                    st.caption("该等级暂无入选文献")
                    continue

                for p in p_list:
                    with st.container(border=True):
                        col_t, col_a = st.columns([4, 1])
                        with col_t:
                            if p.get("retracted"):
                                st.error(f"⚠️ 该文献已被撤稿 (Retracted) — {p.get('retraction_note', 'PubMed 标记')}")
                            st.markdown(f"**{p.get('title')}**")
                            st.caption(f"📖 {p.get('journal', 'Unknown')} ({p.get('pub_year', 'N/A')}) | 作者: {', '.join(p.get('authors', [])[:3])}")
                        with col_a:
                            st.markdown(f"**PMID**: [{p.get('pmid')}](https://pubmed.ncbi.nlm.nih.gov/{p.get('pmid')}/)")
                            if p.get("citation_count", 0) > 0:
                                st.caption(f"🔥 被引数: **{p.get('citation_count')}**")
                            if p.get("open_access_pdf"):
                                st.link_button("📄 PDF 全文", p.get("open_access_pdf"))

                        with st.expander("查看文献摘要与分级评语"):
                            st.markdown(f"**分级理由**：`{p.get('evidence_rationale', 'CEBM分级标准')}`")
                            st.write(p.get("abstract", "暂无可用摘要"))

        # Tab 4: Reference Table
        with tab_raw:
            st.subheader("📄 参考文献标准化清单")
            table_data = []
            for idx, p in enumerate(papers, 1):
                level_str = p.get("evidence_level", "N/A")
                if isinstance(level_str, str) and "(" in level_str:
                    level_str = level_str.split("(")[0].strip()
                table_data.append({
                    "序号": f"[{idx}]",
                    "PMID": p.get("pmid"),
                    "证据等级": level_str,
                    "研究设计": p.get("study_type", "未知"),
                    "文献标题": p.get("title"),
                    "期刊": p.get("journal"),
                    "年份": p.get("pub_year"),
                    "被引数": p.get("citation_count", 0),
                    "撤稿状态": "⚠️ 已撤稿" if p.get("retracted") else "正常",
                    "PubMed链接": f"https://pubmed.ncbi.nlm.nih.gov/{p.get('pmid')}/",
                })
            st.dataframe(table_data, use_container_width=True)


if __name__ == "__main__":
    main()
