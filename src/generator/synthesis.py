"""
Review Synthesis Engine with Citation Traceability and Hallucination Control
Generates structured evidence reviews where every claim is strictly anchored to a PMID.
Supports LLM natural language polishing with automatic fallback to rule template.
"""

import re
import logging
from typing import List, Tuple, Optional
from src.models import Paper, PICOElements, AtomicClaim, EvidenceLevel
from src.llm.client import LLMClient

logger = logging.getLogger(__name__)


class ReviewSynthesizer:
    """Synthesizes evidence-based clinical literature reviews with inline PMIDs."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def synthesize(self, query: str, pico: PICOElements, papers: List[Paper]) -> Tuple[str, List[AtomicClaim]]:
        """Generate structured review text and extract atomic claims with citations."""
        if not papers:
            empty_msg = (
                "### 检索结果与证据不足提示\n\n"
                "**幻觉控制防御触发**：未检索到与提问相关的确切文献证据。根据「无来源不生成」原则，系统拒绝臆造结论。"
            )
            return empty_msg, []

        # Attempt LLM polishing if available
        if self.llm_client.is_available:
            llm_result = self._synthesize_with_llm(query, pico, papers)
            if llm_result:
                return llm_result

        # Fallback to deterministic structured template
        return self._synthesize_template(query, pico, papers)

    def _synthesize_with_llm(self, query: str, pico: PICOElements, papers: List[Paper]) -> Optional[Tuple[str, List[AtomicClaim]]]:
        """Generate polished review using LLM while enforcing strict citation anchors."""
        lit_context = []
        for idx, p in enumerate(papers, 1):
            level_str = p.evidence_level.value if p.evidence_level else "Unknown"
            type_str = p.study_type.value if p.study_type else "Unknown"
            lit_context.append(
                f"[{idx}] PMID: {p.pmid}\n"
                f"    标题: {p.title}\n"
                f"    期刊与年份: {p.journal} ({p.pub_year})\n"
                f"    设计与分级: {type_str} | {level_str}\n"
                f"    摘要核心: {p.abstract[:600]}\n"
            )
        literature_str = "\n".join(lit_context)

        system_prompt = (
            "你是一名资深循证医学与临床流行病学专家。\n"
            "你的任务是根据医生提供的临床问题及已检索分级的医学文献，撰写结构严密、每句论述均带来源引用的综述报告。\n\n"
            "【极端重要的防幻觉与引用准则】：\n"
            "1. 坚守「无来源不生成（No Citation, No Claim）」原则，严禁臆造文献、结论或超出摘要的事实。\n"
            "2. 每一个核心疗效结论、终点数据、亚组分析结论后，必须标注引用标记 [PMID:xxxx]。\n"
            "3. 保持客观中立，明确指出研究局限性与证据不足之处。\n"
            "4. 输出格式为标准 Markdown，并包含以下四个固定板块：\n"
            "   # 医学文献研究综述：<问题>\n"
            "   > 临床 PICO 结构解析...\n"
            "   ## 1. 核心临床结论 (Executive Clinical Summary)\n"
            "   ## 2. 循证医学证据分级详述 (Evidence Synthesis by Hierarchy)\n"
            "   ## 3. 证据冲突、亚组异质性与局限性 (Heterogeneity & Limitations)\n"
            "   ## 4. 参考文献溯源清单 (Traceable References)（含 Markdown 表格）\n"
        )

        user_prompt = (
            f"临床提问: {query}\n\n"
            f"PICO 要素:\n"
            f"- 人群: {pico.population}\n"
            f"- 干预: {pico.intervention}\n"
            f"- 对照: {pico.comparison or '安慰剂 / 标准治疗'}\n"
            f"- 终点: {pico.outcome or '主要心血管预后或临床结局'}\n\n"
            f"已检索文献列表:\n{literature_str}\n\n"
            "请按要求撰写循证综述："
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.llm_client.chat_completion(messages, temperature=0.1, max_tokens=2000)
        if not response or "[PMID:" not in response:
            return None

        # Extract claims from LLM response
        claims: List[AtomicClaim] = []
        counter = 1
        for line in response.split("\n"):
            line_str = line.strip().lstrip("-").lstrip("*").strip()
            if not line_str or line_str.startswith("#") or line_str.startswith("|"):
                continue
            pmids = re.findall(r"\[PMID:(\d+)\]", line_str)
            if pmids:
                claims.append(
                    AtomicClaim(
                        claim_id=f"C{counter}",
                        statement=line_str,
                        cited_pmids=list(dict.fromkeys(pmids)),
                    )
                )
                counter += 1

        if not claims:
            return None

        return response, claims

    def _synthesize_template(self, query: str, pico: PICOElements, papers: List[Paper]) -> Tuple[str, List[AtomicClaim]]:
        """Fallback deterministic template synthesizer.

        Extractive by design: every claim is built from the cited paper's own
        title/abstract, so the verifier can genuinely check it. No clinical
        efficacy statements are invented beyond what the sources state.
        """
        claims: List[AtomicClaim] = []
        claim_counter = 1

        def level_key(p: Paper) -> int:
            name = getattr(p.evidence_level, "name", "") or ""
            for i in range(1, 6):
                if f"LEVEL_{i}" in name:
                    return i
            return 9

        groups: dict = {}
        for p in papers:
            groups.setdefault(level_key(p), []).append(p)

        review_parts = []
        review_parts.append(f"# 医学文献研究综述：{query}\n")
        review_parts.append(
            f"> **临床 PICO 结构解析**：\n"
            f"> - **人群 (P)**: {pico.population}\n"
            f"> - **干预 (I)**: {pico.intervention}\n"
            f"> - **对照 (C)**: {pico.comparison or '安慰剂 / 常规标准治疗'}\n"
            f"> - **终点 (O)**: {pico.outcome or '心血管结局、死亡率或主要预后指标'}\n"
        )

        # 1. 证据概览：只描述检索到的证据基线，不编造疗效结论
        review_parts.append("## 1. 证据概览 (Evidence Landscape)\n")
        dist = ", ".join(f"Level {lv} × {len(ps)}" for lv, ps in sorted(groups.items()) if lv <= 5)
        top_level = min([lv for lv in groups if lv <= 5], default=5)
        review_parts.append(
            f"- 本次共纳入 **{len(papers)}** 篇文献，证据等级分布：{dist or '未分级'}；"
            f"其中最高等级为 **Level {top_level}**。\n"
            "- 以下每条结论均直接转述自所引文献的题目与摘要原文，结论后的 [PMID:xxxx] 为其唯一来源；"
            "凡无来源支撑的推断，本综述一律不写（「无来源不生成」）。\n"
        )

        # 2. 分级证据详述：逐篇抽取
        review_parts.append("## 2. 循证医学证据分级详述 (Evidence Synthesis by Hierarchy)\n")
        section_titles = {
            1: "### 2.1 Level 1 最高等级证据：系统评价与 Meta 分析",
            2: "### 2.2 Level 2 高等级证据：随机对照试验 (RCT)",
            3: "### 2.3 Level 3 中等等级证据：队列研究",
            4: "### 2.4 Level 4 低等级证据：病例对照研究",
            5: "### 2.5 Level 5 最低等级证据：个案报告 / 综述 / 机制研究",
        }
        for lv in sorted(groups):
            if lv > 5:
                continue
            review_parts.append(section_titles.get(lv, f"### Level {lv}"))
            for p in groups[lv]:
                excerpt = self._abstract_excerpt(p.abstract)
                type_str = p.study_type.value if p.study_type else "未知设计"
                if excerpt:
                    stmt = f"【{type_str}｜《{p.journal}》({p.pub_year})】{p.title}。{excerpt} [PMID:{p.pmid}]"
                else:
                    stmt = f"【{type_str}｜《{p.journal}》({p.pub_year})】{p.title}（本条仅有题录信息，摘要缺失，未做内容转述） [PMID:{p.pmid}]"
                review_parts.append(f"- {stmt}")
                claims.append(AtomicClaim(claim_id=f"C{claim_counter}", statement=stmt, cited_pmids=[p.pmid]))
                claim_counter += 1
            review_parts.append("")

        # 3. 局限性：只谈方法学层面的注意事项，不编造亚组结论
        review_parts.append("## 3. 证据局限性与使用提示 (Limitations)\n")
        review_parts.append(
            "- 本综述基于本次检索到的有限文献摘要转述，未做全文偏倚风险评估（如随机化隐藏、失访率），证据强度解读需谨慎。\n"
            "- 各研究在人群特征、干预剂量与随访时长上存在异质性，结论不宜直接外推至未覆盖人群。\n"
            "- 文献检索存在发表偏倚可能；临床决策请结合最新指南与患者个体情况，并咨询专科医生。\n"
        )

        # 4. 参考文献清单（支持点击溯源）
        review_parts.append("## 4. 参考文献溯源清单 (Traceable References)\n")
        review_parts.append("| 序号 | 证据等级 | 研究设计 | 文献标题 | 期刊 (年份) | 被引数 | 溯源与全文 |")
        review_parts.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for idx, p in enumerate(papers, 1):
            level_str = p.evidence_level.value.split()[1] if p.evidence_level else "N/A"
            type_str = p.study_type.value if p.study_type else "未知"
            clean_title = p.title.replace("|", " ")
            cites = f"{p.citation_count}" if p.citation_count > 0 else "-"
            pdf_link = f" · [📄PDF]({p.open_access_pdf})" if p.open_access_pdf else ""
            review_parts.append(
                f"| [{idx}] | Level {level_str} | {type_str} | {clean_title} | {p.journal} ({p.pub_year}) | {cites} | [{p.pmid}]({p.pubmed_url}){pdf_link} |"
            )

        full_review = "\n".join(review_parts)
        return full_review, claims

    @staticmethod
    def _abstract_excerpt(abstract: str, max_chars: int = 420) -> str:
        """Take the first ~2 sentences of an abstract as a faithful excerpt."""
        text = (abstract or "").strip()
        if not text:
            return ""
        # Split into sentences on '. ' / '。'
        sentences = re.split(r"(?<=[.!?。])\s+", text)
        excerpt = " ".join(sentences[:2]).strip()
        if len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars].rstrip() + "…"
        return excerpt
