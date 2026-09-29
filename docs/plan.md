# 医学文献研究 Agent (Medical Research Agent) 架构与实施方案

> **核心设计哲学**：医疗是最看重「可验证性」与「确定性」的场景。本 Agent 坚持 **“无来源不生成（No Citation, No Claim）”** 的核心原则，通过可解释的证据分级、细粒度引用溯源与自反思幻觉核验，为临床医生与医学研究者提供可靠的文献综述。

---

## 1. 项目目标与核心卖点

1. **自然语言临床提问**：支持医生以自然语言（如：“SGLT-2抑制剂对心衰伴射血分数保留患者的预后改善证据如何？”）输入，自动提取临床核心要素（PICO）。
2. **多源医学文献检索**：接入 PubMed（E-Utilities）及 Semantic Scholar API，结合 MeSH 主题词扩展，兼备权威性与学术覆盖面。
3. **证据等级自动化标注**：参考牛津循证医学中心（CEBM）证据分级标准及 GRADE 框架，自动识别文献类型（系统评价/Meta分析、RCT、前瞻性队列、病例对照、病例报告、综述、动物/基础实验）并评定证据等级。
4. **细粒度引用溯源综述**：生成结构化综述（研究背景、主要发现、证据对照、争议与局限、结论），每个论断原子化绑定文献来源（PMID/DOI），支持溯源查看原文摘要与对应支持句。
5. **严密幻觉控制（Refusal & Verification Guardrail）**：
   - **引用回溯校验（Claim-Citation Entailment）**：LLM 裁判或蕴含分类器逐句核验生成的结论是否真实得到所引文献支持。
   - **无证据拒答/标注不确定性**：无可靠文献支撑时明确提示“暂无高等级证据支持”，绝不臆造文献与研究数据。

---

## 2. 总体架构与 LangGraph 流水线设计

系统采用 **LangGraph** 构建有向有环图（DAG + 验证回环），实现自适应检索、分级标注、综述合成与反思核验。

### 2.1 状态定义 (`MedicalResearchState`)

```python
class MedicalResearchState(TypedDict):
    # 输入与解析
    query: str                          # 医生原始提问
    pico: dict                          # PICO 临床要素分解 (Population, Intervention, Comparison, Outcome)
    search_keywords: list[str]          # 生成的 PubMed / Semantic Scholar 检索词式
    
    # 检索与初筛
    retrieved_papers: list[dict]        # 初检文献原始列表 (PMID, Title, Abstract, Journal, Year, Authors, DOI)
    ranked_papers: list[dict]           # 相关性重排与初筛后的 Top-K 文献
    
    # 证据等级与质控
    annotated_papers: list[dict]        # 包含证据等级 (Level 1~5)、研究设计分类、质量评估维度的文献
    
    # 综述与溯源
    draft_review: str                   # 综述草稿（包含引用标记如 [PMID:12345678]）
    claims_with_citations: list[dict]   # 提取的原子论断及其声称的引用
    
    # 幻觉校验与质检
    verification_report: dict           # 引用支持度评估报告 (Supported / Contradicted / Not Mentioned)
    final_review: str                   # 经过校准和清洗的最终交付综述
    references: list[dict]              # 引用的文献标准列表
    hallucination_detected: bool        # 是否发现严重幻觉并触发重新校准/修正
    iteration_count: int                # 反思循环轮数上限控制
```

### 2.2 LangGraph 工作流图

```mermaid
flowchart TD
    Start([用户自然语言临床提问]) --> ParseQuery[1. Query Analyzer / PICO 解析节点]
    ParseQuery --> Retrieve[2. Multi-Source Literature Retrieval 检索节点<br/>PubMed / Semantic Scholar]
    Retrieve --> CheckEmpty{是否有检索结果?}
    CheckEmpty -- 无结果 --> ExpandQuery[检索词扩展/放宽过滤条件]
    ExpandQuery --> Retrieve
    CheckEmpty -- 有结果 --> Rank[3. Relevance & Quality Ranker 重排节点]
    Rank --> AnnotateEvidence[4. Evidence Level Grading 证据等级标注节点<br/>CEBM分级/研究类型分类]
    AnnotateEvidence --> GenerateReview[5. Traceable Review Generator 溯源综述生成节点<br/>要求：每条论点必带 [PMID:xxx]]
    GenerateReview --> ExtractClaims[6. Claim & Citation Extraction 原子论断提取]
    ExtractClaims --> VerifyEntailment[7. Hallucination Verifier 引用蕴含核验节点]
    VerifyEntailment --> CheckHallucination{是否存在虚假引用/幻觉论断?}
    CheckHallucination -- 是 且 迭代数<上限 --> RefineReview[8. Self-Correction 综述校准修正]
    RefineReview --> VerifyEntailment
    CheckHallucination -- 否 或 达到迭代上限 --> Finalize[9. Final Output Assembly 格式化输出]
    Finalize --> End([交付终稿结构化综述 + 证据等级表 + 溯源索引])
```

---

## 3. 具体功能模块设计

### 模块 1：临床问题解析与检索模块 (Query Parser & Retrieval)
- **输入**：医生自然语言提问（支持中英双语）。
- **处理逻辑**：
  1. **PICO 要素抽取**：
     - $P$ (Patient/Population): 目标人群/疾病阶段。
     - $I$ (Intervention): 靶向治疗/药物/干预手段。
     - $C$ (Comparison): 安慰剂/标准疗法/不同剂量。
     - $O$ (Outcome): 临床硬终点（全因死亡、MACE、PFS）或替代终点。
  2. **医学检索式构建**：利用医学词表（MeSH）对关键词做同义词扩充，生成适配 PubMed E-utilities（`esearch` + `esummary` / `efetch`）以及 Semantic Scholar API 的检索词。
  3. **检索适配器体系**：
     - `PubMedRetriever`：官方 E-utilities，抓取 PMID、标题、结构化摘要（Background/Methods/Results/Conclusions）、出版年份、期刊影响因子/权威度。
     - `SemanticScholarRetriever`：获取学术引用数、DOI、开放获取全文/摘要。
     - `MockRetriever`：内置真实临床场景的精标离线文献数据，确保无网或无 API Key 场景下 100% 完整复现与测试。

### 模块 2：相关性排序与质量过滤 (Relevance Ranking & Filter)
- **目标**：从返回的数十篇文献中，挑选出与临床提问最匹配、时效性最高、研究质量最好的 Top-K 篇进入生成上下文。
- **排序打分机制**：
  $$Score = w_1 \cdot \text{SemanticSimilarity} + w_2 \cdot \text{StudyDesignWeight} + w_3 \cdot \text{RecencyDecay} + w_4 \cdot \text{PICO\_Overlap}$$
  - **语义相似度**：问题与文献摘要的语义嵌入余弦相似度或 BM25 词频匹配。
  - **研究类型先验权重**：Meta分析(1.0) > RCT(0.9) > 队列研究(0.7) > 病例对照(0.5) > 病例报告(0.3)。
  - **时效衰减**：近 3~5 年发表的文献优先。
  - **硬性过滤**：排除撤稿文献（Retracted）、过滤无摘要文献。

### 模块 3：证据等级标注模块 (Evidence Level Grading)
参考 Oxford CEBM（牛津循证医学中心）分级标准与 GRADE 原则：
- **Level 1（最高证据）**：高质量同质性随机对照试验（RCT）的系统评价与 Meta 分析。
- **Level 2**：高质量单中心/多中心双盲随机对照临床试验（RCT）。
- **Level 3**：前瞻性队列研究（Cohort Study）或结局研究。
- **Level 4**：病例对照研究（Case-Control Study）或回顾性病例系列分析（Case Series）。
- **Level 5（最低证据）**：个案报告（Case Report）、机理机制研究、动物实验或专家意见。
- **实现手段**：
  - **规则引擎 + 正则识别**：分析 Publication Types（PubMed MedlineTags 如 `Clinical Trial, Phase III`, `Systematic Review`, `Meta-Analysis`）。
  - **LLM/轻量分类器**：针对摘要中的 Methods 章节进行研究类型判别与偏倚风险（Risk of Bias）简评。

### 模块 4：引用溯源综述生成模块 (Citation-Traceable Synthesis)
- **写作规范**：
  - 强制采用结构化输出（分层阐述：【核心临床结论】、【分级证据详述】、【不同研究间的矛盾与争议】、【临床建议与局限性】）。
  - **句句有出处**：每个关键结论句尾必须附带标准引用锚点，格式形如 `[PMID:36528712]` 或 `[DOI:10.1056/NEJMoa...]`。
  - **无证据不结论**：若文献中未提及某维度（如对特定亚组的不良反应），严禁推断，必须标注“检索文献未涉及该亚组”。
- **引用格式**：
  - 综述正文行内标记：`...SGLT-2抑制剂使心血管死亡或心衰恶化风险降低18% [PMID:36049248]...`
  - 结尾提供标准参考文献索引，包含：PMID、标题、期刊、年份、证据等级、摘要核心结论高亮。

### 模块 5：幻觉控制与引用核验模块 (Hallucination Control & Entailment)
这是区别于一般知识库问答的核心壁垒：
1. **论断-引用拆解（Atomic Claim Decomposition）**：将生成的综述拆解为独立的原子论断（Atomic Claims），并关联其标注的 PMID。
2. **NLI（自然语言推理）文本蕴含验证**：
   - 前提（Premise）：对应文献的真实官方摘要。
   - 假设（Hypothesis）：Agent 生成的论述句子。
   - 判定三分类：
     - **Entailment（严格支持）**：保留。
     - **Contradiction（矛盾/篡改数据）**：严重告警，必须剔除或更正。
     - **Neutral / Not Mentioned（无提及/幻觉）**：剔除该引用，并标记该论述为无证据支持。
3. **自纠错修正回环**：将核验不通过的论断反馈给修正节点（RefineReview），重新基于检索文献事实进行精简修订。

---

## 4. 评测方案 (Evaluation Benchmark & Metrics)

为确保医学 Agent 的严谨性，建立专门的自动化与半自动化评测机制：

| 评估维度 | 指标名称 | 计算公式 / 评价准则 | 目标阈值 |
| :--- | :--- | :--- | :--- |
| **引用准确率** | **Citation Precision** | $\frac{\text{被引文献确实支持论断的引用数}}{\text{综述总引用标注数}} \times 100\%$ | $\ge 95\%$ |
| **引用召回率** | **Citation Coverage** | $\frac{\text{正文中包含有效引用的核心论断数}}{\text{正文总核心论断数}} \times 100\%$ | $100\%$ |
| **证据分级准确率**| **Evidence Grading F1** | 模型判定的证据等级（Level 1~5）与医学金标准标签的多分类加权 F1 值 | $\ge 90\%$ |
| **幻觉率** | **Hallucination Rate** | $\frac{\text{未被任何检索文献支持的论断数}}{\text{总论断数}} \times 100\%$ | $\le 2\%$ |
| **检索相关度** | **MRR / Precision@5** | Top-5 检索结果中满足临床提问核心意图的文献比例 | $\ge 80\%$ |

- **基准测试集（Evaluation Dataset）**：
  - 构建包含 50+ 个典型临床场景问答金标准集（涵盖心内科、肿瘤科、内分泌科等常见争议与指南问题）。
  - 每条数据包含：问题、标准 PICO、推荐检索 PMID 列表、权威指南结论摘要。

---

## 5. 最小可运行版本（MVP）实施计划

本次实施分步走，当前立即落实 **MVP 极简全链路**：
- **阶段一（当前）**：
  1. 定义 `MedicalResearchState` 数据模型与核心实体（`Paper`, `EvidenceLevel`, `AtomicClaim` 等）。
  2. 实现检索接口抽象，内置 `MockPubMedRetriever`（包含高保真经典临床研究数据）及标准 `PubMedRetriever` 接口。
  3. 实现证据等级规则解析器（识别 Meta/RCT/Review/Case Report 等）。
  4. 搭建 LangGraph 有向图（StateGraph）：`query_analysis -> retrieve -> rank_filter -> annotate -> generate_review -> verify_citations -> finalize`。
  5. 编写端到端运行脚本与自动化单元测试，确保一行命令跑通全链路。
- **阶段二（后续扩展）**：
  1. 完善 PubMed E-Utilities 真实 HTTP 异步抓取与解析。
  2. 接入真实 LLM（如 OpenAI/Claude/Gemini/DeepSeek 兼容 API）驱动的生成与 NLI 蕴含核验。
  3. 提供 Streamlit / CLI 交互式临床综述溯源界面。
