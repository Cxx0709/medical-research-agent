"""
Mock Medical Literature Retriever with High-Fidelity Clinical Papers
Provides realistic peer-reviewed medical publications across different evidence tiers
to allow full-loop local execution without API rate limits or network dependencies.
"""

from typing import List
from src.models import Paper, StudyType, EvidenceLevel
from src.retriever.base import BaseRetriever

MOCK_DATABASE = [
    Paper(
        pmid="36049247",
        title="SGLT2 inhibitors in heart failure with mildly reduced or preserved ejection fraction: a systematic review and meta-analysis of the DELIVER and EMPEROR-Preserved trials",
        abstract=(
            "Background: Whether sodium-glucose cotransporter 2 (SGLT2) inhibitors improve cardiovascular outcomes "
            "across the spectrum of patients with heart failure with mildly reduced or preserved ejection fraction (HFmrEF/HFpEF) "
            "remains uncertain. Methods: We conducted a systematic review and meta-analysis including data from 12,251 participants "
            "across the DELIVER and EMPEROR-Preserved randomized trials. The primary endpoint was composite cardiovascular death or first hospitalization for heart failure. "
            "Findings: SGLT2 inhibitors significantly reduced composite cardiovascular death or first hospitalization for heart failure "
            "(HR 0.80, 95% CI 0.73-0.87; p<0.0001). Benefits were consistent across predefined subgroups, including patients with or without diabetes "
            "and across ranges of left ventricular ejection fraction. All-cause mortality was also reduced (HR 0.92, 95% CI 0.86-0.99; p=0.026). "
            "Interpretation: SGLT2 inhibitors significantly reduce the risks of cardiovascular death and hospital admission in patients with HFpEF."
        ),
        journal="The Lancet",
        pub_year=2022,
        authors=["Vaduganathan M", "Docherty KF", "Claggett BL", "McMurray JJV", "Solomon SD"],
        doi="10.1016/S0140-6736(22)01429-5",
        mesh_terms=["Sodium-Glucose Transporter 2 Inhibitors", "Heart Failure", "Meta-Analysis", "Stroke Volume"],
        publication_types=["Systematic Review", "Meta-Analysis", "Journal Article"],
    ),
    Paper(
        pmid="36049248",
        title="Dapagliflozin in Heart Failure with Mildly Reduced or Preserved Ejection Fraction",
        abstract=(
            "Background: SGLT2 inhibitors reduce the risk of worsening heart failure or cardiovascular death in patients with heart failure "
            "and a reduced ejection fraction. However, their efficacy in patients with heart failure and a preserved ejection fraction is less clear. "
            "Methods: In this multinational, randomized, double-blind, placebo-controlled trial, we randomly assigned 6263 patients with heart failure "
            "and a left ventricular ejection fraction of greater than 40% to receive either dapagliflozin (at a dose of 10 mg once daily) or matching placebo. "
            "Results: Over a median follow-up of 2.3 years, the primary outcome (worsening heart failure or cardiovascular death) occurred in 512 of 3131 patients (16.4%) "
            "in the dapagliflozin group and in 610 of 3132 patients (19.5%) in the placebo group (HR 0.82; 95% CI 0.73 to 0.92; P<0.001). "
            "Worsening heart failure events occurred in 11.8% in the dapagliflozin group and 14.5% in the placebo group (HR 0.79; 95% CI 0.69 to 0.91). "
            "Adverse events were similar in both groups. Conclusions: Dapagliflozin reduced the combined risk of worsening heart failure or cardiovascular death in HFpEF."
        ),
        journal="New England Journal of Medicine",
        pub_year=2022,
        authors=["Solomon SD", "McMurray JJV", "Claggett B", "de Boer RA", "DeMets D"],
        doi="10.1056/NEJMoa2206986",
        mesh_terms=["Sodium-Glucose Transporter 2 Inhibitors", "Heart Failure", "Randomized Controlled Trial", "Dapagliflozin"],
        publication_types=["Randomized Controlled Trial", "Clinical Trial, Phase III", "Journal Article"],
    ),
    Paper(
        pmid="34449189",
        title="Empagliflozin in Heart Failure with a Preserved Ejection Fraction",
        abstract=(
            "Background: Sodium-glucose cotransporter 2 inhibitors reduce the risk of hospitalization for heart failure in patients with heart failure "
            "and a reduced ejection fraction, but their effects in patients with a preserved ejection fraction remain to be clarified. "
            "Methods: In this double-blind trial, we randomly assigned 5988 patients with class II-IV heart failure and an ejection fraction of >40% "
            "to receive empagliflozin (10 mg once daily) or placebo. The primary outcome was a composite of cardiovascular death or hospitalization for heart failure. "
            "Results: Over a median of 26.2 months, a primary outcome event occurred in 415 of 2997 patients (13.8%) in the empagliflozin group and in 511 of 2991 patients (17.1%) "
            "in the placebo group (hazard ratio 0.79; 95% CI 0.69 to 0.90; P<0.001). This effect was primarily related to a lower risk of hospitalization for heart failure. "
            "Conclusions: Empagliflozin significantly reduced the combined risk of cardiovascular death or hospitalization for heart failure in patients with HFpEF."
        ),
        journal="New England Journal of Medicine",
        pub_year=2021,
        authors=["Anker SD", "Butler J", "Filippatos G", "Ferreira JP", "Bocchi E"],
        doi="10.1056/NEJMoa2107038",
        mesh_terms=["Empagliflozin", "Heart Failure", "Randomized Controlled Trial", "Preserved Ejection Fraction"],
        publication_types=["Randomized Controlled Trial", "Clinical Trial, Phase III", "Journal Article"],
    ),
    Paper(
        pmid="36812850",
        title="Real-world effectiveness and safety of SGLT2 inhibitors in patients with heart failure with preserved ejection fraction: A prospective multicenter cohort study",
        abstract=(
            "Background: Clinical trials have demonstrated efficacy of SGLT2i in HFpEF, but real-world observational data in diverse unselected populations are limited. "
            "Methods: We prospectively observed 2,450 consecutive patients with clinical HFpEF across 8 tertiary centers between 2021 and 2023. "
            "Patients initiating SGLT2i (n=1,210) were compared with non-users (n=1,240) using propensity score matching. "
            "Results: SGLT2i initiation was associated with lower 1-year heart failure rehospitalization rates (14.2% vs 19.8%, HR 0.74, 95% CI 0.61-0.90; p=0.002). "
            "Renal function decline was slower in the SGLT2i group. Urinary tract infection rates were mildly higher in the SGLT2i cohort (4.8% vs 2.9%, p=0.03). "
            "Conclusions: In routine clinical practice, SGLT2i use is associated with reduced heart failure hospitalizations in HFpEF patients, confirming trial findings."
        ),
        journal="Circulation: Heart Failure",
        pub_year=2023,
        authors=["Zhang L", "Chen Y", "Wang H", "Kass DA", "Sharma K"],
        doi="10.1161/CIRCHEARTFAILURE.122.009876",
        mesh_terms=["Observational Study", "Cohort Studies", "Heart Failure", "SGLT2 Inhibitors"],
        publication_types=["Cohort Studies", "Observational Study", "Journal Article"],
    ),
    Paper(
        pmid="35912345",
        title="Euglycemic diabetic ketoacidosis triggered by severe gastroenteritis in a patient with HFpEF receiving dapagliflozin: A case report",
        abstract=(
            "Background: Euglycemic diabetic ketoacidosis (euDKA) is a recognized but rare complication of SGLT2 inhibitors. "
            "Case presentation: A 68-year-old female with HFpEF and well-controlled type 2 diabetes on dapagliflozin 10 mg presented with nausea, vomiting, "
            "and deep tachypnea after acute viral gastroenteritis. Blood glucose was 168 mg/dL, with severe metabolic acidosis (pH 7.12, HCO3- 8 mEq/L, anion gap 24) "
            "and marked ketonemia (beta-hydroxybutyrate 5.4 mmol/L). Dapagliflozin was discontinued, and intravenous insulin with dextrose infusion successfully resolved the ketoacidosis. "
            "Conclusions: Clinicians must remain vigilant for euglycemic DKA in HFpEF patients on SGLT2 inhibitors during acute illness, dehydration, or prolonged fasting."
        ),
        journal="Cardiovascular Diabetology",
        pub_year=2022,
        authors=["Martinez P", "Greenberg B", "Adams K"],
        doi="10.1186/s12933-022-01588-x",
        mesh_terms=["Diabetic Ketoacidosis", "Heart Failure", "Dapagliflozin", "Case Reports"],
        publication_types=["Case Reports", "Journal Article"],
    ),
    Paper(
        pmid="37952131",
        title="Semaglutide and Cardiovascular Outcomes in Obesity without Diabetes",
        abstract=(
            "Background: It is unknown whether semaglutide can reduce cardiovascular events in patients with overweight or obesity without diabetes. "
            "Methods: In a multicenter, double-blind, randomized, placebo-controlled event-driven trial, we enrolled 17,604 patients aged 45 years or older "
            "with preexisting cardiovascular disease and a BMI of 27 or greater, but no history of diabetes. Patients received once-weekly subcutaneous semaglutide (2.4 mg) or placebo. "
            "Results: The primary cardiovascular composite outcome (death from cardiovascular causes, nonfatal myocardial infarction, or nonfatal stroke) occurred in 569 of 8803 patients (6.5%) "
            "in the semaglutide group and in 701 of 8801 patients (8.0%) in the placebo group (HR 0.80; 95% CI 0.72 to 0.90; P<0.001). "
            "Mean weight loss was -9.4% in the semaglutide group versus -0.9% in the placebo group. Serious adverse events were reported in 33.4% and 36.4% respectively. "
            "Conclusions: In patients with preexisting cardiovascular disease and obesity without diabetes, weekly semaglutide 2.4 mg was superior to placebo in reducing MACE by 20%."
        ),
        journal="New England Journal of Medicine",
        pub_year=2023,
        authors=["Lincoff AM", "Brown-Frandsen K", "Colhoun HM", "Deanfield J", "Emerson SS"],
        doi="10.1056/NEJMoa2307563",
        mesh_terms=["Semaglutide", "Cardiovascular Diseases", "Obesity", "Randomized Controlled Trial"],
        publication_types=["Randomized Controlled Trial", "Clinical Trial, Phase III", "Journal Article"],
    ),
    Paper(
        pmid="34800366",
        title="Cardiovascular, mortality, and kidney outcomes with GLP-1 receptor agonists in patients with type 2 diabetes: a systematic review and meta-analysis of randomised trials",
        abstract=(
            "Background: GLP-1 receptor agonists have been evaluated in multiple cardiovascular outcome trials. We aimed to synthesize their overall effects. "
            "Methods: We did a systematic review and meta-analysis of randomized controlled trials examining GLP-1 receptor agonists. We searched PubMed and other databases up to June 2021. "
            "Findings: Eight trials enrolling 60,080 participants were included. Overall, GLP-1 receptor agonists reduced 3-point MACE by 14% (HR 0.86, 95% CI 0.80-0.93; p<0.0001), "
            "all-cause mortality by 12% (HR 0.88, 95% CI 0.82-0.94), and broad kidney outcome by 21% (HR 0.79, 95% CI 0.73-0.87). "
            "Interpretation: GLP-1 receptor agonists have robust cardiovascular and renal protective benefits across patients with type 2 diabetes."
        ),
        journal="The Lancet Diabetes & Endocrinology",
        pub_year=2021,
        authors=["Sattar N", "Lee MMY", "Kristensen SL", "Branch KRH", "Del Prato S"],
        doi="10.1016/S2213-8587(21)00203-5",
        mesh_terms=["Glucagon-Like Peptide-1 Receptor Agonists", "Diabetes Mellitus, Type 2", "Meta-Analysis"],
        publication_types=["Systematic Review", "Meta-Analysis", "Journal Article"],
    ),
]


class MockPubMedRetriever(BaseRetriever):
    """Local mock retriever serving pre-indexed peer-reviewed medical publications."""

    def __init__(self, papers: List[Paper] = None):
        self.papers = papers or MOCK_DATABASE

    def search(self, query: str, max_results: int = 5) -> List[Paper]:
        query_tokens = [w.lower().strip() for w in query.replace(",", " ").replace(";", " ").split() if len(w) > 2]
        
        # Keyword-based matching
        scored_papers = []
        for paper in self.papers:
            searchable_text = f"{paper.title} {paper.abstract} {' '.join(paper.mesh_terms)}".lower()
            match_count = sum(1 for token in query_tokens if token in searchable_text)
            
            # Special medical domain synonyms
            if any(term in query.lower() for term in ["sglt", "sglt2", "sglt-2", "列净", "心衰", "hfpef", "heart failure"]):
                if "sglt" in searchable_text or "heart failure" in searchable_text:
                    match_count += 3
            if any(term in query.lower() for term in ["glp", "glp-1", "semaglutide", "司美格鲁肽", "肥胖", "obesity"]):
                if "glp" in searchable_text or "semaglutide" in searchable_text or "obesity" in searchable_text:
                    match_count += 3

            score = match_count / (len(query_tokens) + 1e-5)
            paper_copy = paper.model_copy()
            paper_copy.relevance_score = round(score, 3)
            scored_papers.append((score, paper_copy))

        # Sort descending by score, take top results
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        results = [p for _, p in scored_papers if _ > 0][:max_results]
        
        # If no strict token match, return top papers as generic clinical fallbacks
        if not results:
            results = [p.model_copy() for p in self.papers[:max_results]]
            
        return results

    async def asearch(self, query: str, max_results: int = 5) -> List[Paper]:
        return self.search(query=query, max_results=max_results)
