"""
Evidence Level Grader based on Oxford CEBM & GRADE Hierarchies
Automatically classifies medical study designs and determines evidence hierarchy levels.
"""

import re
from typing import List, Tuple
from src.models import Paper, StudyType, EvidenceLevel


class EvidenceGrader:
    """Classifies study design and grades evidence level according to Oxford CEBM criteria."""

    STUDY_TYPE_PRIORS = {
        StudyType.SYSTEMATIC_REVIEW_META_ANALYSIS: EvidenceLevel.LEVEL_1,
        StudyType.RANDOMIZED_CONTROLLED_TRIAL: EvidenceLevel.LEVEL_2,
        StudyType.COHORT_STUDY: EvidenceLevel.LEVEL_3,
        StudyType.CASE_CONTROL_STUDY: EvidenceLevel.LEVEL_4,
        StudyType.CASE_REPORT: EvidenceLevel.LEVEL_5,
        StudyType.NARRATIVE_REVIEW: EvidenceLevel.LEVEL_5,
        StudyType.PRECLINICAL: EvidenceLevel.LEVEL_5,
        StudyType.UNKNOWN: EvidenceLevel.LEVEL_4,
    }

    def grade(self, paper: Paper) -> Paper:
        """Assign study type and evidence level to a paper."""
        study_type, rationale = self._classify_study_type(paper)
        evidence_level = self.STUDY_TYPE_PRIORS.get(study_type, EvidenceLevel.LEVEL_5)

        annotated = paper.model_copy()
        annotated.study_type = study_type
        annotated.evidence_level = evidence_level
        annotated.evidence_rationale = rationale
        return annotated

    def grade_batch(self, papers: List[Paper]) -> List[Paper]:
        """Grade a list of papers."""
        return [self.grade(p) for p in papers]

    def _classify_study_type(self, paper: Paper) -> Tuple[StudyType, str]:
        title = paper.title.lower()
        abstract = paper.abstract.lower()
        pub_types = [pt.lower() for pt in paper.publication_types]
        combined_text = f"{title} {abstract}"

        # 1. Systematic Review & Meta-Analysis (Level 1)
        if any("meta-analysis" in pt or "systematic review" in pt for pt in pub_types):
            return StudyType.SYSTEMATIC_REVIEW_META_ANALYSIS, "PubMed Medline publication type lists Systematic Review/Meta-Analysis"
        if re.search(r"\b(meta-analys(is|es)|systematic review|pooled analysis)\b", title):
            return StudyType.SYSTEMATIC_REVIEW_META_ANALYSIS, "Title explicitly specifies Systematic Review or Meta-Analysis"
        if "systematic review" in abstract and "meta-analysis" in abstract:
            return StudyType.SYSTEMATIC_REVIEW_META_ANALYSIS, "Abstract methods indicate systematic review and meta-analytic synthesis"

        # 2. Randomized Controlled Trial (Level 2)
        if any("randomized controlled trial" in pt for pt in pub_types):
            return StudyType.RANDOMIZED_CONTROLLED_TRIAL, "PubMed Medline publication type indicates Randomized Controlled Trial"
        if re.search(r"\b(randomized|randomised|double-blind|placebo-controlled)\b", title):
            return StudyType.RANDOMIZED_CONTROLLED_TRIAL, "Title confirms randomized clinical trial design"
        if re.search(r"\b(randomly assigned|double-blind.*placebo|phase (ii|iii|2|3) trial)\b", abstract):
            return StudyType.RANDOMIZED_CONTROLLED_TRIAL, "Abstract methods describe randomized, double-blind or phase 2/3 clinical trial"

        # 3. Cohort Study (Level 3)
        if any("cohort" in pt for pt in pub_types):
            return StudyType.COHORT_STUDY, "PubMed publication type marks Cohort Study"
        if re.search(r"\b(cohort study|prospective cohort|retrospective cohort|longitudinal study)\b", combined_text):
            return StudyType.COHORT_STUDY, "Study design identified as observational prospective/retrospective cohort"

        # 4. Case-Control Study (Level 4)
        if any("case-control" in pt for pt in pub_types):
            return StudyType.CASE_CONTROL_STUDY, "PubMed publication type marks Case-Control Study"
        if re.search(r"\b(case-control|matched controls)\b", combined_text):
            return StudyType.CASE_CONTROL_STUDY, "Study design identified as Case-Control design"

        # 5. Case Report / Case Series (Level 5)
        if any("case reports" in pt for pt in pub_types):
            return StudyType.CASE_REPORT, "Publication type confirms Case Report"
        if re.search(r"\b(case report|case presentation|a case of)\b", title):
            return StudyType.CASE_REPORT, "Title indicates single or case series clinical presentation"

        # 6. Narrative Review (Level 5)
        if any("review" in pt for pt in pub_types):
            return StudyType.NARRATIVE_REVIEW, "General narrative literature review without meta-analytic pooling"

        # 7. Preclinical / Animal Study (Level 5)
        if re.search(r"\b(mice|murine|rat|in vitro|cell culture|animal model)\b", combined_text) and not any("clinical trial" in pt for pt in pub_types):
            return StudyType.PRECLINICAL, "Preclinical laboratory or animal model experiment"

        return StudyType.UNKNOWN, "Study design not definitively identified from available title and abstract"
