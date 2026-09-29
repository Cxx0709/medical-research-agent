"""
Unit tests for Evidence Level Grader (Oxford CEBM criteria)
"""

import pytest
from src.models import Paper, StudyType, EvidenceLevel
from src.evidence.grader import EvidenceGrader


def test_meta_analysis_grading():
    grader = EvidenceGrader()
    paper = Paper(
        pmid="12345",
        title="Efficacy of drug X: a systematic review and meta-analysis of randomized trials",
        abstract="We performed a systematic review and meta-analysis of 10 trials...",
        publication_types=["Systematic Review", "Meta-Analysis"]
    )
    graded = grader.grade(paper)
    assert graded.study_type == StudyType.SYSTEMATIC_REVIEW_META_ANALYSIS
    assert graded.evidence_level == EvidenceLevel.LEVEL_1


def test_rct_grading():
    grader = EvidenceGrader()
    paper = Paper(
        pmid="23456",
        title="Drug X versus placebo in patients with condition Y: a randomized controlled trial",
        abstract="In this double-blind, randomized, placebo-controlled trial...",
        publication_types=["Randomized Controlled Trial"]
    )
    graded = grader.grade(paper)
    assert graded.study_type == StudyType.RANDOMIZED_CONTROLLED_TRIAL
    assert graded.evidence_level == EvidenceLevel.LEVEL_2


def test_cohort_grading():
    grader = EvidenceGrader()
    paper = Paper(
        pmid="34567",
        title="Long-term outcomes of drug X: a prospective multicenter cohort study",
        abstract="We enrolled 1000 patients in this prospective cohort study...",
        publication_types=["Cohort Studies"]
    )
    graded = grader.grade(paper)
    assert graded.study_type == StudyType.COHORT_STUDY
    assert graded.evidence_level == EvidenceLevel.LEVEL_3


def test_case_report_grading():
    grader = EvidenceGrader()
    paper = Paper(
        pmid="45678",
        title="Severe adverse event after drug X: a case report and review of literature",
        abstract="We report a case of a 55-year-old patient who developed...",
        publication_types=["Case Reports"]
    )
    graded = grader.grade(paper)
    assert graded.study_type == StudyType.CASE_REPORT
    assert graded.evidence_level == EvidenceLevel.LEVEL_5
