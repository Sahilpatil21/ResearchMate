"""Research Intelligence package for ResearchMate.

Provides automated paper summarization, multi-paper comparison, evidence-grounded research gap analysis,
and structured literature review generation.
"""

from src.intelligence.comparator import PaperComparator, PaperComparisonReport
from src.intelligence.gap_analyzer import ResearchGapAnalyzer, ResearchGapItem, ResearchGapReport
from src.intelligence.lit_reviewer import LiteratureReviewGenerator, LiteratureReviewReport
from src.intelligence.summarizer import PaperSummarizer, PaperSummary

__all__ = [
    "PaperSummarizer",
    "PaperSummary",
    "PaperComparator",
    "PaperComparisonReport",
    "ResearchGapAnalyzer",
    "ResearchGapItem",
    "ResearchGapReport",
    "LiteratureReviewGenerator",
    "LiteratureReviewReport",
]
