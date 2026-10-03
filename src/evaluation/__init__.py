"""Evaluation and benchmarking framework for ResearchMate.

Provides benchmark datasets, retrieval metric calculators (Precision, Recall, Hit Rate, MRR),
generation evaluations, human rating scales, and experiment persistence.
"""

from src.evaluation.dataset import EvaluationDataset, EvaluationQuestion
from src.evaluation.experiment_logger import ExperimentLogger
from src.evaluation.generation_evaluator import AutomatedGenerationMetrics, GenerationEvaluator, HumanEvaluationRating
from src.evaluation.retrieval_evaluator import MethodEvaluationSummary, RetrievalEvaluationReport, RetrievalEvaluator

__all__ = [
    "EvaluationDataset",
    "EvaluationQuestion",
    "RetrievalEvaluator",
    "RetrievalEvaluationReport",
    "MethodEvaluationSummary",
    "GenerationEvaluator",
    "AutomatedGenerationMetrics",
    "HumanEvaluationRating",
    "ExperimentLogger",
]
