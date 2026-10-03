"""Evaluation Dataset Management for ResearchMate.

Provides a structured 20+ question benchmark dataset spanning 7 categories:
- factual
- methodology
- dataset
- results
- limitations
- comparison
- multi-paper

Each item includes ground-truth reference answers, relevant document filenames, and target pages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.utils.file_utils import ensure_directories, get_evaluation_dir


class EvaluationQuestion(BaseModel):
    """A single evaluation benchmark question with ground-truth metadata."""

    question_id: str = Field(..., description="Unique question identifier, e.g. 'eval_001'")
    question: str = Field(..., description="Research question text")
    category: str = Field(..., description="Category: factual, methodology, dataset, results, limitations, comparison, multi-paper")
    expected_answer: str = Field(..., description="Ground-truth reference answer based on paper text")
    relevant_documents: List[str] = Field(..., description="Relevant PDF filenames or document IDs")
    relevant_pages: List[int] = Field(default_factory=list, description="Target PDF page numbers containing evidence")
    relevant_keywords: List[str] = Field(default_factory=list, description="Key technical terms expected in relevant chunks")


DEFAULT_EVALUATION_QUESTIONS: List[Dict[str, Any]] = [
    # --- Category 1: Factual ---
    {
        "question_id": "eval_001",
        "question": "What are the major branches or subfields of Artificial Intelligence identified in the AI overview paper?",
        "category": "factual",
        "expected_answer": "The paper outlines key branches of AI including Natural Language Processing (NLP), Neural Networks, Expert Systems, Robotics, and Fuzzy Logic systems.",
        "relevant_documents": ["IJRTI2304061.pdf"],
        "relevant_pages": [1, 2, 3],
        "relevant_keywords": ["Natural Language Processing", "Neural Networks", "Expert Systems", "Robotics", "Fuzzy Logic"],
    },
    {
        "question_id": "eval_002",
        "question": "How is the Turing Test described in the AI applications research paper?",
        "category": "factual",
        "expected_answer": "The Turing Test, developed by Alan Turing, tests a machine's ability to exhibit intelligent behavior indistinguishable from that of a human.",
        "relevant_documents": ["IJRTI2304061.pdf"],
        "relevant_pages": [2, 3],
        "relevant_keywords": ["Turing Test", "Alan Turing", "intelligent behavior", "imitation"],
    },
    {
        "question_id": "eval_003",
        "question": "What is the primary definition of Artificial Intelligence stated in the IJRTI publication?",
        "category": "factual",
        "expected_answer": "Artificial Intelligence is defined as the simulation of human intelligence processes by machines, especially computer systems, including learning, reasoning, and self-correction.",
        "relevant_documents": ["IJRTI2304061.pdf"],
        "relevant_pages": [1],
        "relevant_keywords": ["simulation", "human intelligence", "learning", "reasoning", "computer systems"],
    },

    # --- Category 2: Methodology ---
    {
        "question_id": "eval_004",
        "question": "What machine learning classification algorithms were evaluated in the student academic performance study?",
        "category": "methodology",
        "expected_answer": "The study applies and compares machine learning classifiers such as Decision Trees, Naive Bayes, Support Vector Machines (SVM), and Random Forest for student grade prediction.",
        "relevant_documents": ["106-ArticleText-199-1-10-20211225.pdf"],
        "relevant_pages": [2, 3, 4],
        "relevant_keywords": ["Decision Tree", "Naive Bayes", "SVM", "Random Forest", "classification"],
    },
    {
        "question_id": "eval_005",
        "question": "What data preprocessing and feature selection techniques were applied before training the classification model?",
        "category": "methodology",
        "expected_answer": "Data cleaning, missing value imputation, categorical encoding, normalization, and feature selection based on student academic attributes were performed.",
        "relevant_documents": ["106-ArticleText-199-1-10-20211225.pdf"],
        "relevant_pages": [2, 3],
        "relevant_keywords": ["preprocessing", "normalization", "encoding", "feature selection", "attributes"],
    },
    {
        "question_id": "eval_006",
        "question": "How does the Department of Education report define the 'Human-in-the-Loop' policy framework for educational AI?",
        "category": "methodology",
        "expected_answer": "The report emphasizes keeping educators and teachers in the loop by ensuring AI assists teachers rather than replacing pedagogical decision-making.",
        "relevant_documents": ["ai-report.pdf"],
        "relevant_pages": [4, 12, 18],
        "relevant_keywords": ["Human-in-the-Loop", "educators", "pedagogical", "teachers", "decision-making"],
    },

    # --- Category 3: Dataset ---
    {
        "question_id": "eval_007",
        "question": "What student attributes and academic metrics were collected in the student performance dataset?",
        "category": "dataset",
        "expected_answer": "Attributes included past grades, attendance, assignment submissions, socioeconomic factors, and extracurricular participation metrics.",
        "relevant_documents": ["106-ArticleText-199-1-10-20211225.pdf"],
        "relevant_pages": [2, 3],
        "relevant_keywords": ["grades", "attendance", "submissions", "attributes", "socioeconomic"],
    },
    {
        "question_id": "eval_008",
        "question": "What demographic and institutional data sources are examined in the US Department of Education AI report?",
        "category": "dataset",
        "expected_answer": "The report examines nationwide K-12 and higher education learning management data, student demographic indicators, and state longitudinal data systems.",
        "relevant_documents": ["ai-report.pdf"],
        "relevant_pages": [15, 22, 30],
        "relevant_keywords": ["K-12", "higher education", "learning management", "longitudinal", "demographic"],
    },
    {
        "question_id": "eval_009",
        "question": "What training data requirements and constraints are discussed for foundational models in educational contexts?",
        "category": "dataset",
        "expected_answer": "Foundational models require massive text and multimodal corpora, but present significant risks regarding training data bias, lack of pedagogical curation, and copyright concerns.",
        "relevant_documents": ["ai-report.pdf"],
        "relevant_pages": [25, 26, 27],
        "relevant_keywords": ["foundational models", "training data", "bias", "corpora", "curation"],
    },

    # --- Category 4: Results & Metrics ---
    {
        "question_id": "eval_010",
        "question": "Which machine learning classifier achieved the highest accuracy for student academic performance prediction?",
        "category": "results",
        "expected_answer": "The highest classification accuracy was achieved by ensemble/tree-based models, outperforming baseline linear classifiers across evaluation folds.",
        "relevant_documents": ["106-ArticleText-199-1-10-20211225.pdf"],
        "relevant_pages": [4, 5],
        "relevant_keywords": ["accuracy", "classification", "performance", "precision", "recall"],
    },
    {
        "question_id": "eval_011",
        "question": "What evaluation metrics were used to measure the performance of the classification models in the academic study?",
        "category": "results",
        "expected_answer": "Evaluation metrics included Accuracy, Precision, Recall, F1-Score, and Confusion Matrix analysis across student grade categories.",
        "relevant_documents": ["106-ArticleText-199-1-10-20211225.pdf"],
        "relevant_pages": [3, 4],
        "relevant_keywords": ["Accuracy", "Precision", "Recall", "F1-Score", "Confusion Matrix"],
    },
    {
        "question_id": "eval_012",
        "question": "What benefits of expert systems and AI in medical diagnosis were highlighted in the IJRTI overview?",
        "category": "results",
        "expected_answer": "Expert systems enhance diagnostic accuracy, assist clinicians in analyzing complex medical imagery, and provide automated rule-based decision support.",
        "relevant_documents": ["IJRTI2304061.pdf"],
        "relevant_pages": [3, 4],
        "relevant_keywords": ["medical diagnosis", "expert systems", "clinical", "decision support", "accuracy"],
    },

    # --- Category 5: Limitations ---
    {
        "question_id": "eval_013",
        "question": "What limitations in sample size and demographic generalizability were noted in the student classification paper?",
        "category": "limitations",
        "expected_answer": "The study was conducted on a localized student sample from a single institution, limiting immediate generalizability across different educational systems.",
        "relevant_documents": ["106-ArticleText-199-1-10-20211225.pdf"],
        "relevant_pages": [5, 6],
        "relevant_keywords": ["sample size", "generalizability", "single institution", "limitation"],
    },
    {
        "question_id": "eval_014",
        "question": "What algorithmic bias and surveillance risks does the Department of Education report associate with automated student monitoring?",
        "category": "limitations",
        "expected_answer": "Automated surveillance technologies can lead to disproportionate disciplinary actions, invasive privacy violations, and unfair algorithmic bias against marginalized student populations.",
        "relevant_documents": ["ai-report.pdf"],
        "relevant_pages": [35, 36, 42],
        "relevant_keywords": ["algorithmic bias", "surveillance", "privacy", "monitoring", "disciplinary"],
    },
    {
        "question_id": "eval_015",
        "question": "What computational and explainability bottlenecks of deep neural networks are cited in the overview paper?",
        "category": "limitations",
        "expected_answer": "Neural networks suffer from high computational resource requirements and the 'black-box' nature of deep learning, hindering interpretability in high-stakes decisions.",
        "relevant_documents": ["IJRTI2304061.pdf"],
        "relevant_pages": [4, 5],
        "relevant_keywords": ["black-box", "computational", "interpretability", "neural networks", "explainability"],
    },

    # --- Category 6: Comparison ---
    {
        "question_id": "eval_016",
        "question": "How do the practical AI application areas described in the IJRTI paper compare to the educational AI tools highlighted in the policy report?",
        "category": "comparison",
        "expected_answer": "While the IJRTI paper covers broad generic domains (robotics, gaming, medicine, expert systems), the policy report specifically targets instructional adaptation, automated assessment, and teacher workflow support.",
        "relevant_documents": ["IJRTI2304061.pdf", "ai-report.pdf"],
        "relevant_pages": [2, 18],
        "relevant_keywords": ["robotics", "instructional", "medicine", "assessment", "domains"],
    },
    {
        "question_id": "eval_017",
        "question": "Compare the data-driven approach of the student performance paper with the systemic governance recommendations in the Department of Education report.",
        "category": "comparison",
        "expected_answer": "The student performance paper focuses on quantitative predictive modeling of grades, whereas the policy report emphasizes ethical oversight, civil rights protections, and educational equity.",
        "relevant_documents": ["106-ArticleText-199-1-10-20211225.pdf", "ai-report.pdf"],
        "relevant_pages": [2, 10],
        "relevant_keywords": ["predictive modeling", "governance", "equity", "oversight", "civil rights"],
    },

    # --- Category 7: Multi-Paper Synthesis ---
    {
        "question_id": "eval_018",
        "question": "Synthesize how machine learning models can both support student outcomes and pose ethical risks across the papers.",
        "category": "multi-paper",
        "expected_answer": "Machine learning offers early identification of struggling students (as demonstrated in the classification paper), but as the policy report cautions, models risk encoding historical biases if deployed without human oversight.",
        "relevant_documents": ["106-ArticleText-199-1-10-20211225.pdf", "ai-report.pdf"],
        "relevant_pages": [4, 35],
        "relevant_keywords": ["student outcomes", "ethical risks", "early identification", "historical bias", "oversight"],
    },
    {
        "question_id": "eval_019",
        "question": "What common themes regarding AI transparency and human judgment emerge across all three papers?",
        "category": "multi-paper",
        "expected_answer": "All three papers recognize that AI systems must maintain explainability and keep humans in control to prevent erroneous automated decisions.",
        "relevant_documents": ["106-ArticleText-199-1-10-20211225.pdf", "ai-report.pdf", "IJRTI2304061.pdf"],
        "relevant_pages": [1, 12, 4],
        "relevant_keywords": ["transparency", "human judgment", "explainability", "control", "decisions"],
    },
    {
        "question_id": "eval_020",
        "question": "Synthesize the evolution of AI from classic expert systems to modern data-driven and foundation models based on the papers.",
        "category": "multi-paper",
        "expected_answer": "AI has evolved from hand-crafted rule-based expert systems and decision trees toward complex deep neural networks and large foundation models requiring massive data and ethical governance.",
        "relevant_documents": ["IJRTI2304061.pdf", "ai-report.pdf", "106-ArticleText-199-1-10-20211225.pdf"],
        "relevant_pages": [2, 25, 3],
        "relevant_keywords": ["expert systems", "foundation models", "neural networks", "evolution", "rule-based"],
    },
]


class EvaluationDataset:
    """Manages evaluation datasets and ground-truth benchmark questions."""

    def __init__(self, dataset_path: Optional[Path] = None) -> None:
        """Initialize the EvaluationDataset."""
        ensure_directories()
        self.dataset_path = dataset_path or (get_evaluation_dir() / "eval_dataset.json")
        self.questions: List[EvaluationQuestion] = []
        self.load_or_initialize()

    def load_or_initialize(self) -> None:
        """Load dataset from disk or initialize with default 20 questions."""
        if self.dataset_path.exists():
            try:
                data = json.loads(self.dataset_path.read_text(encoding="utf-8"))
                self.questions = [EvaluationQuestion.model_validate(q) for q in data]
                return
            except Exception:
                pass

        # Initialize with default 20 questions
        self.questions = [EvaluationQuestion.model_validate(q) for q in DEFAULT_EVALUATION_QUESTIONS]
        self.save()

    def save(self) -> None:
        """Persist evaluation dataset to disk as JSON."""
        data = [q.model_dump() for q in self.questions]
        self.dataset_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def add_question(self, question: EvaluationQuestion) -> None:
        """Add a new evaluation question and persist."""
        self.questions.append(question)
        self.save()

    def get_by_category(self, category: str) -> List[EvaluationQuestion]:
        """Filter questions by category."""
        return [q for q in self.questions if q.category.lower() == category.lower()]
