"""Tests for FastAPI backend server (server.py) and REST API endpoints."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from server import app
from src.auth.user_model import User
from src.models.chunk import Chunk, ChunkMetadata
from src.models.citation import Citation
from src.models.document import Document, DocumentMetadata, Page
from src.rag.answer_validator import ValidationResult
from src.rag.rag_pipeline import RAGResponse
from src.intelligence.summarizer import PaperSummary
from src.intelligence.comparator import PaperComparisonReport
from src.intelligence.gap_analyzer import ResearchGapReport, ResearchGapItem
from src.intelligence.lit_reviewer import LiteratureReviewReport


@pytest.fixture
def client():
    """Create a FastAPI TestClient instance."""
    return TestClient(app)


@pytest.fixture
def mock_user():
    """Mock user object for authentication bypass / verification."""
    return User(
        user_id="test_user_server_123",
        email="researcher_server@test.edu",
        full_name="Dr. Test Researcher",
        password_hash="$2b$12$eMockedPasswordHashForTestingOnly1234567890",
        created_at="2026-10-04T00:00:00Z",
    )


@pytest.fixture
def sample_document():
    """Sample Document model for testing API endpoints."""
    meta = DocumentMetadata(
        document_id="doc_server_test_001",
        filename="test_paper.pdf",
        original_filename="test_paper.pdf",
        title="Attention Is All You Need",
        authors=["Vaswani et al."],
        page_count=1,
        total_pages_in_pdf=1,
        user_id="test_user_server_123",
    )
    pages = [Page(page_number=1, text="The dominant sequence transduction models are based on neural networks.")]
    return Document(metadata=meta, pages=pages)


def test_spa_root(client):
    """Test root endpoint serves HTML frontend."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_auth_register_and_login_flow(client):
    """Test user registration, login, profile me, and logout endpoints."""
    with patch("server.ServiceRegistry.get_auth") as mock_auth_getter:
        mock_auth = MagicMock()
        mock_user_inst = User(
            user_id="user_reg_999",
            email="unique_user@test.org",
            full_name="Registration Tester",
            password_hash="$2b$12$eMockedPasswordHashForTestingOnly1234567890",
        )
        mock_auth.register_user.return_value = (True, "Registration successful!", mock_user_inst)
        mock_auth.authenticate_user.return_value = (True, "Login successful!", mock_user_inst)
        mock_auth.create_session.return_value = "test_session_token_xyz"
        mock_auth.get_user_by_session_token.return_value = mock_user_inst
        mock_auth_getter.return_value = mock_auth

        # 1. Register
        reg_resp = client.post(
            "/api/auth/register",
            json={
                "email": "unique_user@test.org",
                "password": "SecurePassword123!",
                "full_name": "Registration Tester",
            },
        )
        assert reg_resp.status_code == 200
        reg_json = reg_resp.json()
        assert reg_json["success"] is True
        assert reg_json["session_token"] == "test_session_token_xyz"
        assert reg_json["user"]["email"] == "unique_user@test.org"

        # 2. Login
        login_resp = client.post(
            "/api/auth/login",
            json={
                "email": "unique_user@test.org",
                "password": "SecurePassword123!",
            },
        )
        assert login_resp.status_code == 200
        login_json = login_resp.json()
        assert login_json["success"] is True
        assert login_json["session_token"] == "test_session_token_xyz"

        # 3. Auth Me
        me_resp = client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer test_session_token_xyz"},
        )
        assert me_resp.status_code == 200
        me_json = me_resp.json()
        assert me_json["user"]["user_id"] == "user_reg_999"

        # 4. Logout
        logout_resp = client.post(
            "/api/auth/logout",
            headers={"Authorization": "Bearer test_session_token_xyz"},
        )
        assert logout_resp.status_code == 200
        assert logout_resp.json()["success"] is True


def test_dashboard_stats(client, mock_user, sample_document):
    """Test dashboard stats calculation endpoint."""
    with patch("server.get_current_user", return_value=mock_user), \
         patch("server.ServiceRegistry.get_storage") as mock_storage_getter, \
         patch("server.ServiceRegistry.get_retriever") as mock_retriever_getter, \
         patch("server.ServiceRegistry.get_cloudinary") as mock_c_getter:

        mock_storage = MagicMock()
        mock_storage.load_user_documents.return_value = [sample_document]
        mock_storage.is_connected = True
        mock_storage_getter.return_value = mock_storage

        mock_retriever = MagicMock()
        mock_retriever.get_stats.return_value = {
            "total_indexed_chunks": 42,
            "dense_count": 42,
            "sparse_count": 42,
        }
        mock_retriever_getter.return_value = mock_retriever

        mock_c = MagicMock()
        mock_c.is_configured = True
        mock_c_getter.return_value = mock_c

        resp = client.get("/api/dashboard/stats", headers={"Authorization": "Bearer fake_token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_papers"] == 1
        assert data["total_chunks"] == 42
        assert data["total_pages"] == 1
        assert data["mongo_connected"] is True
        assert data["cloudinary_active"] is True


def test_papers_list_and_details(client, mock_user, sample_document):
    """Test listing papers and fetching paper details with chunk breakdown."""
    with patch("server.get_current_user", return_value=mock_user), \
         patch("server.ServiceRegistry.get_storage") as mock_storage_getter:

        mock_storage = MagicMock()
        mock_storage.load_user_documents.return_value = [sample_document]
        mock_storage.load_document.return_value = sample_document
        sample_chunk = Chunk(
            text="Attention mechanism sample text",
            metadata=ChunkMetadata(
                chunk_id="chk_001",
                document_id=sample_document.document_id,
                filename="test_paper.pdf",
                page_number=1,
                page_numbers=[1],
                section="Architecture",
                chunk_index=0,
                token_count=10,
            ),
        )
        mock_storage.load_document_chunks.return_value = [sample_chunk]
        mock_storage_getter.return_value = mock_storage

        # List papers
        list_resp = client.get("/api/papers", headers={"Authorization": "Bearer fake_token"})
        assert list_resp.status_code == 200
        list_data = list_resp.json()
        assert list_data["count"] == 1
        assert list_data["documents"][0]["metadata"]["document_id"] == sample_document.document_id

        # Paper details
        details_resp = client.get(f"/api/papers/{sample_document.document_id}", headers={"Authorization": "Bearer fake_token"})
        assert details_resp.status_code == 200
        details_data = details_resp.json()
        assert details_data["document"]["metadata"]["title"] == "Attention Is All You Need"
        assert details_data["chunks_count"] == 1


def test_paper_upload_and_delete(client, mock_user, sample_document):
    """Test PDF paper upload, indexing, and subsequent deletion."""
    with patch("server.get_current_user", return_value=mock_user), \
         patch("server.save_uploaded_pdf", return_value="data/temp/test.pdf"), \
         patch("server.ServiceRegistry.get_pdf_processor") as mock_pdf_getter, \
         patch("server.ServiceRegistry.get_storage") as mock_storage_getter, \
         patch("server.ServiceRegistry.get_cloudinary") as mock_c_getter, \
         patch("server.ServiceRegistry.get_retriever") as mock_retriever_getter, \
         patch("server.save_processed_document"), \
         patch("server.save_document_chunks"), \
         patch("server.delete_local_document"):

        mock_pdf = MagicMock()
        mock_pdf.process_pdf.return_value = sample_document
        mock_pdf_getter.return_value = mock_pdf

        mock_storage = MagicMock()
        mock_storage_getter.return_value = mock_storage

        mock_c = MagicMock()
        mock_c.is_configured = True
        mock_c.upload_pdf.return_value = {"secure_url": "https://res.cloudinary.com/test/paper.pdf"}
        mock_c_getter.return_value = mock_c

        mock_retriever = MagicMock()
        mock_retriever_getter.return_value = mock_retriever

        # Upload
        fake_pdf_file = io.BytesIO(b"%PDF-1.4 mock pdf content bytes")
        upload_resp = client.post(
            "/api/papers/upload",
            files={"file": ("test_paper.pdf", fake_pdf_file, "application/pdf")},
            data={"title": "Custom Test Title"},
            headers={"Authorization": "Bearer fake_token"},
        )
        assert upload_resp.status_code == 200
        upload_data = upload_resp.json()
        assert upload_data["success"] is True
        assert upload_data["document"]["metadata"]["document_id"] == sample_document.document_id

        # Delete
        del_resp = client.delete(f"/api/papers/{sample_document.document_id}", headers={"Authorization": "Bearer fake_token"})
        assert del_resp.status_code == 200
        assert del_resp.json()["success"] is True


def test_multi_paper_upload(client, mock_user, sample_document):
    """Test uploading multiple PDF papers in a single batch request."""
    with patch("server.get_current_user", return_value=mock_user), \
         patch("server.save_uploaded_pdf", return_value="data/temp/test.pdf"), \
         patch("server.ServiceRegistry.get_pdf_processor") as mock_pdf_getter, \
         patch("server.ServiceRegistry.get_storage") as mock_storage_getter, \
         patch("server.ServiceRegistry.get_cloudinary") as mock_c_getter, \
         patch("server.ServiceRegistry.get_retriever") as mock_retriever_getter, \
         patch("server.save_processed_document"), \
         patch("server.save_document_chunks"):

        mock_pdf = MagicMock()
        mock_pdf.process_pdf.return_value = sample_document
        mock_pdf_getter.return_value = mock_pdf

        mock_storage = MagicMock()
        mock_storage_getter.return_value = mock_storage

        mock_c = MagicMock()
        mock_c.is_configured = False
        mock_c_getter.return_value = mock_c

        mock_retriever = MagicMock()
        mock_retriever_getter.return_value = mock_retriever

        # Upload multiple files
        f1 = io.BytesIO(b"%PDF-1.4 paper1 content")
        f2 = io.BytesIO(b"%PDF-1.4 paper2 content")
        upload_resp = client.post(
            "/api/papers/upload",
            files=[
                ("files", ("paper1.pdf", f1, "application/pdf")),
                ("files", ("paper2.pdf", f2, "application/pdf")),
            ],
            headers={"Authorization": "Bearer fake_token"},
        )
        assert upload_resp.status_code == 200
        upload_data = upload_resp.json()
        assert upload_data["success"] is True
        assert len(upload_data["documents"]) == 2
        assert "2 papers" in upload_data["message"]


def test_chat_rag_endpoint(client, mock_user):
    """Test Grounded RAG Chat API with citation validation response."""
    with patch("server.get_current_user", return_value=mock_user), \
         patch("server.ServiceRegistry.get_rag") as mock_rag_getter, \
         patch("server.ServiceRegistry.get_storage") as mock_storage_getter:

        mock_storage = MagicMock()
        mock_storage.is_connected = True
        mock_storage_getter.return_value = mock_storage

        citation_sample = Citation(
            citation_id="[1]",
            citation_index=1,
            document_id="doc_1",
            filename="paper.pdf",
            paper_title="Attention Is All You Need",
            section="Architecture",
            page_number=4,
            chunk_id="chk_001",
            chunk_index=0,
            source_text="Multi-head attention allows the model to attend to information.",
            source_snippet="Multi-head attention allows the model to attend to information.",
            groundedness_score=0.95,
        )

        mock_rag = MagicMock()
        mock_rag.answer_question.return_value = RAGResponse(
            query="How does multi-head attention work?",
            answer="Multi-head attention projects queries, keys, and values into multiple subspaces [1].",
            citations=[citation_sample],
            validation=ValidationResult(
                is_valid=True,
                has_citations=True,
                cited_indices=[1],
                valid_citations=[1],
                invalid_citations=[],
                grounding_score=1.0,
            ),
            provider="gemini",
            model="gemini-2.5-flash",
            latency_seconds=0.45,
            is_insufficient_evidence=False,
        )
        mock_rag_getter.return_value = mock_rag

        chat_resp = client.post(
            "/api/chat",
            json={"query": "How does multi-head attention work?", "top_k": 5, "use_reranker": True},
            headers={"Authorization": "Bearer fake_token"},
        )
        assert chat_resp.status_code == 200
        chat_data = chat_resp.json()
        assert "Multi-head attention" in chat_data["answer"]
        assert len(chat_data["citations"]) == 1
        assert chat_data["citations"][0]["page_number"] == 4
        assert chat_data["validation"]["is_valid"] is True


def test_intelligence_summarize(client, mock_user):
    """Test structured 9-field academic paper summarization endpoint."""
    with patch("server.get_current_user", return_value=mock_user), \
         patch("server.ServiceRegistry.get_summarizer") as mock_sum_getter, \
         patch("server.ServiceRegistry.get_storage") as mock_storage_getter:

        mock_storage = MagicMock()
        mock_storage.is_connected = True
        mock_storage.load_summary.return_value = None
        mock_storage_getter.return_value = mock_storage

        mock_sum = MagicMock()
        mock_sum.summarize_paper.return_value = PaperSummary(
            document_id="doc_sum_001",
            paper_title="Deep Residual Learning for Image Recognition",
            filename="resnet.pdf",
            research_problem="Can deeper networks be trained effectively?",
            objective="Overcome the degradation problem in deep neural networks.",
            methodology="Residual connections with identity shortcut mapping.",
            dataset="ImageNet 2012, CIFAR-10",
            model_architecture="ResNet-50, ResNet-101, ResNet-152",
            experimental_setup="SGD with batch size 256, weight decay 0.0001",
            main_results="Residual networks achieve state of the art with 3.57% top-5 error.",
            limitations="Increased memory footprint during backward pass.",
            conclusion="Deep residual networks are easy to optimize and gain accuracy from increased depth.",
        )
        mock_sum_getter.return_value = mock_sum

        resp = client.post(
            "/api/intelligence/summarize",
            json={"document_id": "doc_sum_001"},
            headers={"Authorization": "Bearer fake_token"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["summary"]["paper_title"] == "Deep Residual Learning for Image Recognition"


def test_intelligence_compare(client, mock_user):
    """Test multi-paper comparative intelligence endpoint."""
    with patch("server.get_current_user", return_value=mock_user), \
         patch("server.ServiceRegistry.get_comparator") as mock_comp_getter:

        mock_comp = MagicMock()
        mock_comp.compare_papers.return_value = PaperComparisonReport(
            document_ids=["doc_1", "doc_2"],
            paper_titles=["Paper A", "Paper B"],
            comparison_table={
                "Architecture": {
                    "Paper A": "Self-Attention mechanism",
                    "Paper B": "Convolutional filters",
                }
            },
            comparative_synthesis="Both paradigms offer complementary strengths in spatial and sequential reasoning.",
        )
        mock_comp_getter.return_value = mock_comp

        resp = client.post(
            "/api/intelligence/compare",
            json={"document_ids": ["doc_1", "doc_2"], "topic": "Attention vs CNNs"},
            headers={"Authorization": "Bearer fake_token"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "Architecture" in data["report"]["comparison_table"]


def test_intelligence_gaps_and_lit_review(client, mock_user):
    """Test research gap analysis and literature review generation endpoints."""
    with patch("server.get_current_user", return_value=mock_user), \
         patch("server.ServiceRegistry.get_gap_analyzer") as mock_gap_getter, \
         patch("server.ServiceRegistry.get_lit_reviewer") as mock_lit_getter:

        # Gaps
        mock_gap = MagicMock()
        mock_gap.analyze_gaps.return_value = ResearchGapReport(
            document_ids=["doc_1"],
            executive_summary="Exploration of computational bottlenecks in transformer models.",
            gaps=[
                ResearchGapItem(
                    gap_statement="Quadratic Complexity in Long Sequences",
                    category="Methodology Limitation",
                    supporting_evidence="Self-attention layers scale O(N^2) with sequence length.",
                    source_paper="Attention Is All You Need",
                    page_number=4,
                    section="Complexity",
                    citation_id="[1]",
                    future_direction_suggestion="Investigate linear attention approximations.",
                )
            ],
        )
        mock_gap_getter.return_value = mock_gap

        gap_resp = client.post(
            "/api/intelligence/gaps",
            json={"document_ids": ["doc_1"], "topic": "Scaling Laws"},
            headers={"Authorization": "Bearer fake_token"},
        )
        assert gap_resp.status_code == 200
        gap_data = gap_resp.json()
        assert gap_data["success"] is True
        assert len(gap_data["report"]["gaps"]) == 1

        # Lit Review
        mock_lit = MagicMock()
        mock_lit.generate_review.return_value = LiteratureReviewReport(
            topic="Transformer Architectures",
            document_ids=["doc_1"],
            introduction="Transformers have revolutionized natural language processing.",
            research_area_overview="Overview of sequence modeling architectures.",
            existing_approaches="RNNs and LSTMs previously dominated.",
            methodology_comparison="Self-attention replaces recurrence entirely.",
            dataset_trends="WMT 2014 English-to-German.",
            key_findings="Superior BLEU score with lower training time.",
            limitations="Memory requirement for long context.",
            potential_research_gaps="Efficient cross-document attention.",
            future_research_directions="Sparse attention patterns.",
            references_markdown="[1] Vaswani et al., 2017.",
        )
        mock_lit_getter.return_value = mock_lit

        lit_resp = client.post(
            "/api/intelligence/lit-review",
            json={"document_ids": ["doc_1"], "topic": "Transformer Architectures"},
            headers={"Authorization": "Bearer fake_token"},
        )
        assert lit_resp.status_code == 200
        lit_data = lit_resp.json()
        assert lit_data["success"] is True
        assert lit_data["report"]["topic"] == "Transformer Architectures"


def test_evaluation_endpoints(client, mock_user):
    """Test evaluation studio benchmarking and human rating endpoints."""
    with patch("server.get_current_user", return_value=mock_user), \
         patch("server.ServiceRegistry.get_eval_suite") as mock_eval_getter:

        mock_dataset = MagicMock()
        mock_evaluator = MagicMock()
        mock_logger = MagicMock()

        mock_report = MagicMock()
        mock_report.model_dump.return_value = {
            "evaluation_id": "eval_test_123",
            "k": 5,
            "dense_results": {"precision_at_k": 0.85, "recall_at_k": 0.90, "mrr": 0.88, "hit_rate": 1.0},
            "sparse_results": {"precision_at_k": 0.75, "recall_at_k": 0.80, "mrr": 0.78, "hit_rate": 0.95},
            "hybrid_results": {"precision_at_k": 0.92, "recall_at_k": 0.95, "mrr": 0.94, "hit_rate": 1.0},
            "reranked_results": {"precision_at_k": 0.96, "recall_at_k": 0.98, "mrr": 0.97, "hit_rate": 1.0},
        }
        mock_evaluator.evaluate_all_methods.return_value = mock_report
        mock_logger.load_recent_experiments.return_value = []
        mock_logger.load_human_ratings.return_value = []

        mock_eval_getter.return_value = (mock_dataset, mock_evaluator, mock_logger)

        # Run retrieval eval
        eval_resp = client.post("/api/evaluation/run-retrieval?top_k=5", headers={"Authorization": "Bearer fake_token"})
        assert eval_resp.status_code == 200
        eval_data = eval_resp.json()
        assert eval_data["success"] is True
        assert eval_data["report"]["reranked_results"]["precision_at_k"] == 0.96

        # Human rating
        rate_resp = client.post(
            "/api/evaluation/rate",
            json={
                "evaluator_id": "Researcher A",
                "query": "What is self-attention?",
                "correctness": 5.0,
                "relevance": 5.0,
                "completeness": 4.5,
                "citation_quality": 5.0,
                "groundedness": 5.0,
                "feedback_notes": "Flawless grounded response.",
            },
            headers={"Authorization": "Bearer fake_token"},
        )
        assert rate_resp.status_code == 200
        assert rate_resp.json()["success"] is True

        # Eval history
        hist_resp = client.get("/api/evaluation/history", headers={"Authorization": "Bearer fake_token"})
        assert hist_resp.status_code == 200
        assert "experiments" in hist_resp.json()


def test_diagnostics_endpoint(client, mock_user):
    """Test diagnostics and infrastructure status API."""
    with patch("server.get_current_user", return_value=mock_user), \
         patch("server.ServiceRegistry.get_storage") as mock_storage_getter, \
         patch("server.ServiceRegistry.get_cloudinary") as mock_c_getter, \
         patch("server.ServiceRegistry.get_retriever") as mock_retriever_getter:

        mock_storage = MagicMock()
        mock_storage.is_connected = True
        mock_storage.db_name = "ResearchMate_Test"
        mock_storage_getter.return_value = mock_storage

        mock_c = MagicMock()
        mock_c.is_configured = True
        mock_c.cloud_name = "demo-cloud"
        mock_c_getter.return_value = mock_c

        mock_retriever = MagicMock()
        mock_retriever.reranker.get_stats.return_value = {
            "model_name": "ms-marco-MiniLM-L-6-v2",
            "is_loaded": True,
        }
        mock_retriever_getter.return_value = mock_retriever

        resp = client.get("/api/settings/diagnostics", headers={"Authorization": "Bearer fake_token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["cloud_storage"]["mongodb_connected"] is True
        assert data["cloud_storage"]["cloudinary_active"] is True
        assert data["dense_embeddings"]["dimensions"] == 384
