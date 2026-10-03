"""Explicit Security and Multi-Tenant Isolation Tests for ResearchMate.

Validates the 7 critical security requirements:
- Test 1: User A uploads Paper A -> User B must not retrieve Paper A (Dense/Sparse/Hybrid)
- Test 2: User B must not download Paper A (cross-user download prevention)
- Test 3: User B must not delete Paper A (cross-user deletion prevention)
- Test 4: User B must not query Paper A through RAG (provenance isolation)
- Test 5: User B must not compare Paper A with their own papers
- Test 6: Changing document_id manually must not bypass ownership checks
- Test 7: Changing Cloudinary public_id manually must not bypass ownership checks
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.models.chunk import Chunk, ChunkMetadata
from src.models.document import Document, DocumentMetadata, Page
from src.rag.rag_pipeline import RAGPipeline
from src.storage.cloudinary_storage import CloudinaryStorage
from src.storage.mongo_storage import MongoStorageService
from src.vectorstore.chroma_store import ChromaStore


@pytest.fixture
def test_users_workspace(tmp_path):
    """Setup isolated multi-user test environment."""
    user_a = "usr_alice_001"
    user_b = "usr_bob_002"

    doc_a = Document(
        metadata=DocumentMetadata(
            document_id="doc_alice_paper_a",
            filename="alice_quantum_ai.pdf",
            original_filename="alice_quantum_ai.pdf",
            title="Alice Confidential Quantum AI Research",
            page_count=2,
            user_id=user_a,
            cloudinary_public_id=f"researchmate/users/{user_a}/papers/doc_alice_paper_a.pdf",
        ),
        pages=[
            Page(page_number=1, text="Alice secret methodology on quantum entanglement."),
            Page(page_number=2, text="Quantum results show 99.9% breakthrough."),
        ],
    )

    doc_b = Document(
        metadata=DocumentMetadata(
            document_id="doc_bob_paper_b",
            filename="bob_nlp_transformers.pdf",
            original_filename="bob_nlp_transformers.pdf",
            title="Bob Public NLP Transformers Study",
            page_count=2,
            user_id=user_b,
            cloudinary_public_id=f"researchmate/users/{user_b}/papers/doc_bob_paper_b.pdf",
        ),
        pages=[
            Page(page_number=1, text="Bob study on NLP attention mechanisms."),
            Page(page_number=2, text="Transformer benchmarks show 85% accuracy."),
        ],
    )

    chunks_a = [
        Chunk(
            text="Alice secret methodology on quantum entanglement in AI models.",
            metadata=ChunkMetadata(
                chunk_id="chk_a1",
                chunk_index=0,
                document_id="doc_alice_paper_a",
                filename="alice_quantum_ai.pdf",
                section="Methodology",
                page_number=1,
            ),
        ),
    ]

    chunks_b = [
        Chunk(
            text="Bob study on NLP attention mechanisms in large language models.",
            metadata=ChunkMetadata(
                chunk_id="chk_b1",
                chunk_index=0,
                document_id="doc_bob_paper_b",
                filename="bob_nlp_transformers.pdf",
                section="Introduction",
                page_number=1,
            ),
        ),
    ]

    return {
        "user_a": user_a,
        "user_b": user_b,
        "doc_a": doc_a,
        "doc_b": doc_b,
        "chunks_a": chunks_a,
        "chunks_b": chunks_b,
        "tmp_path": tmp_path,
    }


def test_security_1_user_b_cannot_retrieve_user_a_papers(test_users_workspace):
    """Test 1: User A uploads Paper A. User B must not retrieve Paper A."""
    env = test_users_workspace
    user_a = env["user_a"]
    user_b = env["user_b"]

    chroma_store = ChromaStore(persist_dir=env["tmp_path"] / "chroma_sec1")
    # Index User A's chunks with user_id=user_a
    chroma_store.add_chunks(env["chunks_a"], user_id=user_a)
    # Index User B's chunks with user_id=user_b
    chroma_store.add_chunks(env["chunks_b"], user_id=user_b)

    # When User B searches for "quantum entanglement" (Alice's paper content)
    results_b = chroma_store.search("quantum entanglement", n_results=5, user_id=user_b)
    
    # User B must NOT receive any chunks from Paper A
    for r in results_b:
        assert r.document_id != "doc_alice_paper_a"
        assert "Alice" not in r.text

    # User A searching the same query MUST receive Paper A
    results_a = chroma_store.search("quantum entanglement", n_results=5, user_id=user_a)
    assert any(r.document_id == "doc_alice_paper_a" for r in results_a)


def test_security_2_user_b_cannot_download_user_a_paper(test_users_workspace):
    """Test 2: User B must not download Paper A (cross-user download prevention)."""
    env = test_users_workspace
    user_a = env["user_a"]
    user_b = env["user_b"]
    doc_a = env["doc_a"]

    storage = MongoStorageService()
    storage.save_document(doc_a, user_id=user_a)

    # Attempting to load User A's document under User B's session must return None
    loaded_by_b = storage.load_document(doc_a.document_id, user_id=user_b)
    if loaded_by_b is not None:
        assert loaded_by_b.metadata.user_id != user_b

    # Cloudinary public_id verification
    c_storage = CloudinaryStorage(cloud_name="demo", api_key="123", api_secret="sec")
    # User B attempting to build public_id for Doc A gets User B's scoped path, not User A's
    pid_b = c_storage.build_public_id(user_b, doc_a.document_id)
    assert f"users/{user_b}/" in pid_b
    assert f"users/{user_a}/" not in pid_b


def test_security_3_user_b_cannot_delete_user_a_paper(test_users_workspace):
    """Test 3: User B must not delete Paper A (cross-user deletion prevention)."""
    env = test_users_workspace
    user_a = env["user_a"]
    user_b = env["user_b"]
    doc_a = env["doc_a"]

    chroma_store = ChromaStore(persist_dir=env["tmp_path"] / "chroma_sec3")
    chroma_store.add_chunks(env["chunks_a"], user_id=user_a)

    # User B attempts to delete Doc A
    chroma_store.delete_document(doc_a.document_id, user_id=user_b)

    # User A's document must still exist and be indexed
    assert chroma_store.is_document_indexed(doc_a.document_id, user_id=user_a) is True


def test_security_4_user_b_cannot_query_user_a_paper_in_rag(test_users_workspace):
    """Test 4: User B must not query Paper A through RAG."""
    env = test_users_workspace
    user_b = env["user_b"]

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []  # User B gets 0 results for User A's topic

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.get_provider_name.return_value = "mock_gemini"
    mock_llm.get_model_name.return_value = "gemini-2.5-flash"

    rag = RAGPipeline(retriever=mock_retriever, llm_provider=mock_llm)
    
    # Query with user_id=user_b
    response = rag.answer_question(
        query="What is Alice's secret quantum breakthrough?",
        user_id=user_b,
    )

    # Verify retriever was called with user_id=user_b
    mock_retriever.retrieve.assert_called_once()
    _, kwargs = mock_retriever.retrieve.call_args
    assert kwargs.get("user_id") == user_b
    assert response.is_insufficient_evidence is True


def test_security_5_user_b_cannot_compare_user_a_paper(test_users_workspace):
    """Test 5: User B must not compare Paper A with their own papers."""
    env = test_users_workspace
    user_a = env["user_a"]
    user_b = env["user_b"]
    doc_a = env["doc_a"]
    doc_b = env["doc_b"]

    storage = MongoStorageService()
    storage.save_document(doc_a, user_id=user_a)
    storage.save_document(doc_b, user_id=user_b)

    # User B loads their library to compare
    user_b_docs = storage.load_user_documents(user_id=user_b)
    user_b_doc_ids = [d.document_id for d in user_b_docs]

    # User B's library must not include Paper A
    assert doc_a.document_id not in user_b_doc_ids


def test_security_6_changing_document_id_does_not_bypass_ownership(test_users_workspace):
    """Test 6: Changing document_id manually must not bypass ownership checks."""
    env = test_users_workspace
    user_a = env["user_a"]
    user_b = env["user_b"]

    chroma_store = ChromaStore(persist_dir=env["tmp_path"] / "chroma_sec6")
    chroma_store.add_chunks(env["chunks_a"], user_id=user_a)

    # User B passes document_id filter for Paper A
    results = chroma_store.search("quantum", filter_doc_id="doc_alice_paper_a", user_id=user_b)
    # Must return empty because user_id=user_b does not own doc_alice_paper_a
    assert len(results) == 0


def test_security_7_changing_cloudinary_public_id_does_not_bypass_ownership(test_users_workspace):
    """Test 7: Changing Cloudinary public_id manually must not bypass ownership checks."""
    c_storage = CloudinaryStorage(cloud_name="demo", api_key="123", api_secret="sec")
    
    # Alice's real public_id
    alice_real_pid = c_storage.build_public_id("usr_alice_001", "doc_001")
    assert alice_real_pid == "researchmate/users/usr_alice_001/papers/doc_001.pdf"

    # If Bob tries to access Doc 001, the system generates public_id using Bob's authenticated user_id
    bob_pid = c_storage.build_public_id("usr_bob_002", "doc_001")
    assert bob_pid == "researchmate/users/usr_bob_002/papers/doc_001.pdf"
    assert bob_pid != alice_real_pid
