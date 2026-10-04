"""
Comprehensive End-to-End Feature Audit Script for ResearchMate.
Tests every single backend endpoint and feature pipeline with real data flow.
"""
import io
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from starlette.testclient import TestClient
from server import app

client = TestClient(app)

def run_audit():
    print("=" * 70)
    print("STARTING RESEARCHMATE FULL FEATURE AUDIT")
    print("=" * 70)

    # 1. Health & Root Check
    print("\n[1/11] Checking Root & System Endpoints...")
    res = client.get("/")
    assert res.status_code == 200, f"Root returned {res.status_code}"
    print("  ✓ Root SPA page loaded successfully")

    res = client.get("/api/diagnostics/system")
    assert res.status_code == 200, f"System diagnostics returned {res.status_code}"
    diag = res.json()
    print(f"  ✓ System Diagnostics: Python {diag.get('python_version')}, Device={diag.get('torch_device')}")

    # 2. Authentication
    print("\n[2/11] Testing User Registration and Authentication...")
    test_user = {
        "email": "audit_researcher@test.edu",
        "password": "SecurePassword123!",
        "full_name": "Dr. Feature Auditor"
    }
    reg_res = client.post("/api/auth/register", json=test_user)
    if reg_res.status_code == 200:
        token = reg_res.json().get("session_token") or reg_res.json().get("token")
        print(f"  ✓ User registered successfully (token: {token[:12]}...)")
    else:
        login_res = client.post("/api/auth/login", json={"email": test_user["email"], "password": test_user["password"]})
        assert login_res.status_code == 200, f"Login failed: {login_res.text}"
        token = login_res.json().get("session_token") or login_res.json().get("token")
        print(f"  ✓ User logged in successfully (token: {token[:12]}...)")

    headers = {"Authorization": f"Bearer {token}"}

    # 3. Dashboard Stats
    print("\n[3/11] Testing Dashboard Analytics & Counts...")
    stats_res = client.get("/api/dashboard/stats", headers=headers)
    assert stats_res.status_code == 200, f"Dashboard failed: {stats_res.text}"
    stats = stats_res.json()
    print(f"  ✓ Dashboard Stats: {stats.get('total_papers', 0)} papers, {stats.get('total_chunks', 0)} chunks, {stats.get('retrieval_mode')} mode")

    # 4. Upload & PDF Ingestion
    print("\n[4/11] Testing PDF Paper Upload & Ingestion...")
    sample_pdf_path = PROJECT_ROOT / "data" / "temp" / "attention_is_all_you_need.pdf"
    if not sample_pdf_path.exists():
        # Create a sample test pdf
        import fitz
        doc = fitz.open()
        page1 = doc.new_page()
        page1.insert_text((50, 50), "Attention Is All You Need\nAshish Vaswani, Noam Shazeer, Niki Parmar\n\nAbstract\nThe dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose the Transformer, a model architecture eschewing recurrence and relying entirely on an attention mechanism.\n\n1 Introduction\nRecurrent neural networks have been firmly established as state of the art approaches in sequence modeling.\n\n2 Background\nThe goal of reducing sequential computation also forms the foundation of the Extended Neural GPU.\n\n3 Model Architecture\nThe Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.\n\n4 Results\nOn the WMT 2014 English-to-German translation task, the big transformer model achieves 28.4 BLEU.")
        page2 = doc.new_page()
        page2.insert_text((50, 50), "5 Conclusion\nIn this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.")
        sample_pdf_bytes = doc.tobytes()
        doc.close()
    else:
        sample_pdf_bytes = sample_pdf_path.read_bytes()

    upload_res = client.post(
        "/api/papers/upload",
        files={"file": ("attention_is_all_you_need.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
        data={"title": "Attention Is All You Need"},
        headers=headers,
    )
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.status_code} {upload_res.text}"
    upload_data = upload_res.json()
    doc_info = upload_data.get("document", {})
    doc_id = doc_info.get("document_id")
    print(f"  ✓ Paper Uploaded & Ingested: ID={doc_id}, Title='{doc_info.get('title')}', Chunks={upload_data.get('chunks_count')}")

    # 5. Paper Library & Details
    print("\n[5/11] Testing Paper Library & Document Details...")
    list_res = client.get("/api/papers", headers=headers)
    assert list_res.status_code == 200
    papers_data = list_res.json()
    papers_list = papers_data.get("documents", papers_data) if isinstance(papers_data, dict) else papers_data
    assert any(p.get("document_id") == doc_id for p in papers_list), "Uploaded doc not found in library list"
    print(f"  ✓ Library List: {len(papers_list)} papers retrieved")

    detail_res = client.get(f"/api/papers/{doc_id}", headers=headers)
    assert detail_res.status_code == 200
    doc_detail = detail_res.json()
    print(f"  ✓ Paper Detail: {doc_detail.get('document', {}).get('title')} ({doc_detail.get('document', {}).get('page_count')} pages)")

    # 6. Grounded RAG Chat
    print("\n[6/11] Testing Grounded RAG Chat & Citation Extraction...")
    chat_res = client.post(
        "/api/chat",
        json={
            "query": "What is the key architecture proposed in Attention Is All You Need?",
            "document_id": doc_id,
            "retrieval_mode": "hybrid",
            "top_k": 5,
            "use_reranker": False,
        },
        headers=headers,
    )
    assert chat_res.status_code == 200, f"Chat failed: {chat_res.text}"
    chat_data = chat_res.json()
    print(f"  ✓ Chat Response: {chat_data.get('answer')[:120]}...")
    print(f"  ✓ Grounded Citations: {len(chat_data.get('citations', []))} citations found (Valid: {chat_data.get('validation', {}).get('is_valid')})")

    # 7. 9-Field Structured Summarizer
    print("\n[7/11] Testing 9-Field Academic Paper Summarizer...")
    sum_res = client.post(
        "/api/intelligence/summarize",
        json={"document_id": doc_id},
        headers=headers,
    )
    assert sum_res.status_code == 200, f"Summarize failed: {sum_res.text}"
    sum_data = sum_res.json()
    summary = sum_data.get("summary", {})
    print(f"  ✓ 9-Field Summary generated:")
    print(f"    - Research Problem: {summary.get('research_problem', '')[:80]}...")
    print(f"    - Methodology: {summary.get('methodology', '')[:80]}...")
    print(f"    - Main Results: {summary.get('main_results', '')[:80]}...")
    print(f"    - Limitations: {summary.get('limitations', '')[:80]}...")

    # Upload a 2nd paper for comparative testing
    import fitz
    doc2 = fitz.open()
    p1 = doc2.new_page()
    p1.insert_text((50, 50), "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding\nJacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova\n\nAbstract\nWe introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text.\n\n1 Introduction\nLanguage model pre-training has been shown to be effective for improving many natural language processing tasks.\n\n2 BERT Framework\nThere are two steps in our framework: pre-training and fine-tuning. BERT's model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al. (2017).\n\n3 Experiments\nBERT achieves state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5%.")
    doc2_bytes = doc2.tobytes()
    doc2.close()

    upload_res2 = client.post(
        "/api/papers/upload",
        files={"file": ("bert_paper.pdf", io.BytesIO(doc2_bytes), "application/pdf")},
        data={"title": "BERT: Pre-training of Deep Bidirectional Transformers"},
        headers=headers,
    )
    assert upload_res2.status_code == 200
    doc2_id = upload_res2.json()["document"]["document_id"]
    print(f"  ✓ Second Paper Uploaded: ID={doc2_id}, Title='BERT'")

    # 8. Multi-Paper Comparative Analysis
    print("\n[8/11] Testing Multi-Paper Comparison Matrix...")
    comp_res = client.post(
        "/api/intelligence/compare",
        json={"document_ids": [doc_id, doc2_id], "topic": "Transformer Architectures: Self-Attention vs Bidirectional Pre-training"},
        headers=headers,
    )
    assert comp_res.status_code == 200, f"Compare failed: {comp_res.text}"
    comp_data = comp_res.json()
    report = comp_data.get("report", {})
    matrix = report.get("comparison_matrix", report.get("comparison_table", []))
    print(f"  ✓ Comparison Matrix: {len(matrix)} dimension rows evaluated across 2 papers")

    # 9. Research Gap Analyzer & Literature Review
    print("\n[9/11] Testing Research Gap Discovery & 10-Section Literature Review...")
    gaps_res = client.post(
        "/api/intelligence/gaps",
        json={"document_ids": [doc_id], "topic": "Transformer scaling & limitations"},
        headers=headers,
    )
    assert gaps_res.status_code == 200, f"Gaps failed: {gaps_res.text}"
    gaps_data = gaps_res.json()
    print(f"  ✓ Research Gaps Analyzed: {len(gaps_data.get('gaps', []))} gaps found")

    lit_res = client.post(
        "/api/intelligence/lit-review",
        json={"document_ids": [doc_id], "topic": "Evolution of Attention Mechanisms"},
        headers=headers,
    )
    assert lit_res.status_code == 200, f"Lit Review failed: {lit_res.text}"
    lit_data = lit_res.json()
    lit_report = lit_data.get("review", {})
    print(f"  ✓ Literature Review Generated: Title='{lit_report.get('title')}', Sections={len(lit_report.get('sections', []))}")

    # 10. Evaluation Studio Benchmarks
    print("\n[10/11] Testing Evaluation Studio (Hit@k, MRR, Precision@k)...")
    eval_res = client.post(
        "/api/evaluation/run-retrieval",
        json={"k": 5, "rerank": False},
        headers=headers,
    )
    assert eval_res.status_code == 200, f"Evaluation failed: {eval_res.text}"
    eval_data = eval_res.json()
    metrics = eval_data.get("metrics", {})
    print(f"  ✓ Retrieval Evaluation Benchmark: {metrics.get('sample_count', 0)} queries evaluated")
    for method, smry in metrics.get("method_summaries", {}).items():
        print(f"    - {method}: Hit@5={smry.get('hit_at_k', 0):.2f}, MRR={smry.get('mrr', 0):.2f}, Precision@5={smry.get('precision_at_k', 0):.2f}")

    # 11. System Diagnostics & Cleanup
    print("\n[11/11] Testing System Diagnostics & Cleanup...")
    diag_res = client.get("/api/diagnostics/system", headers=headers)
    assert diag_res.status_code == 200
    diag = diag_res.json()
    print(f"  ✓ System Diagnostics: Python {diag.get('python_version')}, Device={diag.get('torch_device')}, Services={list(diag.get('services', {}).keys())}")

    print("\n" + "=" * 70)
    print("🎉 ALL 11 FEATURES PASSED END-TO-END VERIFICATION!")
    print("=" * 70)

if __name__ == "__main__":
    run_audit()
