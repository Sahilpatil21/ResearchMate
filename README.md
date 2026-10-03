# ResearchMate 📚

**"ResearchMate: An AI-Powered Research Paper Analysis and Question Answering System Using Retrieval-Augmented Generation"**

ResearchMate is a modular, production-grade AI research platform designed to assist researchers, students, and academics in ingesting, analyzing, indexing, querying, synthesizing, and evaluating research papers with high precision using Retrieval-Augmented Generation (RAG).

The system combines **Cloudinary PDF Cloud Storage**, **MongoDB Atlas Multi-Tenant User Management & Paper Metadata**, local zero-cost neural embeddings (**ChromaDB**), sparse keyword search (**BM25**), **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder Context Reranking**, and the **Google Gemini API** with strict citation grounding and multi-paper comparative intelligence.

---

## 🏗️ System Architecture

```
                    User
                      │
                      ↓
                Streamlit App
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
     MongoDB       Cloudinary   Auth
      Atlas           │
          │           ↓
          │       Original PDFs
          │           │
          │           ↓
          │      PDF Processing
          │           │
          │           ↓
          │        Chunks
          │           │
          │      ┌────┴────┐
          │      ↓         ↓
          │   ChromaDB    BM25
          │      │         │
          │      └────┬────┘
          │           ↓
          │      Hybrid RRF
          │           ↓
          │    Cross Encoder
          │           ↓
          │       RAG Context
          │           ↓
          └──────→ Gemini
                      ↓
               Grounded Answer
                + Citations
```

### Component Roles & Boundaries:
- **Cloudinary**: Secure, isolated cloud storage for original PDF documents (`researchmate/users/<user_id>/papers/<doc_id>.pdf`).
- **MongoDB Atlas**: User authentication (`users`), paper metadata records (`papers`), document structures (`documents`), chunk records (`chunks`), summaries (`summaries`), comparisons (`comparisons`), and conversation history (`chat_history`).
- **ChromaDB**: 384-dimensional dense semantic embeddings tagged and hard-filtered by `user_id`.
- **BM25**: User-scoped sparse keyword and acronym inverted index.
- **Cross-Encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker for context precision.
- **Google Gemini**: LLM for evidence-grounded answer generation with inline citation attribution.

---

## ☁️ Cloudinary Cloud Storage Configuration

### Setup Instructions:
1. Create a free account at [Cloudinary.com](https://cloudinary.com).
2. From the Cloudinary Dashboard, obtain your **Cloud Name**, **API Key**, and **API Secret**.
3. Add credentials to your `.env` file:
   ```env
   CLOUDINARY_CLOUD_NAME=your_cloud_name
   CLOUDINARY_API_KEY=your_api_key
   CLOUDINARY_API_SECRET=your_api_secret
   ```

> [!IMPORTANT]
> The `CLOUDINARY_API_SECRET` must **never** be committed to Git. `.env` is gitignored by default. Credentials are encrypted and masked in all logs, UI displays, and error tracebacks.

### User-Scoped Public ID Organization:
All uploaded PDFs follow a predictable multi-tenant directory hierarchy:
```
researchmate/users/<user_id>/papers/<document_id>.pdf
```
- **User A:** `researchmate/users/usr_alice_001/papers/doc_101.pdf`
- **User B:** `researchmate/users/usr_bob_002/papers/doc_202.pdf`

---

## 📦 Migration Utility (Local Papers to Cloudinary)

If you have existing papers stored locally on disk, migrate them to Cloudinary seamlessly using the built-in migration script:

```powershell
# Migrate all user papers to Cloudinary (preserves local files safely)
python scripts/migrate_to_cloudinary.py

# Migrate a specific user and delete local copies after verified upload
python scripts/migrate_to_cloudinary.py --user usr_12345 --delete-local
```

---

## 🔒 Multi-Tenant Security & Privacy

1. **User Ownership Verification**: Every operation (viewing, downloading, summarizing, comparing, deleting, and querying) verifies `current_user.user_id == paper.user_id`.
2. **Provenance & Filter Isolation**: Vector and keyword searches are hard-filtered by `user_id`. User A can never retrieve or view User B's documents.
3. **No Local PDF Persistence**: Uploaded PDFs are stored in Cloudinary; local temporary processing files are cleaned up immediately.

---

## 🔑 LLM Provider Configuration (Google Gemini)

ResearchMate uses the official **Google Gemini API** (`gemini-2.5-flash`) as its default provider.

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

---

## 🚀 Running the Application

### 1. Virtual Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch Streamlit Web Dashboard
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing Suite (94 / 94 Tests Passed — 100% Pass Rate)

Run the full automated pytest suite across all test modules:

```powershell
pytest -v
```

### Test Coverage Summary:
- ✅ **Cloudinary Cloud Storage (7 tests)** (`test_cloudinary_storage.py`): Upload, download, deletion, secure URLs, public_id generation, secret redaction, migration manager.
- ✅ **Multi-Tenant Security (7 tests)** (`test_security_multitenant.py`): Cross-user retrieval prevention, cross-user download prevention, cross-user deletion prevention, RAG query isolation, cross-user comparison prevention, document_id tampering prevention, public_id tampering prevention.
- ✅ **Storage & MongoDB Atlas (5 tests)** (`test_storage.py`): Documents, chunks, summaries, comparisons, and chat history persistence.
- ✅ **Authentication & BCrypt (4 tests)** (`test_auth.py`): User serialization, salted bcrypt hashing, registration & login validation, multi-tenant path routing.
- ✅ **PDF Processing (8 tests)** (`test_pdf_processor.py`): Text extraction, blank page skipping, corrupted file handling, fallback metadata.
- ✅ **Section-Aware Chunking (8 tests)** (`test_chunking.py`): Academic header detection, abbreviation protection, overlap, token counting.
- ✅ **Vector Embeddings & ChromaDB (7 tests)** (`test_vectorstore.py`): Local dense vector generation, cosine similarity, document-level filtering, deduplication, persistence.
- ✅ **Advanced Retrieval (7 tests)** (`test_retrieval.py`): Academic tokenization, BM25 exact keyword matching, metadata filtering, RRF fusion math, weighted score fusion.
- ✅ **Cross-Encoder Reranking (5 tests)** (`test_reranking.py`): Lazy loading, candidate pool reranking, provenance preservation.
- ✅ **Citation Attribution (7 tests)** (`test_citations.py`): 1-indexed citation creation, page number grounding, snippet extraction, text integrity, Markdown bibliography formatting.
- ✅ **LLM Provider Abstraction (7 tests)** (`test_llm_provider.py`): Gemini provider configuration, mocked inference, error key redaction, OpenAI, Ollama.
- ✅ **Grounded RAG Pipeline & Validator (12 tests)** (`test_rag.py`): Context builder formatting, token budget truncation, strict prompt generation, citation validation, hallucination detection, insufficient evidence branching.
- ✅ **Research Intelligence (4 tests)** (`test_research_intelligence.py`): 9-field paper summarizer, 10-dimension paper comparator, evidence-grounded research gap analyzer, 10-section literature review generator.
- ✅ **Evaluation Framework (6 tests)** (`test_evaluation.py`): 20-question benchmark dataset management, Precision@K, Recall@K, HitRate@K, MRR evaluation calculations, experiment run persistence.
