"""ResearchMate - AI-Powered Research Paper Analysis & Multi-User Intelligence System.

Features:
- 🔐 Secure MongoDB & Local Authentication with bcrypt
- 👤 Multi-Tenant Workspace & Private Data Isolation per user
- 📊 Interactive Dashboard & User Onboarding
- 📤 PDF Ingestion & Automatic Vector Embedding
- 📚 Private Paper Library with Search & Filtering
- 💬 Grounded Academic Q&A (RAG + Verifiable Citations)
- 📝 Structured 9-Field Academic Paper Summaries
- ⚖️ Multi-Paper Comparative Analysis (10 Academic Dimensions for >= 2 papers)
- 🔬 Evidence-Grounded Research Gap Discovery
- 📖 10-Section Publication-Grade Literature Reviews
- 🧪 Evaluation Studio & Benchmarking
- ⚙️ System Settings & Diagnostics
"""

from __future__ import annotations

import html
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import pandas as pd
import streamlit as st

from src.auth.auth_service import AuthService, get_auth_service
from src.auth.user_model import User
from src.chunking.text_chunker import TextChunker, chunk_document
from src.citations.citation_engine import CitationEngine
from src.embeddings.embedding_model import get_embedding_model
from src.evaluation.dataset import EvaluationDataset, EvaluationQuestion
from src.evaluation.experiment_logger import ExperimentLogger
from src.evaluation.generation_evaluator import GenerationEvaluator, HumanEvaluationRating
from src.evaluation.retrieval_evaluator import MethodEvaluationSummary, RetrievalEvaluationReport, RetrievalEvaluator
from src.ingestion.pdf_processor import PDFProcessingError, PDFProcessor
from src.intelligence.comparator import PaperComparator, PaperComparisonReport
from src.intelligence.gap_analyzer import ResearchGapAnalyzer, ResearchGapReport
from src.intelligence.lit_reviewer import LiteratureReviewGenerator, LiteratureReviewReport
from src.intelligence.summarizer import PaperSummarizer, PaperSummary
from src.llm.factory import get_llm_provider
from src.llm.provider import BaseLLMProvider
from src.models.chunk import Chunk
from src.models.citation import Citation
from src.models.document import Document
from src.models.retrieval import RetrievalResult
from src.rag.rag_pipeline import RAGPipeline, RAGResponse
from src.reranking.cross_encoder import CrossEncoderReranker, get_reranker
from src.retrieval.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from src.storage.cloudinary_storage import CloudinaryStorage, get_cloudinary_storage
from src.storage.mongo_storage import MongoStorageService, get_mongo_storage
from src.utils.file_utils import (
    clear_all_documents,
    delete_document,
    ensure_directories,
    get_chroma_dir,
    get_chunks_dir,
    get_data_dir,
    get_evaluation_dir,
    get_logs_dir,
    get_papers_dir,
    get_processed_dir,
    get_user_data_dir,
    load_all_document_chunks,
    load_all_processed_documents,
    load_document_chunks,
    load_processed_document,
    save_processed_document,
    save_uploaded_pdf,
)
from src.vectorstore.chroma_store import ChromaStore

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="ResearchMate - AI Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, premium styling
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 35%, #312e81 70%, #4338ca 100%);
    color: white;
    padding: 1.8rem 2.2rem;
    border-radius: 16px;
    margin-bottom: 1.75rem;
    box-shadow: 0 12px 30px -6px rgba(49, 46, 129, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.main-header h1 {
    font-size: 2.25rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.025em;
    color: #ffffff !important;
}

.main-header p {
    font-size: 1.05rem;
    color: #c7d2fe;
    margin-top: 0.4rem;
    margin-bottom: 0;
}

/* User Profile Badge */
.user-profile-box {
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 1rem;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.badge-success {
    background-color: #dcfce7;
    color: #15803d;
    border: 1px solid #bbf7d0;
}

.badge-warning {
    background-color: #fef3c7;
    color: #b45309;
    border: 1px solid #fde68a;
}

.badge-info {
    background-color: #e0e7ff;
    color: #4338ca;
    border: 1px solid #c7d2fe;
}

.badge-purple {
    background-color: #f3e8ff;
    color: #7e22ce;
    border: 1px solid #e9d5ff;
}

.badge-teal {
    background-color: #ccfbf1;
    color: #0f766e;
    border: 1px solid #99f6e4;
}

.badge-rose {
    background-color: #ffe4e6;
    color: #be123c;
    border: 1px solid #fecdd3;
}

/* Welcome Card */
.welcome-card {
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.05) 0%, rgba(99, 102, 241, 0.1) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 14px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.75rem;
}

.welcome-step {
    background: var(--background-color, #ffffff);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 10px;
    padding: 1.25rem;
    height: 100%;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.welcome-step h4 {
    margin-top: 0;
    margin-bottom: 0.5rem;
    color: #4338ca;
}

/* Intelligence Cards */
.intelligence-card {
    background: var(--background-color, #ffffff);
    border: 1px solid rgba(148, 163, 184, 0.28);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
    border-left: 5px solid #4f46e5;
}

.citation-box {
    background: #faf5ff;
    border: 1px solid #e9d5ff;
    border-radius: 10px;
    padding: 1.25rem;
    margin-top: 1rem;
    margin-bottom: 1.25rem;
}

.citation-box h4 {
    color: #6b21a8;
    margin-top: 0;
    margin-bottom: 0.75rem;
}

.snippet-box {
    background-color: #f0fdf4;
    border-left: 4px solid #22c55e;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.85rem;
    font-size: 0.92rem;
    color: #166534;
    line-height: 1.55;
    font-style: italic;
}

.page-text-container {
    background-color: var(--secondary-background-color, #f8fafc);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 8px;
    padding: 1.25rem;
    max-height: 520px;
    overflow-y: auto;
    font-size: 0.95rem;
    line-height: 1.65;
    white-space: pre-wrap;
    word-break: break-word;
}

pre, code {
    font-family: 'JetBrains Mono', monospace !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing Authentication Engine...")
def get_auth() -> AuthService:
    """Instantiate and cache singleton AuthService."""
    return get_auth_service()


@st.cache_resource(show_spinner="Connecting to MongoDB Atlas Cloud Storage...")
def get_storage() -> MongoStorageService:
    """Instantiate and cache singleton MongoStorageService."""
    return get_mongo_storage()


@st.cache_resource(show_spinner="Connecting to Cloudinary PDF Cloud Storage...")
def get_cloudinary() -> CloudinaryStorage:
    """Instantiate and cache singleton CloudinaryStorage."""
    return get_cloudinary_storage()


@st.cache_resource(show_spinner="Initializing Hybrid Retrieval Engine...")
def get_retriever() -> HybridRetriever:
    """Instantiate and cache singleton HybridRetriever."""
    ensure_directories()
    return get_hybrid_retriever()


@st.cache_resource(show_spinner="Initializing Grounded RAG Pipeline...")
def get_rag_pipeline() -> RAGPipeline:
    """Instantiate and cache singleton RAGPipeline."""
    retriever = get_retriever()
    llm = get_llm_provider()
    return RAGPipeline(retriever=retriever, llm_provider=llm)


@st.cache_resource(show_spinner="Initializing Research Intelligence Engines...")
def get_intelligence_suite():
    """Instantiate and cache summarizer, comparator, gap analyzer, and lit reviewer."""
    retriever = get_retriever()
    llm = get_llm_provider()
    summarizer = PaperSummarizer(retriever=retriever, llm_provider=llm)
    comparator = PaperComparator(retriever=retriever, llm_provider=llm, summarizer=summarizer)
    gap_analyzer = ResearchGapAnalyzer(retriever=retriever, llm_provider=llm)
    lit_reviewer = LiteratureReviewGenerator(retriever=retriever, llm_provider=llm)
    return summarizer, comparator, gap_analyzer, lit_reviewer


@st.cache_resource(show_spinner="Initializing Evaluation Engine...")
def get_evaluation_suite():
    """Instantiate and cache evaluation dataset, evaluator, and experiment logger."""
    retriever = get_retriever()
    dataset = EvaluationDataset()
    evaluator = RetrievalEvaluator(retriever=retriever, dataset=dataset)
    logger = ExperimentLogger()
    return dataset, evaluator, logger


def render_session_sync_bridge(session_token: Optional[str] = None) -> None:
    """Synchronize session token between Streamlit and browser localStorage / URL.

    Ensures seamless persistent login across page refreshes, tab re-openings, and navigation.
    """
    token_json = json.dumps(session_token or "")
    js_code = f"""
    <script>
    (function() {{
        try {{
            const currentToken = {token_json};
            const parentWin = window.parent;
            if (!parentWin) return;
            
            const urlParams = new URLSearchParams(parentWin.location.search);
            const urlToken = urlParams.get('session_token');

            if (currentToken && currentToken !== "") {{
                // Active session: persist in browser localStorage
                parentWin.localStorage.setItem('rm_session_token', currentToken);
                if (urlToken !== currentToken) {{
                    urlParams.set('session_token', currentToken);
                    parentWin.history.replaceState(null, '', '?' + urlParams.toString());
                }}
            }} else {{
                // If not authenticated in current run, check if browser has a saved session token
                const savedToken = parentWin.localStorage.getItem('rm_session_token');
                if (savedToken && (!urlToken || urlToken !== savedToken)) {{
                    urlParams.set('session_token', savedToken);
                    parentWin.location.search = '?' + urlParams.toString();
                }}
            }}
        }} catch (e) {{
            // Ignore cross-origin issues if any
        }}
    }})();
    </script>
    """
    import streamlit.components.v1 as components
    components.html(js_code, height=0, width=0)


def init_app_state(auth_service: Optional[AuthService] = None) -> None:
    """Initialize session state variables and restore active user session from token."""
    ensure_directories()
    if "authenticated_user" not in st.session_state:
        st.session_state.authenticated_user = None
    if "session_token" not in st.session_state:
        st.session_state.session_token = None
    if "selected_doc_id" not in st.session_state:
        st.session_state.selected_doc_id = None
    if "processor" not in st.session_state:
        st.session_state.processor = PDFProcessor(ignore_empty_pages=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "eval_report" not in st.session_state:
        st.session_state.eval_report = None

    # Check query_params for persistent session token
    if st.session_state.authenticated_user is None and auth_service is not None:
        raw_token = st.query_params.get("session_token")
        if isinstance(raw_token, list):
            raw_token = raw_token[0] if raw_token else None
        if raw_token:
            token = str(raw_token).strip()
            user = auth_service.get_user_by_session_token(token)
            if user:
                st.session_state.authenticated_user = user
                st.session_state.session_token = token
            else:
                try:
                    del st.query_params["session_token"]
                except Exception:
                    pass


# =====================================================================
# AUTHENTICATION PORTAL (LOGIN & SIGN UP)
# =====================================================================
def render_auth_portal(auth_service: AuthService) -> None:
    """Render authentication portal when no user is logged in."""
    st.markdown(
        """
        <div class="main-header" style="text-align: center; padding: 2.2rem 2rem;">
            <h1>📚 ResearchMate</h1>
            <p style="font-size: 1.15rem; max-width: 650px; margin: 0.5rem auto 0 auto;">
                AI-Powered Multi-Paper Research Intelligence, Comparative Analysis & Grounded Q&A
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_space1, col_center, col_space2 = st.columns([1, 2, 1])

    with col_center:
        auth_tab_login, auth_tab_register = st.tabs(["🔑 Sign In", "📝 Create Free Account"])

        with auth_tab_login:
            st.markdown("#### Welcome Back")
            st.caption("Sign in to access your private research workspace and paper library.")

            with st.form("login_form"):
                login_email = st.text_input("Email Address:", placeholder="researcher@university.edu")
                login_password = st.text_input("Password:", type="password", placeholder="••••••••")
                login_btn = st.form_submit_button("Sign In to Workspace", type="primary", use_container_width=True)

                if login_btn:
                    if not login_email or not login_password:
                        st.error("Please enter both email and password.")
                    else:
                        with st.spinner("Authenticating credentials..."):
                            success, msg, user = auth_service.authenticate_user(login_email, login_password)
                        if success and user:
                            token = auth_service.create_session(user.user_id)
                            st.query_params["session_token"] = token
                            st.session_state.authenticated_user = user
                            st.session_state.session_token = token
                            st.session_state.chat_history = []
                            st.session_state.selected_doc_id = None
                            render_session_sync_bridge(token)
                            st.success(f"Welcome back, {user.full_name}!")
                            st.rerun()
                        else:
                            st.error(msg)

        with auth_tab_register:
            st.markdown("#### Create New Account")
            st.caption("Register to get an isolated private research workspace and vector store.")

            with st.form("register_form"):
                reg_name = st.text_input("Full Name:", placeholder="Dr. Jane Doe")
                reg_email = st.text_input("Email Address:", placeholder="jane.doe@university.edu")
                reg_password = st.text_input("Password (min 6 characters):", type="password", placeholder="••••••••")
                reg_confirm = st.text_input("Confirm Password:", type="password", placeholder="••••••••")
                reg_btn = st.form_submit_button("Create Account", type="primary", use_container_width=True)

                if reg_btn:
                    if reg_password != reg_confirm:
                        st.error("Passwords do not match.")
                    else:
                        with st.spinner("Creating account..."):
                            success, msg, user = auth_service.register_user(reg_email, reg_password, reg_name)
                        if success and user:
                            token = auth_service.create_session(user.user_id)
                            st.query_params["session_token"] = token
                            st.session_state.authenticated_user = user
                            st.session_state.session_token = token
                            st.session_state.chat_history = []
                            st.session_state.selected_doc_id = None
                            render_session_sync_bridge(token)
                            st.success("Account created successfully!")
                            st.rerun()
                        else:
                            st.error(msg)

        st.markdown("---")
        # Features highlight
        st.markdown("##### 🌟 What You Can Do with ResearchMate:")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("📁 **Private Library**\nIsolated storage per user")
        with c2:
            st.markdown("⚖️ **Compare Papers**\n10-dimension matrix for >=2 papers")
        with c3:
            st.markdown("💬 **Grounded Q&A**\nVerifiable inline citations [1]")

        st.markdown("---")
        db_badge = "🟢 **MongoDB Connected**" if auth_service.is_mongo_connected else "⚡ **Local Auth Storage Active**"
        st.caption(f"Database Backend: {db_badge} • Total Users: `{auth_service.get_all_users_count()}`")


# =====================================================================
# AUTHENTICATED SIDEBAR & HEADER
# =====================================================================
def render_sidebar(
    current_user: User,
    documents: List[Document],
    retriever: HybridRetriever,
    rag_pipeline: RAGPipeline,
) -> None:
    """Render sidebar with user profile, library status, and engine health."""
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 0.85rem; padding: 0.25rem 0 1rem 0;">
                <div style="
                    background: linear-gradient(135deg, #4338ca 0%, #6366f1 50%, #8b5cf6 100%);
                    width: 48px;
                    height: 48px;
                    border-radius: 14px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 8px 20px -4px rgba(79, 70, 229, 0.45);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    flex-shrink: 0;
                ">
                    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M4 19.5C4 18.837 4.26339 18.2011 4.73223 17.7322C5.20107 17.2634 5.83696 17 6.5 17H20" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M6.5 2H20V22H6.5C5.83696 22 5.20107 21.7366 4.73223 21.2678C4.26339 20.7989 4 20.163 4 19.5V4.5C4 3.83696 4.26339 3.20107 4.73223 2.73223C5.20107 2.26339 5.83696 2 6.5 2Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M9 7H15M9 11H13" stroke="#c7d2fe" stroke-width="1.75" stroke-linecap="round"/>
                    </svg>
                </div>
                <div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #1e1b4b; line-height: 1.15; letter-spacing: -0.02em;">ResearchMate</div>
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px;">AI Research Platform</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # User profile box
        st.markdown(
            f"""
            <div class="user-profile-box">
                <div style="font-weight: 700; color: #1e1b4b; font-size: 0.95rem;">👤 {html.escape(current_user.full_name)}</div>
                <div style="font-size: 0.8rem; color: #64748b;">{html.escape(current_user.email)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        storage = get_storage()
        if storage.is_connected:
            st.markdown(
                """
                <div style="background: rgba(34, 197, 94, 0.08); border: 1px solid rgba(34, 197, 94, 0.25); border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.75rem;">
                    <div style="font-size: 0.82rem; font-weight: 600; color: #15803d;">☁️ MongoDB Atlas Cloud Active</div>
                    <div style="font-size: 0.72rem; color: #64748b;">Synced: Users • Papers • Chunks • Summaries • Chats</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("🚪 Sign Out", use_container_width=True):
            token = st.session_state.get("session_token") or st.query_params.get("session_token")
            if token:
                if isinstance(token, list):
                    token = token[0]
                auth = get_auth()
                auth.invalidate_session(str(token).strip())
            try:
                st.query_params.clear()
            except Exception:
                pass
            st.session_state.authenticated_user = None
            st.session_state.session_token = None
            st.session_state.chat_history = []
            st.session_state.selected_doc_id = None
            
            import streamlit.components.v1 as components
            clear_js = """
            <script>
            try {
                const parentWin = window.parent;
                if (parentWin) {
                    parentWin.localStorage.removeItem('rm_session_token');
                    const urlParams = new URLSearchParams(parentWin.location.search);
                    urlParams.delete('session_token');
                    parentWin.history.replaceState(null, '', parentWin.location.pathname);
                }
            } catch (e) {}
            </script>
            """
            components.html(clear_js, height=0, width=0)
            st.rerun()

        st.markdown("---")
        st.markdown("### 📚 My Paper Library")
        st.write(f"**Uploaded Papers:** `{len(documents)}`")

        if len(documents) >= 2:
            st.success("🟢 Ready for Multi-Paper Comparison (>= 2 papers)")
        elif len(documents) == 1:
            st.info("ℹ️ 1 paper uploaded. Add 1 more to unlock Comparison & Lit Review.")
        else:
            st.warning("⚠️ No papers uploaded yet.")

        st.markdown("---")
        st.markdown("### 🤖 System & Model Health")

        llm = rag_pipeline.llm_provider
        if llm.is_available():
            st.success(f"🟢 **AI Engine:** `{llm.get_provider_name().upper()}` ({llm.get_model_name()})")
        else:
            st.warning(f"⚠️ **AI Engine:** `{llm.get_provider_name().upper()}` (API Key Missing in `.env`)")

        try:
            stats = retriever.get_stats(user_id=current_user.user_id)
            reranker_stats = retriever.reranker.get_stats()
            st.write(f"🧠 **ChromaDB Vectors:** `{stats['chroma_vectors']:,}`")
            st.write(f"📖 **BM25 Chunks:** `{stats['bm25_chunks']:,}`")
            st.write(f"🎯 **Reranker:** `{reranker_stats['model_name'].split('/')[-1]}`")
        except Exception as e:
            st.error(f"Retrieval Engine Notice: {e}")

        st.markdown("---")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("🔄 Refresh", use_container_width=True):
                retriever.sync_from_storage(user_id=current_user.user_id)
                st.rerun()
        with col_s2:
            if st.button("🗑️ Reset", use_container_width=True, help="Clear your uploaded papers and vectors"):
                clear_all_documents(user_id=current_user.user_id)
                retriever.clear_all(user_id=current_user.user_id)
                st.session_state.selected_doc_id = None
                st.session_state.chat_history = []
                st.success("Library reset successfully.")
                st.rerun()


def render_header(current_user: User) -> None:
    """Render the main hero header banner."""
    st.markdown(
        f"""
        <div class="main-header">
            <h1>ResearchMate</h1>
            <p>Welcome back, <strong>{html.escape(current_user.full_name)}</strong> • AI Research Intelligence & Comparative Analysis Workspace</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def handle_file_uploads(uploaded_files: list, retriever: HybridRetriever, user_id: str) -> None:
    """Process, upload to Cloudinary, chunk, embed, and index PDF files for a specific user."""
    if not uploaded_files:
        return

    processor: PDFProcessor = st.session_state.processor
    cloudinary = get_cloudinary()
    storage = get_storage()
    success_count = 0
    warnings: List[str] = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, uploaded_file in enumerate(uploaded_files):
        file_bytes = uploaded_file.getvalue()
        orig_name = uploaded_file.name

        # 1. Validate file format and content
        if not file_bytes or len(file_bytes) < 32:
            st.error(f"Cannot process '{orig_name}': File is empty or too small.")
            continue

        if not orig_name.lower().endswith(".pdf") and not file_bytes.startswith(b"%PDF"):
            st.error(f"Cannot process '{orig_name}': File is not a valid PDF document.")
            continue

        status_text.text(f"Processing & Indexing ({idx + 1}/{len(uploaded_files)}): {orig_name}...")

        try:
            # 2. Extract content and sections using PyMuPDF from memory
            doc = processor.process_file(
                file_input=file_bytes,
                filename=orig_name,
                storage_filename=orig_name,
            )
            doc.metadata.user_id = user_id

            # 3. Always save local PDF for resilient local fallback and instant offline access
            save_uploaded_pdf(file_bytes, orig_name, user_id=user_id)

            # 4. Upload PDF to Cloudinary under user's isolated path (if configured)
            if cloudinary.is_configured:
                upload_res = cloudinary.upload_pdf(
                    file_input=file_bytes,
                    user_id=user_id,
                    document_id=doc.document_id,
                    filename=orig_name,
                )
                if upload_res.get("success"):
                    doc.metadata.cloudinary_public_id = upload_res.get("public_id")
                    doc.metadata.cloudinary_url = upload_res.get("secure_url")
                    doc.metadata.status = "indexed"
                else:
                    doc.metadata.status = "indexed"
                    err_msg = upload_res.get("error", "Cloudinary upload unsuccessful")
                    warnings.append(f"Cloud upload notice for '{orig_name}': {err_msg} (Paper preserved in local storage).")
            else:
                doc.metadata.status = "indexed"

            # 5. Save processed document JSON to local cache
            save_processed_document(doc, user_id=user_id)

            # 6. Automatically chunk, create vector embeddings in ChromaDB, and index in BM25 with user_id
            chunk_count, chunks = retriever.index_document(doc, overwrite=True, user_id=user_id)

            # 7. Sync document and chunks directly to MongoDB Atlas cloud database
            storage.save_document(doc, user_id=user_id)
            if chunks:
                storage.save_chunks(doc.document_id, chunks, user_id=user_id)

            success_count += 1
            st.session_state.selected_doc_id = doc.document_id
        except Exception as e:
            st.error(f"Failed to process {orig_name}: {e}")

        progress_bar.progress((idx + 1) / len(uploaded_files))

    status_text.empty()
    progress_bar.empty()

    if warnings:
        for w in warnings:
            st.warning(w)

    if success_count > 0:
        st.success(f"✅ Successfully ingested, uploaded, and indexed {success_count} research paper(s)!")
        st.rerun()


# =====================================================================
# TAB 1: DASHBOARD
# =====================================================================
def render_dashboard_tab(
    current_user: User,
    documents: List[Document],
    retriever: HybridRetriever,
    logger: ExperimentLogger,
) -> None:
    """Render user-scoped overview dashboard or welcoming onboarding guide."""
    total_papers = len(documents)
    total_pages = sum(d.metadata.page_count for d in documents)
    total_words = sum(d.total_words for d in documents)

    try:
        stats = retriever.get_stats(user_id=current_user.user_id)
        total_vectors = stats["chroma_vectors"]
    except Exception:
        total_vectors = 0

    recent_runs = logger.load_recent_experiments(limit=50)

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="📚 My Papers", value=f"{total_papers}")
    with col2:
        st.metric(label="📑 Extracted Pages", value=f"{total_pages:,}")
    with col3:
        st.metric(label="🧩 My Indexed Chunks", value=f"{total_vectors:,}")
    with col4:
        st.metric(label="💬 Q&A Queries Logged", value=f"{len(recent_runs):,}")

    st.markdown("---")

    # If new user / empty library, display welcoming onboarding workflow
    if total_papers == 0:
        st.markdown(
            f"""
            <div class="welcome-card">
                <h3 style="margin-top: 0; color: #312e81;">👋 Welcome to your workspace, {html.escape(current_user.full_name)}!</h3>
                <p style="color: #475569; font-size: 1.05rem; margin-bottom: 1.25rem;">
                    Your private library is currently empty. Get started by uploading <strong>2 or more research papers (PDFs)</strong> to unlock cross-paper comparison, grounded Q&A, and literature synthesis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c_step1, c_step2, c_step3 = st.columns(3)
        with c_step1:
            st.markdown(
                """
                <div class="welcome-step">
                    <h4>1. 📤 Upload Papers</h4>
                    <p style="color: #64748b; font-size: 0.92rem;">
                        Go to the <strong>Upload Papers</strong> tab and upload 2 or more PDF research papers. Text is automatically parsed, section-split, and vector-embedded.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_step2:
            st.markdown(
                """
                <div class="welcome-step">
                    <h4>2. 💬 Grounded Q&A</h4>
                    <p style="color: #64748b; font-size: 0.92rem;">
                        Ask factual, methodology, or results questions in <strong>Research Q&A</strong>. Every answer is grounded strictly in retrieved passages with verifiable citations [1].
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_step3:
            st.markdown(
                """
                <div class="welcome-step">
                    <h4>3. ⚖️ Compare & Discover</h4>
                    <p style="color: #64748b; font-size: 0.92rem;">
                        Use <strong>Compare Papers</strong> to generate a 10-dimension comparison matrix, extract evidence-grounded research gaps, and build literature reviews.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### ⚙️ Pipeline Specifications")
        st.markdown(
            """
            - **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-d, local)
            - **Vector Store:** ChromaDB (Cosine Distance Persistence)
            - **Keyword Engine:** BM25 (Academic Tokenization)
            - **Fusion Strategy:** Reciprocal Rank Fusion (RRF $k=60$)
            - **Cross-Encoder Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
            - **Workspace Storage:** Isolated user workspace under `data/users/`
            """
        )
        return

    # Active library view for populated library
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### 📚 Active Library Papers")
        table_data = []
        for d in documents:
            is_idx = retriever.chroma_store.is_document_indexed(d.document_id, user_id=current_user.user_id)
            chunk_c = retriever.chroma_store.get_document_chunk_count(d.document_id, user_id=current_user.user_id) if is_idx else 0
            table_data.append({
                "Title": d.title,
                "Filename": d.metadata.original_filename,
                "Pages": d.page_count,
                "Words": f"{d.total_words:,}",
                "Index Status": f"🟢 Indexed ({chunk_c} chunks)" if is_idx else "⏳ Unindexed",
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    with col_right:
        st.markdown("#### ⚙️ Pipeline Specifications")
        st.markdown(
            """
            - **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-d, local)
            - **Vector Store:** ChromaDB (Cosine Distance Persistence)
            - **Keyword Engine:** BM25 (Academic Tokenization)
            - **Fusion Strategy:** Reciprocal Rank Fusion (RRF $k=60$)
            - **Cross-Encoder Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
            - **User Workspace:** `data/users/{user_id}/`
            """
        )

    st.markdown("---")
    st.markdown("#### 🕒 Recent Activity & Query History")
    if recent_runs:
        runs_df = pd.DataFrame(recent_runs[:10])
        display_cols = [c for c in ["timestamp", "query", "retrieval_method", "model", "latency_seconds", "faithfulness_score"] if c in runs_df.columns]
        st.dataframe(runs_df[display_cols], use_container_width=True)
    else:
        st.caption("No queries logged yet. Ask questions in the Research Q&A tab to populate activity.")


# =====================================================================
# TAB 2: UPLOAD PAPERS
# =====================================================================
def render_upload_tab(current_user: User, retriever: HybridRetriever) -> None:
    """Render upload research papers tab with multi-file drag and drop."""
    st.markdown("### 📤 Upload Research Papers")
    st.markdown(
        "Upload **2 or more research papers (PDFs)** to your private library to enable side-by-side comparative analysis, "
        "evidence-grounded Q&A, research gap discovery, and literature review generation."
    )

    uploaded_files = st.file_uploader(
        "Upload Research Papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF research papers",
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.info(f"📁 **{len(uploaded_files)}** file(s) selected for ingestion.")
        if st.button("🚀 Process & Ingest Papers", type="primary", use_container_width=True):
            handle_file_uploads(uploaded_files, retriever, user_id=current_user.user_id)


# =====================================================================
# TAB 3: ENHANCED PAPER LIBRARY & SEARCH
# =====================================================================
def render_library_tab(current_user: User, documents: List[Document], retriever: HybridRetriever) -> None:
    """Render enhanced paper library with title, author, and keyword filtering."""
    st.markdown("### 📚 My Research Paper Library")
    st.markdown("Search, inspect, and manage research papers in your private collection.")

    if not documents:
        st.info("No research papers in library. Upload PDFs in the **Upload Papers** tab to get started.")
        return

    # Search and Filtering Controls
    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
        search_kw = st.text_input("🔎 Search by Title / Keyword:", placeholder="e.g. attention, transformer, healthcare...")
    with col_s2:
        author_kw = st.text_input("👤 Filter by Author:", placeholder="e.g. Vaswani, Hinton...")
    with col_s3:
        status_filter = st.selectbox("Index Status:", options=["All", "Indexed Only", "Unindexed Only"])

    # Filter documents
    filtered_docs = []
    for d in documents:
        if search_kw:
            q_low = search_kw.lower()
            if q_low not in d.title.lower() and q_low not in d.metadata.original_filename.lower():
                continue

        if author_kw:
            a_low = author_kw.lower()
            authors_str = " ".join(d.metadata.authors).lower()
            if a_low not in authors_str:
                continue

        is_idx = retriever.chroma_store.is_document_indexed(d.document_id, user_id=current_user.user_id)
        if status_filter == "Indexed Only" and not is_idx:
            continue
        if status_filter == "Unindexed Only" and is_idx:
            continue

        filtered_docs.append(d)

    st.caption(f"Showing **{len(filtered_docs)}** of **{len(documents)}** papers.")

    if not filtered_docs:
        st.warning("No papers matched your search criteria.")
        return

    doc_options = {d.document_id: f"{d.title} ({d.metadata.original_filename})" for d in filtered_docs}

    if st.session_state.selected_doc_id not in doc_options:
        st.session_state.selected_doc_id = filtered_docs[0].document_id

    col_select, col_act1, col_act2, col_act3 = st.columns([3, 1, 1, 1])
    with col_select:
        selected_id = st.selectbox(
            "Select Paper to Inspect:",
            options=list(doc_options.keys()),
            format_func=lambda doc_id: doc_options[doc_id],
            index=list(doc_options.keys()).index(st.session_state.selected_doc_id),
            label_visibility="collapsed",
        )
        st.session_state.selected_doc_id = selected_id

    selected_doc = get_storage().load_document(st.session_state.selected_doc_id, user_id=current_user.user_id) or load_processed_document(st.session_state.selected_doc_id, user_id=current_user.user_id)

    with col_act1:
        if selected_doc:
            st.download_button(
                label="📥 Export JSON",
                data=selected_doc.to_json(),
                file_name=f"{selected_doc.document_id}.json",
                mime="application/json",
                use_container_width=True,
            )

    with col_act2:
        if selected_doc:
            c_storage = get_cloudinary()
            sec_url = selected_doc.metadata.cloudinary_url or (c_storage.generate_secure_url(current_user.user_id, selected_doc.document_id) if c_storage.is_configured else None)
            if sec_url:
                st.link_button("☁️ View PDF", url=sec_url, use_container_width=True)
            else:
                # Fallback to downloading local copy if present
                local_pdf = get_papers_dir(current_user.user_id) / selected_doc.metadata.filename
                if local_pdf.exists():
                    st.download_button("📥 PDF File", data=local_pdf.read_bytes(), file_name=selected_doc.metadata.original_filename, use_container_width=True)

    with col_act3:
        if st.button("🗑️ Delete Paper", type="secondary", use_container_width=True):
            if selected_doc:
                # 1. Delete from Cloudinary
                get_cloudinary().delete_pdf(current_user.user_id, selected_doc.document_id)
                # 2. Delete from ChromaDB & BM25
                retriever.delete_document(selected_doc.document_id, user_id=current_user.user_id)
                # 3. Delete from local cache
                delete_document(selected_doc.document_id, user_id=current_user.user_id)
                # 4. Delete from MongoDB Atlas
                get_storage().delete_document(selected_doc.document_id, user_id=current_user.user_id)
                st.success(f"Deleted '{selected_doc.metadata.original_filename}'.")
                st.session_state.selected_doc_id = None
                st.rerun()

    # Detailed Paper Inspector
    if selected_doc:
        st.markdown(f"#### 🔍 Document Details: *{selected_doc.title}*")
        meta = selected_doc.metadata
        authors_str = ", ".join(meta.authors) if meta.authors else "Not detected"
        is_indexed = retriever.chroma_store.is_document_indexed(selected_doc.document_id, user_id=current_user.user_id)
        chunk_count = retriever.chroma_store.get_document_chunk_count(selected_doc.document_id, user_id=current_user.user_id) if is_indexed else 0
        cloud_badge = "☁️ Cloudinary Active" if meta.cloudinary_public_id else "💾 Local Cache"

        st.markdown(
            f"""
            <div style="background-color: var(--secondary-background-color, #f1f5f9); padding: 1.1rem 1.3rem; border-radius: 12px; margin-bottom: 1.25rem; border-left: 4px solid #6366f1;">
                <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 0.35rem;">{meta.title}</div>
                <div style="color: #475569; font-size: 0.92rem; margin-bottom: 0.5rem;"><strong>Authors:</strong> {authors_str}</div>
                <div style="display: flex; gap: 1.2rem; flex-wrap: wrap; font-size: 0.85rem; color: #64748b;">
                    <span>📁 <strong>File:</strong> {meta.original_filename}</span>
                    <span>📄 <strong>Pages:</strong> {meta.page_count}</span>
                    <span>📊 <strong>Words:</strong> {selected_doc.total_words:,}</span>
                    <span>🧠 <strong>Index:</strong> {'✅ ' + str(chunk_count) + ' chunks' if is_indexed else '⏳ Not Indexed'}</span>
                    <span>☁️ <strong>Storage:</strong> {cloud_badge}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_pages, tab_chunks, tab_full_text = st.tabs(["📖 Page View", "🧩 Chunks", "📜 Full Extracted Text"])

        with tab_pages:
            page_numbers = [p.page_number for p in selected_doc.pages]
            if page_numbers:
                sel_page_num = st.selectbox("Select Page:", options=page_numbers, format_func=lambda n: f"Page {n}")
                page_obj = selected_doc.get_page(sel_page_num)
                if page_obj:
                    st.markdown(f'<div class="page-text-container">{page_obj.text}</div>', unsafe_allow_html=True)

        with tab_chunks:
            cached_chunks = load_document_chunks(selected_doc.document_id, user_id=current_user.user_id)
            if not cached_chunks:
                chunker = TextChunker()
                cached_chunks = chunker.chunk_document(selected_doc)

            if cached_chunks:
                st.write(f"Total **{len(cached_chunks)}** structured chunks:")
                for chk in cached_chunks:
                    with st.expander(f"🧩 Chunk #{chk.chunk_index} • Page {chk.page_number} • Section: '{chk.section}' ({chk.metadata.char_count} chars)"):
                        st.code(chk.text, language="markdown")

        with tab_full_text:
            full_text = selected_doc.get_full_text()
            st.download_button("📥 Download Full Text (.txt)", data=full_text, file_name=f"{selected_doc.document_id}.txt")
            st.text_area("Full Content", value=full_text, height=350, disabled=True)


# =====================================================================
# TAB 4: RESEARCH Q&A (GROUNDED RAG)
# =====================================================================
def render_qa_tab(
    current_user: User,
    documents: List[Document],
    retriever: HybridRetriever,
    rag_pipeline: RAGPipeline,
    logger: ExperimentLogger,
) -> None:
    """Render Grounded RAG Chat tab with citation attribution and source inspection."""
    st.markdown("### 💬 Research Q&A (Grounded Academic RAG)")
    st.markdown(
        "Ask factual, methodology, results, or comparative questions across your private research library. "
        "Answers are grounded exclusively in retrieved paper passages and validated for citation fidelity."
    )

    if not documents:
        st.info("No research papers in library. Please upload PDF papers in the **Upload Papers** tab to start asking questions.")
        return

    llm = rag_pipeline.llm_provider
    if not llm.is_available():
        st.warning("⚠️ **AI API Setup**: Add your API key in `.env` to enable live LLM generation.")

    with st.expander("⚙️ Retrieval & Generation Parameters", expanded=False):
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            retrieval_strategy = st.selectbox(
                "Retrieval Strategy:",
                options=["hybrid", "weighted", "dense", "sparse"],
                format_func=lambda s: {
                    "hybrid": "🔀 Hybrid RRF (Dense + BM25)",
                    "weighted": "📊 Weighted Score Fusion",
                    "dense": "🧠 Dense Vector Only (ChromaDB)",
                    "sparse": "📖 Sparse Keyword Only (BM25)",
                }[s],
                index=0,
            )
        doc_filter_options = {"all": "🌐 All My Papers"}
        for d in documents:
            doc_filter_options[d.document_id] = f"📄 {d.title}"

        with col_c2:
            filter_doc = st.selectbox("Paper Scope:", options=list(doc_filter_options.keys()), format_func=lambda k: doc_filter_options[k])
        with col_c3:
            final_top_k = st.slider("Context Chunks (Top-K):", min_value=1, max_value=10, value=5)
            candidate_k = st.slider("Candidate Pool:", min_value=5, max_value=30, value=20)
        with col_c4:
            enable_reranker = st.toggle("Enable Cross-Encoder Reranker", value=True)
            temperature = st.slider("LLM Temperature:", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    # Quick Inquiry Presets
    st.markdown("**💡 Quick Research Inquiries:**")
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    preset_query = None

    with col_q1:
        if st.button("🎯 Core Contributions", use_container_width=True):
            preset_query = "What are the primary research contributions and core objectives of the papers?"
    with col_q2:
        if st.button("⚙️ Methodology & Arch", use_container_width=True):
            preset_query = "Explain the methodology, model architecture, and algorithms used."
    with col_q3:
        if st.button("📊 Datasets & Metrics", use_container_width=True):
            preset_query = "What datasets and experimental benchmarks were used to validate the results?"
    with col_q4:
        if st.button("⚠️ Limitations & Gaps", use_container_width=True):
            preset_query = "What are the primary limitations, constraints, and future work mentioned?"

    col_chat_t, col_chat_c = st.columns([5, 1])
    with col_chat_c:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            get_storage().clear_user_chat_history(user_id=current_user.user_id)
            st.session_state.chat_history = []
            st.rerun()

    # Render past conversation
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                resp: Optional[RAGResponse] = msg.get("response_obj")
                if resp:
                    val = resp.validation
                    if resp.is_insufficient_evidence:
                        status_badge = '<span class="badge badge-warning">⚠️ Insufficient Evidence</span>'
                    elif val.is_valid:
                        status_badge = f'<span class="badge badge-success">🟢 Grounded & Validated ({val.grounding_score * 100:.0f}%)</span>'
                    else:
                        status_badge = '<span class="badge badge-rose">⚠️ Citation Validation Notice</span>'

                    meta_header = (
                        f"<div style='display: flex; gap: 0.75rem; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap;'>"
                        f"{status_badge}"
                        f"<span class='badge badge-info'>🤖 {resp.provider.upper()}: {resp.model}</span>"
                        f"<span class='badge badge-teal'>⚡ {resp.latency_seconds:.2f}s</span>"
                        f"<span class='badge badge-purple'>📚 {len(resp.citations)} Sources</span>"
                        f"</div>"
                    )
                    st.markdown(meta_header, unsafe_allow_html=True)
                    st.markdown(resp.answer)

                    if resp.citations:
                        st.markdown(
                            f"""
                            <div class="citation-box">
                                <h4>📚 Grounded Source Citations ({len(resp.citations)} references)</h4>
                                <div style="font-size: 0.9rem; line-height: 1.7; color: #4b5563;">
                                    {'<br/>'.join([f"<strong>{c.citation_id}</strong> <em>{html.escape(c.paper_title)}</em> — <code>{html.escape(c.filename)}</code> (Page <strong>{c.page_number}</strong>, Section: <em>{html.escape(c.section)}</em>)" for c in resp.citations])}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        with st.expander(f"🔍 Inspect Evidence Passages ({len(resp.citations)})", expanded=False):
                            for cit in resp.citations:
                                score_label = f"Reranker: {cit.reranker_score:+.4f}" if cit.reranker_score is not None else f"RRF: {cit.rrf_score or 0:.4f}"
                                st.markdown(f"**{cit.citation_id}** **{cit.paper_title}** — `{cit.filename}` (Page **{cit.page_number}**, Section: *{cit.section}*) — `{score_label}`")
                                if cit.source_snippet:
                                    st.markdown(f"> *\"{cit.source_snippet}\"*")
                                st.markdown(f'<div class="page-text-container">{cit.source_text}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(msg["content"])

    user_query = st.chat_input("Ask a research question about your uploaded papers...")
    active_query = preset_query or user_query

    if active_query:
        st.session_state.chat_history.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.write(active_query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence, reranking passages, and generating grounded answer..."):
                filter_id = None if filter_doc == "all" else filter_doc
                chat_context = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history[:-1]]

                response: RAGResponse = rag_pipeline.answer_question(
                    query=active_query,
                    chat_history=chat_context,
                    retrieval_mode=retrieval_strategy,
                    candidate_k=candidate_k,
                    top_k=final_top_k,
                    filter_doc_id=filter_id,
                    rerank=enable_reranker,
                    temperature=temperature,
                    user_id=current_user.user_id,
                )

                # Log experiment
                logger.log_experiment_run(
                    response=response,
                    retrieval_method=f"{retrieval_strategy}{'+rerank' if enable_reranker else ''}",
                    candidate_k=candidate_k,
                    final_top_k=final_top_k,
                )

                # Save chat turn to MongoDB Atlas cloud database
                storage = get_storage()
                storage.save_chat_turn(
                    user_id=current_user.user_id,
                    role="user",
                    content=active_query,
                    filter_document_id=filter_id,
                )
                storage.save_chat_turn(
                    user_id=current_user.user_id,
                    role="assistant",
                    content=response.answer,
                    citations=[c.model_dump() for c in response.citations],
                    filter_document_id=filter_id,
                )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.answer,
                    "response_obj": response,
                })
                st.rerun()


# =====================================================================
# TAB 5: PAPER SUMMARIZATION
# =====================================================================
def render_summarization_tab(current_user: User, documents: List[Document], summarizer: PaperSummarizer) -> None:
    """Render structured 9-field academic summarization tab."""
    st.markdown("### 📝 Structured Academic Paper Summarization")
    st.markdown(
        "Generate a structured 9-field academic summary for any uploaded research paper. "
        "Summaries are grounded strictly in targeted retrieved passages with verifiable citations."
    )

    if not documents:
        st.info("No papers available in library. Upload PDFs in the **Upload Papers** tab first.")
        return

    doc_options = {d.document_id: f"{d.title} ({d.metadata.original_filename})" for d in documents}
    col_sel, col_btn = st.columns([3, 1])

    with col_sel:
        selected_id = st.selectbox("Select Paper to Summarize:", options=list(doc_options.keys()), format_func=lambda k: doc_options[k])

    with col_btn:
        force_refresh = st.checkbox("Force Re-Generate", value=False)
        generate_btn = st.button("⚡ Generate Summary", type="primary", use_container_width=True)

    if generate_btn or selected_id:
        with st.spinner("Extracting structured 9-field academic summary..."):
            summary: PaperSummary = summarizer.summarize_paper(selected_id, force_refresh=force_refresh, user_id=current_user.user_id)
            # Sync summary to MongoDB Atlas cloud database
            get_storage().save_summary(summary.document_id, summary.model_dump(), user_id=current_user.user_id)

        cache_badge = "🟢 (Cached)" if summary.is_cached else "⚡ (Live Generated)"
        st.markdown(f"#### 📄 Summary: *{summary.paper_title}* `{cache_badge}`")

        # 9 Structured Fields
        fields = [
            ("1. 🎯 Research Problem", summary.research_problem),
            ("2. 💡 Objective & Contributions", summary.objective),
            ("3. ⚙️ Methodology", summary.methodology),
            ("4. 📊 Dataset & Corpus", summary.dataset),
            ("5. 🧠 Model / Architecture", summary.model_architecture),
            ("6. 🔬 Experimental Setup", summary.experimental_setup),
            ("7. 📈 Main Results & Metrics", summary.main_results),
            ("8. ⚠️ Limitations & Constraints", summary.limitations),
            ("9. 🏁 Conclusion & Future Directions", summary.conclusion),
        ]

        for title, content in fields:
            with st.container():
                st.markdown(f"**{title}**")
                st.markdown(f"> {content}")
                st.markdown("<hr style='margin: 0.4rem 0 0.8rem 0; opacity: 0.15;'/>", unsafe_allow_html=True)

        if summary.citations:
            st.markdown(
                f"""
                <div class="citation-box">
                    <h4>📚 Grounded Source Citations ({len(summary.citations)} references)</h4>
                    <div style="font-size: 0.9rem; line-height: 1.7; color: #4b5563;">
                        {'<br/>'.join([f"<strong>{c.citation_id}</strong> <em>{html.escape(c.paper_title)}</em> — <code>{html.escape(c.filename)}</code> (Page <strong>{c.page_number}</strong>, Section: <em>{html.escape(c.section)}</em>)" for c in summary.citations])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("📋 Copy Publication-Ready Markdown Summary", expanded=False):
            st.code(summary.to_markdown(), language="markdown")


# =====================================================================
# TAB 6: PAPER COMPARISON (REQUIRES >= 2 PAPERS)
# =====================================================================
def render_comparison_tab(current_user: User, documents: List[Document], comparator: PaperComparator) -> None:
    """Render multi-paper comparative analysis across 10 academic dimensions."""
    st.markdown("### ⚖️ Multi-Paper Comparative Analysis")
    st.markdown(
        "Compare 2 or more research papers side-by-side across "
        "**10 structured dimensions** and an executive comparative synthesis narrative."
    )

    if len(documents) < 2:
        st.markdown(
            f"""
            <div class="welcome-card" style="border-left: 4px solid #f59e0b;">
                <h4 style="margin-top: 0; color: #b45309;">💡 Multi-Paper Comparison Requires at Least 2 Papers</h4>
                <p style="color: #475569; font-size: 0.95rem; margin-bottom: 0;">
                    You currently have <strong>{len(documents)} paper(s)</strong> in your library.
                    Please go to the <strong>Upload Papers</strong> tab and upload 2 or more research papers (PDFs)
                    to unlock cross-paper comparison matrices, architectural trade-offs, and synthesis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    doc_options = {d.document_id: f"{d.title} ({d.metadata.original_filename})" for d in documents}
    selected_ids = st.multiselect(
        "Select 2–5 Papers to Compare:",
        options=list(doc_options.keys()),
        default=list(doc_options.keys())[:min(3, len(documents))],
        format_func=lambda k: doc_options[k],
        max_selections=5,
    )

    if len(selected_ids) < 2:
        st.warning("Please select at least 2 papers from the list above to compare.")
        return

    if st.button("🚀 Compare Selected Papers", type="primary"):
        with st.spinner("Generating comparative analysis table and narrative synthesis..."):
            report: PaperComparisonReport = comparator.compare_papers(selected_ids, user_id=current_user.user_id)
            # Sync comparison to MongoDB Atlas cloud database
            get_storage().save_comparison(selected_ids, report.to_dataframe_dict(), user_id=current_user.user_id)

        st.markdown("#### 📊 Comparative Dimension Matrix")
        df_rows = report.to_dataframe_dict()
        st.dataframe(pd.DataFrame(df_rows), use_container_width=True)

        st.markdown("#### 📝 Comparative Synthesis Narrative")
        st.markdown(f'<div class="intelligence-card">{report.comparative_synthesis}</div>', unsafe_allow_html=True)

        if report.citations:
            st.markdown(
                f"""
                <div class="citation-box">
                    <h4>📚 Supporting Source Citations ({len(report.citations)} references)</h4>
                    <div style="font-size: 0.9rem; line-height: 1.7; color: #4b5563;">
                        {'<br/>'.join([f"<strong>{c.citation_id}</strong> <em>{html.escape(c.paper_title)}</em> — <code>{html.escape(c.filename)}</code> (Page <strong>{c.page_number}</strong>)" for c in report.citations])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =====================================================================
# TAB 7: RESEARCH GAP ANALYSIS
# =====================================================================
def render_gap_analysis_tab(current_user: User, documents: List[Document], gap_analyzer: ResearchGapAnalyzer) -> None:
    """Render evidence-grounded research gap analysis."""
    st.markdown("### 🔬 Evidence-Grounded Research Gap Discovery")
    st.markdown(
        "Automatically identifies potential research gaps, recurring limitations, and unexplored challenges. "
        "Every gap is tied directly to source evidence passages and explicit page citations."
    )

    if not documents:
        st.info("No papers available in library. Upload PDFs in the **Upload Papers** tab first.")
        return

    doc_options = {d.document_id: f"{d.title} ({d.metadata.original_filename})" for d in documents}
    selected_ids = st.multiselect(
        "Select Papers to Analyze for Research Gaps:",
        options=list(doc_options.keys()),
        default=list(doc_options.keys()),
        format_func=lambda k: doc_options[k],
    )

    if selected_ids:
        if st.button("🔍 Discover Research Gaps", type="primary"):
            with st.spinner("Scanning limitations, discussion sections, and synthesizing research gaps..."):
                report: ResearchGapReport = gap_analyzer.analyze_gaps(selected_ids, user_id=current_user.user_id)

            st.markdown("#### 📌 Executive Gap Synthesis")
            st.markdown(f'<div class="intelligence-card">{report.executive_summary}</div>', unsafe_allow_html=True)

            st.markdown(f"#### 🔍 Grounded Research Gaps ({len(report.gaps)} identified)")
            for idx, g in enumerate(report.gaps, start=1):
                with st.container():
                    st.markdown(f"##### Gap #{idx}: {g.gap_statement}")
                    st.markdown(f"**Category:** `{g.category}` | **Source:** `{g.source_paper}` (Page **{g.page_number}**, Section: *{g.section}*) {g.citation_id}")
                    st.markdown(f'<div class="snippet-box"><strong>Evidence from Paper:</strong> "{g.supporting_evidence}"</div>', unsafe_allow_html=True)
                    st.markdown(f"💡 **Proposed Research Pathway:** {g.future_direction_suggestion}")
                    st.markdown("<hr style='margin: 0.75rem 0;'/>", unsafe_allow_html=True)

            with st.expander("📋 Copy Markdown Gap Report", expanded=False):
                st.code(report.to_markdown(), language="markdown")


# =====================================================================
# TAB 8: LITERATURE REVIEW GENERATOR
# =====================================================================
def render_lit_review_tab(current_user: User, documents: List[Document], lit_reviewer: LiteratureReviewGenerator) -> None:
    """Render 10-section publication-ready literature review generator."""
    st.markdown("### 📖 Literature Review Generator")
    st.markdown(
        "Synthesize multiple research papers into a formal **10-section academic literature review** "
        "with complete citation grounding."
    )

    if not documents:
        st.info("No papers available in library. Upload PDFs in the **Upload Papers** tab first.")
        return

    doc_options = {d.document_id: f"{d.title} ({d.metadata.original_filename})" for d in documents}
    col_t1, col_t2 = st.columns([2, 1])

    with col_t1:
        topic_focus = st.text_input("Review Topic / Focus Area:", placeholder="e.g. Advancements and Challenges in Deep Learning Architectures")

    with col_t2:
        selected_ids = st.multiselect(
            "Include Papers:",
            options=list(doc_options.keys()),
            default=list(doc_options.keys()),
            format_func=lambda k: doc_options[k],
        )

    if selected_ids:
        if st.button("📜 Generate Literature Review", type="primary"):
            with st.spinner("Synthesizing 10-section literature review with citation grounding..."):
                report: LiteratureReviewReport = lit_reviewer.generate_review(
                    document_ids=selected_ids,
                    topic_focus=topic_focus or "Synthesized Literature Review",
                    user_id=current_user.user_id,
                )

            st.markdown(f"## 📖 {report.topic}")

            sections = [
                ("1. Introduction", report.introduction),
                ("2. Research Area Overview", report.research_area_overview),
                ("3. Existing Approaches", report.existing_approaches),
                ("4. Methodology Comparison", report.methodology_comparison),
                ("5. Dataset Trends", report.dataset_trends),
                ("6. Key Findings", report.key_findings),
                ("7. Limitations", report.limitations),
                ("8. Potential Research Gaps", report.potential_research_gaps),
                ("9. Future Research Directions", report.future_research_directions),
                ("10. References", report.references_markdown),
            ]

            for sec_title, sec_content in sections:
                st.markdown(f"### {sec_title}")
                st.markdown(sec_content)
                st.markdown("<hr style='margin: 0.5rem 0 1rem 0; opacity: 0.15;'/>", unsafe_allow_html=True)

            with st.expander("📋 Copy Complete Markdown Literature Review", expanded=False):
                st.code(report.to_markdown(), language="markdown")


# =====================================================================
# TAB 9: EVALUATION & BENCHMARKS
# =====================================================================
def render_evaluation_tab(dataset: EvaluationDataset, evaluator: RetrievalEvaluator, logger: ExperimentLogger) -> None:
    """Render evaluation dataset viewer, retrieval benchmarks, and human rating form."""
    st.markdown("### 🧪 Evaluation & IR Benchmarks Studio")
    st.markdown(
        "Evaluate retrieval precision, recall, hit rate, and MRR across the benchmark dataset. "
        "Record human feedback and inspect experimental run logs."
    )

    tab_bench, tab_data, tab_human, tab_history = st.tabs(
        ["📈 Retrieval Benchmarks", "📋 Benchmark Dataset", "⭐ Human Evaluation Rating", "🕒 Experiment History"]
    )

    with tab_bench:
        st.markdown("#### 🔬 Multi-Strategy Retrieval Benchmark")
        st.markdown(
            "Execute automated Information Retrieval (IR) evaluation across all dataset questions comparing:\n"
            "- **Dense Only (ChromaDB)**\n"
            "- **Sparse Only (BM25)**\n"
            "- **Hybrid RRF**\n"
            "- **Hybrid RRF + Cross-Encoder Reranking**"
        )

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            top_k_eval = st.slider("Top-K Evaluation Budget:", min_value=1, max_value=10, value=5)
        with col_b2:
            cand_k_eval = st.slider("Candidate Pool (K):", min_value=10, max_value=40, value=20)
        with col_b3:
            run_bench_btn = st.button("⚡ Run Full Retrieval Benchmark", type="primary", use_container_width=True)

        if run_bench_btn:
            with st.spinner("Running retrieval evaluation across benchmark dataset..."):
                report: RetrievalEvaluationReport = evaluator.run_full_benchmark(
                    top_k=top_k_eval,
                    candidate_k=cand_k_eval,
                )
                st.session_state.eval_report = report

        if st.session_state.eval_report:
            rep: RetrievalEvaluationReport = st.session_state.eval_report
            st.markdown("##### 📊 Comparative Summary Table")

            summary_rows = []
            for m_name, s in rep.method_summaries.items():
                summary_rows.append({
                    "Strategy": {
                        "dense": "Dense Only (ChromaDB)",
                        "sparse": "Sparse Only (BM25)",
                        "hybrid": "Hybrid RRF (Dense + BM25)",
                        "hybrid_rerank": "Hybrid RRF + Cross-Encoder",
                    }.get(m_name, m_name),
                    "Precision@K": f"{s.precision_at_k * 100:.1f}%",
                    "Recall@K": f"{s.recall_at_k * 100:.1f}%",
                    "Hit Rate@K": f"{s.hit_rate_at_k * 100:.1f}%",
                    "MRR": f"{s.mrr:.4f}",
                    "Latency (ms)": f"{s.average_latency_ms:.1f} ms",
                    "Queries": s.total_queries_evaluated,
                })

            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            with st.expander("🔍 Inspect Per-Query Evaluation Breakdown", expanded=False):
                q_rows = []
                for d in rep.detailed_query_results:
                    q_rows.append({
                        "ID": d.question_id,
                        "Category": d.category,
                        "Method": d.method_name,
                        "Precision": d.precision,
                        "Recall": d.recall,
                        "Hit Rate": d.hit_rate,
                        "MRR": d.reciprocal_rank,
                        "Latency (ms)": d.latency_ms,
                    })
                st.dataframe(pd.DataFrame(q_rows), use_container_width=True)

    with tab_data:
        st.markdown(f"#### 📋 Ground-Truth Benchmark Dataset ({len(dataset.questions)} Questions)")
        cat_filter = st.selectbox("Filter by Category:", options=["All", "factual", "methodology", "dataset", "results", "limitations", "comparison", "multi-paper"])

        q_list = dataset.questions if cat_filter == "All" else dataset.get_by_category(cat_filter)
        q_rows = []
        for q in q_list:
            q_rows.append({
                "ID": q.question_id,
                "Category": q.category,
                "Question": q.question,
                "Expected Ground Truth": q.expected_answer,
                "Target Papers": ", ".join(q.relevant_documents),
                "Target Pages": ", ".join(map(str, q.relevant_pages)),
            })
        st.dataframe(pd.DataFrame(q_rows), use_container_width=True)

    with tab_human:
        st.markdown("#### ⭐ Human Expert Evaluation Rating (1 to 5 Scale)")
        st.markdown("Submit rigorous human evaluation feedback for research Q&A answers.")

        with st.form("human_eval_form"):
            eval_query = st.text_input("Evaluated Query / Prompt:", value="What are the key contributions and architecture details?")
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                r_corr = st.slider("1. Factual Correctness (1-5):", 1, 5, 5)
                r_rel = st.slider("2. Question Relevance (1-5):", 1, 5, 5)
            with col_r2:
                r_comp = st.slider("3. Completeness (1-5):", 1, 5, 4)
                r_cit = st.slider("4. Citation Quality (1-5):", 1, 5, 5)
            with col_r3:
                r_gnd = st.slider("5. Groundedness (1-5):", 1, 5, 5)
                eval_id = st.text_input("Reviewer ID:", value="researcher_1")

            notes = st.text_area("Reviewer Notes & Qualitative Feedback:", placeholder="Citations were accurate and pointed to exact pages.")
            submit_rating = st.form_submit_button("Submit Rating Record", type="primary")

            if submit_rating:
                rating = HumanEvaluationRating(
                    evaluator_id=eval_id,
                    query=eval_query,
                    correctness=r_corr,
                    relevance=r_rel,
                    completeness=r_comp,
                    citation_quality=r_cit,
                    groundedness=r_gnd,
                    feedback_notes=notes,
                )
                logger.log_human_rating(rating)
                st.success(f"✅ Logged evaluation rating! Composite score: **{rating.average_score} / 5.0**")

    with tab_history:
        st.markdown("#### 🕒 Logged Experiments & Human Ratings")
        exp_records = logger.load_recent_experiments()
        if exp_records:
            st.write(f"Showing last **{len(exp_records)}** logged experiment runs:")
            st.dataframe(pd.DataFrame(exp_records), use_container_width=True)
        else:
            st.caption("No experiment records found.")

        st.markdown("---")
        human_ratings = logger.load_human_ratings()
        if human_ratings:
            st.write(f"Showing **{len(human_ratings)}** human rating records:")
            st.dataframe(pd.DataFrame(human_ratings), use_container_width=True)


# =====================================================================
# TAB 10: SETTINGS & SYSTEM DIAGNOSTICS
# =====================================================================
def render_settings_tab(current_user: User, retriever: HybridRetriever, rag_pipeline: RAGPipeline) -> None:
    """Render configuration and system diagnostics tab."""
    st.markdown("### ⚙️ Settings & Workspace Diagnostics")

    llm = rag_pipeline.llm_provider
    stats = retriever.get_stats(user_id=current_user.user_id)
    reranker_stats = retriever.reranker.get_stats()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👤 User Account Details")
        st.write(f"**Name:** `{current_user.full_name}`")
        st.write(f"**Email:** `{current_user.email}`")
        st.write(f"**User ID:** `{current_user.user_id}`")
        st.write(f"**Member Since:** `{current_user.created_at[:10]}`")

    with col2:
        c_storage = get_cloudinary()
        st.markdown("#### ☁️ Cloud & AI Infrastructure")
        st.write(f"**AI Provider:** `{llm.get_provider_name().upper()}` ({llm.get_model_name()})")
        st.write(f"**Dense Embeddings:** `all-MiniLM-L6-v2` (384-d, local)")
        st.write(f"**Reranker Model:** `{reranker_stats['model_name']}`")
        
        c_status = "🟢 Connected (Cloudinary Active)" if c_storage.is_configured else "⚡ Local Fallback Storage"
        st.write(f"**PDF Cloud Storage:** {c_status}")
        if c_storage.is_configured:
            st.write(f"**Cloud Name:** `{c_storage.cloud_name}`")
            st.write(f"**API Secret:** `[REDACTED / ENCRYPTED]`")

    st.markdown("---")
    st.markdown("#### 🗑️ Danger Zone: Reset My Library")
    st.caption("Clears only your uploaded research papers, vector embeddings in ChromaDB, and sparse BM25 indexes.")
    if st.button("⚠️ Reset My Private Library Data", type="secondary"):
        clear_all_documents(user_id=current_user.user_id)
        retriever.clear_all(user_id=current_user.user_id)
        st.session_state.selected_doc_id = None
        st.session_state.chat_history = []
        st.success("Your private library and vector stores have been reset.")
        st.rerun()


# =====================================================================
# MAIN APPLICATION ROUTING
# =====================================================================
def main() -> None:
    """Main application execution router."""
    auth_service = get_auth()
    init_app_state(auth_service=auth_service)
    render_session_sync_bridge(st.session_state.get("session_token"))
    storage = get_storage()

    # If not logged in, show Auth Portal (Sign In / Sign Up)
    if st.session_state.authenticated_user is None:
        render_auth_portal(auth_service)
        return

    current_user: User = st.session_state.authenticated_user
    retriever = get_retriever()
    rag_pipeline = get_rag_pipeline()
    summarizer, comparator, gap_analyzer, lit_reviewer = get_intelligence_suite()
    eval_dataset, retrieval_evaluator, exp_logger = get_evaluation_suite()

    # Load documents specific to the logged-in user from MongoDB Atlas / local cache
    documents = storage.load_user_documents(user_id=current_user.user_id)

    # Restore persisted chat history if session state is empty
    if not st.session_state.chat_history and storage.is_connected:
        persisted_turns = storage.load_user_chat_history(user_id=current_user.user_id)
        if persisted_turns:
            st.session_state.chat_history = [
                {
                    "role": t["role"],
                    "content": t["content"],
                }
                for t in persisted_turns
            ]

    # Sidebar & Header
    render_sidebar(current_user, documents, retriever, rag_pipeline)
    render_header(current_user)

    # 10 Main Navigation Tabs
    (
        tab_dash,
        tab_upload,
        tab_library,
        tab_qa,
        tab_summary,
        tab_compare,
        tab_gap,
        tab_lit,
        tab_eval,
        tab_settings,
    ) = st.tabs(
        [
            "📊 Dashboard",
            "📤 Upload Papers",
            "📚 Paper Library",
            "💬 Research Q&A",
            "📝 Paper Summary",
            "⚖️ Compare Papers",
            "🔬 Research Gap Analysis",
            "📖 Literature Review",
            "🧪 Evaluation",
            "⚙️ Settings",
        ]
    )

    with tab_dash:
        render_dashboard_tab(current_user, documents, retriever, exp_logger)

    with tab_upload:
        render_upload_tab(current_user, retriever)

    with tab_library:
        render_library_tab(current_user, documents, retriever)

    with tab_qa:
        render_qa_tab(current_user, documents, retriever, rag_pipeline, exp_logger)

    with tab_summary:
        render_summarization_tab(current_user, documents, summarizer)

    with tab_compare:
        render_comparison_tab(current_user, documents, comparator)

    with tab_gap:
        render_gap_analysis_tab(current_user, documents, gap_analyzer)

    with tab_lit:
        render_lit_review_tab(current_user, documents, lit_reviewer)

    with tab_eval:
        render_evaluation_tab(eval_dataset, retrieval_evaluator, exp_logger)

    with tab_settings:
        render_settings_tab(current_user, retriever, rag_pipeline)


if __name__ == "__main__":
    main()
