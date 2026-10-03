"""File handling and storage utilities for ResearchMate.

Provides safe storage, unique filename generation, hash calculation,
and JSON persistence for raw PDFs, processed document structures, text chunks,
and vector database directories with multi-tenant user workspace support.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

from src.models.chunk import Chunk
from src.models.document import Document

# Load environment variables from .env if present
load_dotenv()


def get_project_root() -> Path:
    """Return the absolute Path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_data_dir() -> Path:
    """Return the Path to the root data directory."""
    env_path = os.getenv("DATA_DIR", "data")
    path = Path(env_path)
    if not path.is_absolute():
        path = get_project_root() / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_user_data_dir(user_id: Optional[str] = None) -> Path:
    """Return the Path to a specific user's data directory."""
    if not user_id:
        return get_data_dir()
    clean_user = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)
    path = get_data_dir() / "users" / clean_user
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logs_dir(user_id: Optional[str] = None) -> Path:
    """Return the Path to the query logs directory."""
    if user_id:
        path = get_user_data_dir(user_id) / "logs"
    else:
        path = get_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_evaluation_dir() -> Path:
    """Return the Path to the evaluation datasets and experiment logs directory."""
    path = get_data_dir() / "evaluation"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_papers_dir(user_id: Optional[str] = None) -> Path:
    """Return the Path to the raw papers storage directory."""
    if user_id:
        path = get_user_data_dir(user_id) / "papers"
    else:
        env_path = os.getenv("DATA_PAPERS_DIR", "data/papers")
        path = Path(env_path)
        if not path.is_absolute():
            path = get_project_root() / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_processed_dir(user_id: Optional[str] = None) -> Path:
    """Return the Path to the processed JSON documents directory."""
    if user_id:
        path = get_user_data_dir(user_id) / "processed"
    else:
        env_path = os.getenv("DATA_PROCESSED_DIR", "data/processed")
        path = Path(env_path)
        if not path.is_absolute():
            path = get_project_root() / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_chunks_dir(user_id: Optional[str] = None) -> Path:
    """Return the Path to the persisted text chunks JSON directory."""
    if user_id:
        path = get_user_data_dir(user_id) / "chunks"
    else:
        env_path = os.getenv("DATA_CHUNKS_DIR", "data/chunks")
        path = Path(env_path)
        if not path.is_absolute():
            path = get_project_root() / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_chroma_dir() -> Path:
    """Return the Path to the ChromaDB persistent vector storage directory."""
    env_path = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
    path = Path(env_path)
    if not path.is_absolute():
        path = get_project_root() / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_directories(user_id: Optional[str] = None) -> Tuple[Path, Path, Path, Path]:
    """Ensure that all necessary data directories exist."""
    papers_dir = get_papers_dir(user_id)
    processed_dir = get_processed_dir(user_id)
    chunks_dir = get_chunks_dir(user_id)
    chroma_dir = get_chroma_dir()
    logs_dir = get_logs_dir(user_id)
    eval_dir = get_evaluation_dir()
    return papers_dir, processed_dir, chunks_dir, chroma_dir


def calculate_sha256(file_bytes: bytes) -> str:
    """Calculate the SHA-256 hexadecimal digest for raw file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing or replacing unsafe characters."""
    name = Path(filename).name
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "document.pdf"


def get_unique_filepath(dest_dir: Path, filename: str) -> Path:
    """Generate a unique file path within dest_dir without overwriting existing files."""
    sanitized = sanitize_filename(filename)
    target_path = dest_dir / sanitized

    if not target_path.exists():
        return target_path

    stem = Path(sanitized).stem
    suffix = Path(sanitized).suffix

    counter = 1
    while True:
        candidate = dest_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def save_uploaded_pdf(
    file_bytes: bytes,
    original_filename: str,
    dest_dir: Optional[Path] = None,
    user_id: Optional[str] = None,
) -> Tuple[Path, str]:
    """Save an uploaded PDF buffer to the papers storage directory."""
    target_dir = dest_dir or get_papers_dir(user_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    unique_path = get_unique_filepath(target_dir, original_filename)
    unique_path.write_bytes(file_bytes)

    return unique_path, unique_path.name


def save_processed_document(
    document: Document,
    dest_dir: Optional[Path] = None,
    user_id: Optional[str] = None,
) -> Path:
    """Persist a processed Document instance as a JSON file."""
    target_dir = dest_dir or get_processed_dir(user_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / f"{document.document_id}.json"
    json_path.write_text(document.to_json(indent=2), encoding="utf-8")

    return json_path


def load_processed_document(
    document_id: str,
    dest_dir: Optional[Path] = None,
    user_id: Optional[str] = None,
) -> Optional[Document]:
    """Load a processed Document instance by document_id."""
    target_dir = dest_dir or get_processed_dir(user_id)
    json_path = target_dir / f"{document_id}.json"

    if not json_path.exists():
        # Check global processed dir as fallback
        fallback_path = get_processed_dir(None) / f"{document_id}.json"
        if fallback_path.exists():
            json_path = fallback_path
        else:
            return None

    try:
        content = json_path.read_text(encoding="utf-8")
        return Document.from_json(content)
    except Exception:
        return None


def load_all_processed_documents(
    dest_dir: Optional[Path] = None,
    user_id: Optional[str] = None,
) -> List[Document]:
    """Load all processed documents from the user or global storage directory."""
    target_dir = dest_dir or get_processed_dir(user_id)
    if not target_dir.exists():
        return []

    documents: List[Document] = []
    for json_file in target_dir.glob("*.json"):
        try:
            content = json_file.read_text(encoding="utf-8")
            doc = Document.from_json(content)
            documents.append(doc)
        except Exception:
            continue

    documents.sort(key=lambda d: d.metadata.upload_time, reverse=True)
    return documents


def save_document_chunks(
    document_id: str,
    chunks: List[Chunk],
    dest_dir: Optional[Path] = None,
    user_id: Optional[str] = None,
) -> Path:
    """Save chunk list to JSON file under chunks directory."""
    target_dir = dest_dir or get_chunks_dir(user_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / f"{document_id}.json"
    chunk_dicts = [c.to_dict() for c in chunks]
    json_path.write_text(json.dumps(chunk_dicts, indent=2, ensure_ascii=False), encoding="utf-8")
    return json_path


def load_document_chunks(
    document_id: str,
    dest_dir: Optional[Path] = None,
    user_id: Optional[str] = None,
) -> Optional[List[Chunk]]:
    """Load chunks for a given document_id from disk."""
    target_dir = dest_dir or get_chunks_dir(user_id)
    json_path = target_dir / f"{document_id}.json"

    if not json_path.exists():
        # Fallback to global chunks
        fallback_path = get_chunks_dir(None) / f"{document_id}.json"
        if fallback_path.exists():
            json_path = fallback_path
        else:
            return None

    try:
        content = json_path.read_text(encoding="utf-8")
        data = json.loads(content)
        return [Chunk.from_dict(item) for item in data]
    except Exception:
        return None


def load_all_document_chunks(
    dest_dir: Optional[Path] = None,
    user_id: Optional[str] = None,
) -> List[Chunk]:
    """Load all chunk instances across all documents from the chunks directory."""
    target_dir = dest_dir or get_chunks_dir(user_id)
    if not target_dir.exists():
        return []

    all_chunks: List[Chunk] = []
    for json_file in target_dir.glob("*.json"):
        try:
            content = json_file.read_text(encoding="utf-8")
            data = json.loads(content)
            for item in data:
                all_chunks.append(Chunk.from_dict(item))
        except Exception:
            continue

    return all_chunks


def delete_document_chunks(
    document_id: str,
    dest_dir: Optional[Path] = None,
    user_id: Optional[str] = None,
) -> bool:
    """Delete chunks JSON file for a given document_id."""
    target_dir = dest_dir or get_chunks_dir(user_id)
    json_path = target_dir / f"{document_id}.json"
    if json_path.exists():
        json_path.unlink()
        return True
    return False


def delete_document(
    document_id: str,
    papers_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    chunks_dir: Optional[Path] = None,
    user_id: Optional[str] = None,
) -> bool:
    """Delete a document, its raw PDF, processed JSON metadata, and chunks JSON."""
    p_dir = processed_dir or get_processed_dir(user_id)
    raw_dir = papers_dir or get_papers_dir(user_id)

    deleted = False
    json_path = p_dir / f"{document_id}.json"

    if json_path.exists():
        try:
            doc = Document.from_json(json_path.read_text(encoding="utf-8"))
            raw_pdf_path = raw_dir / doc.metadata.filename
            if raw_pdf_path.exists():
                raw_pdf_path.unlink()
                deleted = True
        except Exception:
            pass

        json_path.unlink()
        deleted = True

    c_deleted = delete_document_chunks(document_id, dest_dir=chunks_dir, user_id=user_id)
    if c_deleted:
        deleted = True

    return deleted


def clear_all_documents(
    papers_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    chunks_dir: Optional[Path] = None,
    user_id: Optional[str] = None,
) -> int:
    """Delete all stored raw PDFs, processed JSONs, and cached chunk files."""
    p_dir = processed_dir or get_processed_dir(user_id)
    raw_dir = papers_dir or get_papers_dir(user_id)
    c_dir = chunks_dir or get_chunks_dir(user_id)
    count = 0
    if raw_dir.exists():
        for p in raw_dir.glob("*.pdf"):
            try:
                p.unlink()
                count += 1
            except Exception:
                pass
    if p_dir.exists():
        for p in p_dir.glob("*.json"):
            try:
                p.unlink()
                count += 1
            except Exception:
                pass
        sum_dir = p_dir / "summaries"
        if sum_dir.exists():
            for p in sum_dir.glob("*.json"):
                try:
                    p.unlink()
                except Exception:
                    pass
    if c_dir.exists():
        for p in c_dir.glob("*.json"):
            try:
                p.unlink()
                count += 1
            except Exception:
                pass
    return count
