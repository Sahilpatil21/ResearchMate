"""BM25 sparse keyword retrieval engine for ResearchMate.

Provides fast, exact-match and keyword-based retrieval over research paper chunks
using the BM25Okapi ranking function with academic term tokenization and multi-tenant user isolation.
"""

from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from rank_bm25 import BM25Okapi, BM25Plus

from src.models.chunk import Chunk
from src.models.retrieval import RetrievalResult
from src.utils.file_utils import get_chunks_dir, load_all_document_chunks, load_document_chunks

# Academic / English stop words to filter out for cleaner sparse matching
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
}


def tokenize_academic_text(text: str, remove_stopwords: bool = False) -> List[str]:
    """Tokenize academic text into normalized terms."""
    if not text:
        return []

    cleaned = text.lower()
    raw_tokens = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b", cleaned)
    
    tokens: List[str] = []
    for token in raw_tokens:
        if remove_stopwords and token in STOP_WORDS:
            continue
        tokens.append(token)
        if "-" in token:
            sub_parts = [p for p in token.split("-") if p and (not remove_stopwords or p not in STOP_WORDS)]
            tokens.extend(sub_parts)

    return tokens


class BM25Retriever:
    """In-memory BM25 sparse keyword retriever with multi-tenant user scoping."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.25,
        remove_stopwords: bool = False,
    ) -> None:
        """Initialize BM25Retriever."""
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.remove_stopwords = remove_stopwords
        
        self.chunks: List[Chunk] = []
        self.chunk_ids: Set[str] = set()
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[Union[BM25Plus, BM25Okapi]] = None
        self._doc_id_to_chunks: Dict[str, List[Chunk]] = {}
        self._chunk_to_user: Dict[str, str] = {}

    def _rebuild_index(self) -> None:
        """Rebuild the underlying BM25Plus model from current chunks."""
        if not self.chunks:
            self.bm25 = None
            self.tokenized_corpus = []
            return

        self.tokenized_corpus = [
            tokenize_academic_text(c.text, remove_stopwords=self.remove_stopwords)
            for c in self.chunks
        ]
        try:
            self.bm25 = BM25Plus(
                self.tokenized_corpus,
                k1=self.k1,
                b=self.b,
                delta=self.delta,
            )
        except Exception:
            self.bm25 = BM25Okapi(
                self.tokenized_corpus,
                k1=self.k1,
                b=self.b,
            )

    def add_chunks(
        self,
        chunks: List[Chunk],
        overwrite: bool = False,
        user_id: Optional[str] = None,
    ) -> int:
        """Add a list of chunks to the BM25 index."""
        if not chunks:
            return 0

        target_doc_ids = {c.document_id for c in chunks}
        
        if overwrite:
            for doc_id in target_doc_ids:
                self.delete_document(doc_id)

        added_count = 0
        for chunk in chunks:
            if chunk.chunk_id not in self.chunk_ids:
                self.chunks.append(chunk)
                self.chunk_ids.add(chunk.chunk_id)
                if user_id:
                    self._chunk_to_user[chunk.chunk_id] = user_id
                if chunk.document_id not in self._doc_id_to_chunks:
                    self._doc_id_to_chunks[chunk.document_id] = []
                self._doc_id_to_chunks[chunk.document_id].append(chunk)
                added_count += 1

        if added_count > 0:
            self._rebuild_index()

        return added_count

    def index_document(
        self,
        document_id: str,
        chunks: List[Chunk],
        overwrite: bool = True,
        user_id: Optional[str] = None,
    ) -> int:
        """Index chunks for a specific document."""
        if overwrite and self.is_document_indexed(document_id):
            self.delete_document(document_id)

        return self.add_chunks(chunks, overwrite=False, user_id=user_id)

    def delete_document(self, document_id: str) -> bool:
        """Remove all chunks belonging to a document from the index."""
        if document_id not in self._doc_id_to_chunks:
            return False

        # Filter out chunks for this document
        self.chunks = [c for c in self.chunks if c.document_id != document_id]
        self.chunk_ids = {c.chunk_id for c in self.chunks}
        for chk in self._doc_id_to_chunks.get(document_id, []):
            self._chunk_to_user.pop(chk.chunk_id, None)
        del self._doc_id_to_chunks[document_id]

        self._rebuild_index()
        return True

    def clear(self, user_id: Optional[str] = None) -> None:
        """Reset the BM25 index completely or for a specific user."""
        if user_id:
            user_chunk_ids = {cid for cid, uid in self._chunk_to_user.items() if uid == user_id}
            self.chunks = [c for c in self.chunks if c.chunk_id not in user_chunk_ids]
            self.chunk_ids = {c.chunk_id for c in self.chunks}
            for cid in user_chunk_ids:
                self._chunk_to_user.pop(cid, None)
            self._rebuild_index()
        else:
            self.chunks = []
            self.chunk_ids = set()
            self.tokenized_corpus = []
            self.bm25 = None
            self._doc_id_to_chunks = {}
            self._chunk_to_user = {}

    def is_document_indexed(self, document_id: str) -> bool:
        """Check whether a document is present in the BM25 index."""
        return document_id in self._doc_id_to_chunks and len(self._doc_id_to_chunks[document_id]) > 0

    def sync_from_storage(self, chunks_dir: Optional[Path] = None, user_id: Optional[str] = None) -> int:
        """Load and index all chunk files from chunks directory."""
        all_chunks = load_all_document_chunks(dest_dir=chunks_dir, user_id=user_id)
        if all_chunks:
            self.add_chunks(all_chunks, overwrite=True, user_id=user_id)
        return len(self.chunks)

    def search(
        self,
        query: str,
        n_results: int = 10,
        filter_doc_id: Optional[str] = None,
        min_score: float = 0.0,
        user_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Execute BM25 keyword search."""
        if not query or not query.strip() or not self.chunks or self.bm25 is None:
            return []

        tokenized_query = tokenize_academic_text(query, remove_stopwords=self.remove_stopwords)
        if not tokenized_query:
            return []

        raw_scores = self.bm25.get_scores(tokenized_query)
        
        candidates: List[Tuple[float, int, Chunk]] = []
        for idx, (score, chunk) in enumerate(zip(raw_scores, self.chunks)):
            if user_id and self._chunk_to_user.get(chunk.chunk_id, user_id) != user_id:
                continue
            if filter_doc_id and chunk.document_id != filter_doc_id:
                continue
            if score > min_score and score > 0.0:
                candidates.append((float(score), idx, chunk))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[:n_results]

        max_score = top_candidates[0][0] if top_candidates else 1.0

        results: List[RetrievalResult] = []
        for rank_idx, (score, _, chunk) in enumerate(top_candidates, start=1):
            normalized_score = round(score / max_score, 4) if max_score > 0 else 0.0
            
            result = RetrievalResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                section=chunk.section or "General",
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                sparse_score=round(float(score), 4),
                sparse_rank=rank_idx,
                score=normalized_score,
                retrieval_method="bm25",
            )
            results.append(result)

        return results


    def get_stats(self) -> Dict[str, Any]:
        """Return status and count statistics for the BM25 index."""
        return {
            "total_indexed_chunks": len(self.chunks),
            "indexed_documents_count": len(self._doc_id_to_chunks),
            "indexed_document_ids": sorted(list(self._doc_id_to_chunks.keys())),
            "status": "BM25 Index Active",
        }

