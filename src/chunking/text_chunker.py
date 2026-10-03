"""Text chunking and section-aware splitting module for ResearchMate.

Decomposes research paper pages into semantically cohesive, search-ready chunks
with metadata preservation (document ID, filename, page numbers, section headers, chunk index).
Avoids splitting sentences mid-stream and handles academic section detection.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.models.chunk import Chunk, ChunkMetadata
from src.models.document import Document, Page

# Common academic section header patterns
ACADEMIC_SECTION_PATTERNS = [
    # Numbered sections e.g. "1. Introduction", "1.1 Background", "IV. Methodology"
    r"^(?:(?:[0-9]+(?:\.[0-9]+)*|[IVXLCDM]+)\.?\s+)?(?:Abstract|Introduction|Related\s+Work|Background|Literature\s+Review|Proposed\s+Method(?:ology)?|Method(?:ology)?|Materials?\s+and\s+Methods?|Architecture|Implementation|Experiments?|Experimental\s+(?:Setup|Results|Evaluation)|Results?(?:\s+and\s+Discussion)?|Evaluation|Discussion|Conclusions?(?:\s+and\s+Future\s+Work)?|Future\s+Work|References|Bibliography|Acknowledgments?|Appendix(?:\s+[A-Z0-9]+)?)\b",
    # All uppercase section titles e.g. "ABSTRACT", "INTRODUCTION", "METHODOLOGY"
    r"^(?:[0-9]+\s+)?(?:ABSTRACT|INTRODUCTION|RELATED\s+WORK|BACKGROUND|METHODOLOGY|METHODS|EXPERIMENTS|RESULTS|DISCUSSION|CONCLUSION|CONCLUSIONS|REFERENCES|APPENDIX)\b",
]

COMPILED_SECTION_REGEX = [re.compile(p, re.IGNORECASE) for p in ACADEMIC_SECTION_PATTERNS]


def estimate_token_count(text: str) -> int:
    """Estimate token count for a text string using tiktoken or character ratio fallback."""
    if not text:
        return 0
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text, disallowed_special=()))
    except Exception:
        # Standard rule of thumb: ~4 characters per token in English
        return max(1, len(text) // 4)


def is_section_header(line: str) -> Optional[str]:
    """Determine if a line is an academic section header.

    Returns the cleaned section title if detected, otherwise None.
    """
    clean_line = line.strip()
    if not clean_line or len(clean_line) > 100 or len(clean_line) < 3:
        return None

    for regex in COMPILED_SECTION_REGEX:
        match = regex.match(clean_line)
        if match:
            # Clean up trailing colons or excess whitespace
            return clean_line.rstrip(":").strip()

    return None


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while avoiding splits at common abbreviations.

    Protects abbreviations like: e.g., i.e., et al., Fig., Eq., Dr., Prof., etc.
    """
    if not text:
        return []

    # Protect abbreviations by temporarily replacing their periods
    protected = text
    abbreviations = [
        "e.g.", "i.e.", "et al.", "Fig.", "Figs.", "Eq.", "Eqs.",
        "Ref.", "Refs.", "Sec.", "Secs.", "Tab.", "Tabs.", "Dr.", "Prof.",
        "vs.", "approx.", "dept.", "no.", "vol.", "pp.", "p."
    ]
    substitutions = {}
    for idx, abbr in enumerate(abbreviations):
        placeholder = f"__ABBR_{idx}__"
        if abbr.lower() in protected.lower():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            substitutions[placeholder] = abbr
            protected = pattern.sub(placeholder, protected)

    # Split on sentence boundaries: punctuation followed by whitespace and capital letter / newline
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\(\"\'\n])", protected)

    sentences = []
    for s in raw_sentences:
        # Restore protected abbreviations
        for placeholder, original in substitutions.items():
            s = s.replace(placeholder, original)
        cleaned = s.strip()
        if cleaned:
            sentences.append(cleaned)

    return sentences if sentences else [text.strip()]


class TextChunker:
    """Chunking engine for splitting research documents into structured, searchable chunks."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        min_chunk_size: int = 100,
        strategy: str = "section_aware",
    ) -> None:
        """Initialize the text chunker.

        Args:
            chunk_size: Target maximum chunk size in characters (default: ~800 chars / ~180 tokens).
            chunk_overlap: Overlap size in characters between consecutive chunks (default: 150 chars).
            min_chunk_size: Minimum character length for a standalone chunk.
            strategy: Chunking strategy identifier ("section_aware", "recursive", "fixed").
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.strategy = strategy

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Process a structured Document instance into an ordered list of search chunks.

        Preserves:
        - document_id & filename
        - 1-indexed page number(s)
        - Academic section tagging across pages
        - Sequential chunk_index and unique chunk_id
        """
        if not document.pages:
            return []

        chunks: List[Chunk] = []
        current_section = "Abstract"  # Default initial section
        chunk_index = 0

        for page in document.pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            # Process page content
            page_chunks, current_section, chunk_index = self._chunk_page(
                page=page,
                document=document,
                current_section=current_section,
                start_chunk_index=chunk_index,
            )
            chunks.extend(page_chunks)

        return chunks

    def _chunk_page(
        self,
        page: Page,
        document: Document,
        current_section: str,
        start_chunk_index: int,
    ) -> Tuple[List[Chunk], str, int]:
        """Chunk a single page, tracking and updating section headers."""
        chunks: List[Chunk] = []
        chunk_index = start_chunk_index

        # Split page into paragraphs
        paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]

        current_buffer: List[str] = []
        current_buffer_len = 0

        for paragraph in paragraphs:
            # Check if this paragraph starts with or is a section header
            first_line = paragraph.splitlines()[0].strip()
            detected_sec = is_section_header(first_line)

            if detected_sec:
                # If we have buffered text and section is changing, flush existing buffer
                if current_buffer and current_buffer_len >= self.min_chunk_size:
                    chunk_text = " ".join(current_buffer).strip()
                    chunk = self._create_chunk(
                        text=chunk_text,
                        document=document,
                        page_number=page.page_number,
                        section=current_section,
                        chunk_index=chunk_index,
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_buffer = []
                    current_buffer_len = 0

                current_section = detected_sec
                # If paragraph had more than just the header, keep remaining body text
                body_lines = paragraph.splitlines()[1:]
                paragraph = "\n".join(body_lines).strip()
                if not paragraph:
                    continue

            # Split paragraph into sentences
            sentences = split_into_sentences(paragraph)

            for sentence in sentences:
                sent_len = len(sentence)

                # If adding this sentence exceeds target chunk_size and buffer is non-empty
                if current_buffer_len + sent_len + 1 > self.chunk_size and current_buffer:
                    chunk_text = " ".join(current_buffer).strip()
                    if len(chunk_text) >= self.min_chunk_size:
                        chunk = self._create_chunk(
                            text=chunk_text,
                            document=document,
                            page_number=page.page_number,
                            section=current_section,
                            chunk_index=chunk_index,
                        )
                        chunks.append(chunk)
                        chunk_index += 1

                    # Compute overlap buffer from trailing sentences
                    overlap_buffer = []
                    overlap_len = 0
                    for s in reversed(current_buffer):
                        if overlap_len + len(s) + 1 <= self.chunk_overlap:
                            overlap_buffer.insert(0, s)
                            overlap_len += len(s) + 1
                        else:
                            break

                    current_buffer = overlap_buffer
                    current_buffer_len = sum(len(s) for s in current_buffer) + max(0, len(current_buffer) - 1)

                # If a single sentence is larger than chunk_size, hard-split by word boundaries
                if sent_len > self.chunk_size:
                    sub_words = sentence.split()
                    temp_sub: List[str] = []
                    temp_len = 0
                    for w in sub_words:
                        if temp_len + len(w) + 1 > self.chunk_size and temp_sub:
                            sub_text = " ".join(temp_sub)
                            chunk = self._create_chunk(
                                text=sub_text,
                                document=document,
                                page_number=page.page_number,
                                section=current_section,
                                chunk_index=chunk_index,
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                            temp_sub = []
                            temp_len = 0
                        temp_sub.append(w)
                        temp_len += len(w) + 1
                    if temp_sub:
                        current_buffer.extend(temp_sub)
                        current_buffer_len += temp_len
                else:
                    current_buffer.append(sentence)
                    current_buffer_len += sent_len + 1

        # Flush any remaining buffer on this page
        if current_buffer:
            chunk_text = " ".join(current_buffer).strip()
            if len(chunk_text) >= self.min_chunk_size or not chunks:
                chunk = self._create_chunk(
                    text=chunk_text,
                    document=document,
                    page_number=page.page_number,
                    section=current_section,
                    chunk_index=chunk_index,
                )
                chunks.append(chunk)
                chunk_index += 1

        return chunks, current_section, chunk_index

    def _create_chunk(
        self,
        text: str,
        document: Document,
        page_number: int,
        section: str,
        chunk_index: int,
    ) -> Chunk:
        """Helper to instantiate a valid Chunk instance."""
        doc_id = document.document_id
        chunk_id = f"chk_{doc_id}_{chunk_index:04d}"
        token_count = estimate_token_count(text)

        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            document_id=doc_id,
            filename=document.metadata.original_filename or document.metadata.filename,
            page_number=page_number,
            page_numbers=[page_number],
            section=section,
            chunk_index=chunk_index,
            char_count=len(text),
            token_count=token_count,
            strategy=self.strategy,
        )

        return Chunk(text=text, metadata=metadata)


def chunk_document(
    document: Document,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    strategy: str = "section_aware",
) -> List[Chunk]:
    """Convenience function to chunk a Document using the default TextChunker."""
    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy,
    )
    return chunker.chunk_document(document)
