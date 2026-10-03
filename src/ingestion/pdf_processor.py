"""PDF extraction and metadata detection processor for ResearchMate.

Uses PyMuPDF (fitz) to extract text per page, detect title and authors,
preserve 1-indexed page numbering, filter blank pages, and produce structured Document objects.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Union

import fitz  # PyMuPDF

from src.models.document import Document, DocumentMetadata, Page
from src.utils.file_utils import calculate_sha256, sanitize_filename


class PDFProcessingError(Exception):
    """Custom exception raised when PDF extraction or parsing fails."""
    pass


class PDFProcessor:
    """Processes PDF research papers into structured Document instances."""

    def __init__(self, ignore_empty_pages: bool = True) -> None:
        """Initialize the PDF Processor.

        Args:
            ignore_empty_pages: If True, completely blank pages are skipped from
                the extracted pages list while recording the total physical page count.
        """
        self.ignore_empty_pages = ignore_empty_pages

    def process_file(
        self,
        file_input: Union[str, Path, bytes],
        filename: Optional[str] = None,
        storage_filename: Optional[str] = None,
    ) -> Document:
        """Process a PDF file from a path or raw bytes.

        Args:
            file_input: Path to PDF file or raw bytes.
            filename: Original uploaded filename (used as fallback for metadata).
            storage_filename: Name used when saved on disk.

        Returns:
            A structured Document instance.

        Raises:
            PDFProcessingError: If the PDF cannot be opened, is corrupted, or has no pages.
        """
        raw_bytes: bytes
        if isinstance(file_input, (str, Path)):
            path = Path(file_input)
            if not path.exists():
                raise PDFProcessingError(f"PDF file not found at path: {path}")
            try:
                raw_bytes = path.read_bytes()
            except Exception as e:
                raise PDFProcessingError(f"Failed to read file: {e}") from e
            orig_name = filename or path.name
        elif isinstance(file_input, bytes):
            raw_bytes = file_input
            orig_name = filename or "uploaded_paper.pdf"
        else:
            raise PDFProcessingError(f"Unsupported file_input type: {type(file_input)}")

        if not raw_bytes:
            raise PDFProcessingError("Cannot process an empty (0-byte) file.")

        stored_name = storage_filename or sanitize_filename(orig_name)
        sha256_hash = calculate_sha256(raw_bytes)
        file_size = len(raw_bytes)

        # Open PDF with PyMuPDF
        try:
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
        except fitz.FileDataError as e:
            raise PDFProcessingError(f"Corrupted or invalid PDF format: {e}") from e
        except Exception as e:
            raise PDFProcessingError(f"Failed to parse PDF document: {e}") from e

        try:
            if doc.is_encrypted:
                if doc.needs_pass:
                    raise PDFProcessingError("PDF is password protected and cannot be read.")

            total_pages = doc.page_count
            if total_pages == 0:
                raise PDFProcessingError("PDF document contains 0 pages.")

            # Extract metadata heuristics
            title, authors = self._extract_metadata(doc, fallback_name=orig_name)

            # Extract pages
            pages = self._extract_pages(doc)

            has_extractable_text = len(pages) > 0 and any(p.word_count > 0 for p in pages)

            # Generate unique document ID
            doc_id = f"doc_{uuid.uuid4().hex[:12]}"

            doc_metadata = DocumentMetadata(
                document_id=doc_id,
                filename=stored_name,
                original_filename=orig_name,
                title=title,
                authors=authors,
                page_count=len(pages),
                total_pages_in_pdf=total_pages,
                upload_time=datetime.now(timezone.utc).isoformat(),
                file_size_bytes=file_size,
                has_extractable_text=has_extractable_text,
                sha256_hash=sha256_hash,
            )

            return Document(metadata=doc_metadata, pages=pages)

        finally:
            doc.close()

    def _extract_pages(self, doc: fitz.Document) -> List[Page]:
        """Extract text from each page preserving 1-indexed page numbering."""
        pages: List[Page] = []

        for page_idx in range(doc.page_count):
            page_num = page_idx + 1  # 1-indexed
            try:
                page = doc.load_page(page_idx)
                text = page.get_text("text") or ""
                cleaned_text = self._clean_page_text(text)

                # Skip completely blank pages if configured
                if self.ignore_empty_pages and not cleaned_text:
                    continue

                page_obj = Page(
                    page_number=page_num,
                    text=cleaned_text,
                    char_count=len(cleaned_text),
                    word_count=len(cleaned_text.split()) if cleaned_text else 0,
                )
                pages.append(page_obj)
            except Exception:
                # If a single page fails to extract, continue with remaining pages
                continue

        return pages

    def _clean_page_text(self, text: str) -> str:
        """Clean whitespace and standardize line breaks in extracted page text."""
        if not text:
            return ""
        # Normalize carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Remove null characters
        text = text.replace("\x00", "")
        # Remove excessive blank lines (> 2 consecutive newlines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_metadata(
        self,
        doc: fitz.Document,
        fallback_name: str,
    ) -> Tuple[str, List[str]]:
        """Detect title and authors from PDF metadata and page content heuristics.

        Falls back gracefully if metadata is missing or generic.
        """
        meta = doc.metadata or {}
        raw_title = (meta.get("title") or "").strip()
        raw_author = (meta.get("author") or "").strip()

        # Check if PDF metadata title is useful (not empty or generic template)
        generic_patterns = [
            r"^untitled",
            r"^microsoft word",
            r"^latex",
            r"^paper",
            r"^\.pdf$",
            r"^document",
            r"^scan",
        ]
        is_generic_title = not raw_title or any(
            re.search(pat, raw_title, re.IGNORECASE) for pat in generic_patterns
        )

        detected_title = ""
        if not is_generic_title:
            detected_title = raw_title
        else:
            # Try heuristic extraction from the first page text blocks
            detected_title = self._detect_title_from_first_page(doc)

        # Fallback to cleaned filename without extension if still empty
        if not detected_title:
            stem = Path(fallback_name).stem
            # Replace underscores and hyphens with spaces
            cleaned_stem = re.sub(r"[-_]+", " ", stem).strip()
            detected_title = cleaned_stem.title() if cleaned_stem else "Untitled Research Paper"

        # Detect Authors
        authors: List[str] = []
        if raw_author and not any(
            re.search(pat, raw_author, re.IGNORECASE) for pat in generic_patterns
        ):
            # Split by comma, semicolon, or 'and'
            parts = re.split(r"[,;]|\band\b", raw_author)
            authors = [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]

        if not authors:
            authors = self._detect_authors_from_first_page(doc, detected_title)

        return detected_title, authors

    def _detect_title_from_first_page(self, doc: fitz.Document) -> str:
        """Inspect the first page for the largest font size or leading block as the title."""
        if doc.page_count == 0:
            return ""

        try:
            page = doc.load_page(0)
            page_dict = page.get_text("dict")
            blocks = page_dict.get("blocks", [])

            candidates = []
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            size = span.get("size", 0.0)
                            flags = span.get("flags", 0)
                            if text and len(text) > 3:
                                candidates.append((size, flags, text))

            if candidates:
                # Find the maximum font size
                max_size = max(c[0] for c in candidates)
                # In research papers, paper titles are distinctly larger than body text (typically >= 13pt)
                if max_size >= 13.0:
                    # Group text spans that share the maximum font size or near max size (within 1pt)
                    title_spans = [
                        c[2] for c in candidates if c[0] >= max_size - 1.0 and len(c[2]) > 2
                    ]
                    # Join candidate spans
                    title_candidate = " ".join(title_spans)
                    # Clean up repeated whitespace
                    title_candidate = re.sub(r"\s+", " ", title_candidate).strip()
                    # If title candidate looks like a plausible title (e.g. 5 to 250 chars)
                    if 5 <= len(title_candidate) <= 250:
                        # Ignore if it starts with "Abstract" or "Contents"
                        if not re.match(r"^(abstract|table of contents|arxiv)", title_candidate, re.IGNORECASE):
                            return title_candidate
        except Exception:
            pass

        return ""

    def _detect_authors_from_first_page(self, doc: fitz.Document, title: str) -> List[str]:
        """Try to detect author line below the title on page 1."""
        if doc.page_count == 0:
            return []

        try:
            page = doc.load_page(0)
            text = page.get_text("text")
            lines = [line.strip() for line in text.splitlines() if line.strip()]

            # Look for lines between title and 'Abstract'
            abstract_idx = -1
            title_idx = -1
            for idx, line in enumerate(lines):
                if title and title[:20].lower() in line.lower():
                    title_idx = idx
                if re.match(r"^abstract\b", line, re.IGNORECASE):
                    abstract_idx = idx
                    break

            if 0 <= title_idx < abstract_idx:
                author_candidates = lines[title_idx + 1 : abstract_idx]
                authors = []
                for cand in author_candidates:
                    # Filter out affiliations or emails
                    if "@" not in cand and "university" not in cand.lower() and "department" not in cand.lower():
                        # Split by commas or 'and'
                        parts = re.split(r"[,;]|\band\b", cand)
                        for p in parts:
                            cleaned = re.sub(r"[*†‡§\d]", "", p).strip()
                            if 2 <= len(cleaned.split()) <= 4 and len(cleaned) < 40:
                                authors.append(cleaned)
                if authors:
                    return authors[:6]  # Return up to first 6 authors
        except Exception:
            pass

        return []
