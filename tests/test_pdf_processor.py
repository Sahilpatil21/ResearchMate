"""Unit tests for ResearchMate PDF Processing and Storage.

Tests cover:
- PDF text extraction
- Blank page handling and skipping
- Corrupted/Invalid PDF handling
- Metadata extraction & fallback
- Unique document ID generation
- File storage and retrieval round-trips
"""

import os
from pathlib import Path
import pytest
import fitz  # PyMuPDF

from src.ingestion.pdf_processor import PDFProcessor, PDFProcessingError
from src.models.document import Document, DocumentMetadata, Page
from src.utils.file_utils import (
    calculate_sha256,
    delete_document,
    get_unique_filepath,
    load_all_processed_documents,
    load_processed_document,
    sanitize_filename,
    save_processed_document,
    save_uploaded_pdf,
)


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Create a sample 3-page PDF with known title, author, and text."""
    doc = fitz.open()
    
    # Page 1: Title in large font, author, and intro
    page1 = doc.new_page()
    page1.insert_text(fitz.Point(50, 72), "Deep Residual Learning for Image Recognition", fontsize=20)
    page1.insert_text(fitz.Point(50, 110), "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun", fontsize=11)
    page1.insert_text(fitz.Point(50, 140), "Abstract\nDeeper neural networks are more difficult to train.", fontsize=10)
    
    # Page 2: Methods section
    page2 = doc.new_page()
    page2.insert_text(fitz.Point(50, 72), "Methodology and Architecture Details.", fontsize=12)
    page2.insert_text(fitz.Point(50, 100), "We formulate the residual mapping as F(x) + x.", fontsize=10)

    # Page 3: Results
    page3 = doc.new_page()
    page3.insert_text(fitz.Point(50, 72), "Experimental Results and Conclusion.", fontsize=12)
    page3.insert_text(fitz.Point(50, 100), "The residual networks achieve top accuracy on ImageNet.", fontsize=10)

    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


@pytest.fixture
def pdf_with_blank_pages_bytes() -> bytes:
    """Create a 4-page PDF where page 2 and page 4 have no text."""
    doc = fitz.open()
    
    p1 = doc.new_page()
    p1.insert_text(fitz.Point(50, 72), "Attention Mechanisms in NLP", fontsize=18)
    
    # p2 is blank
    doc.new_page()

    p3 = doc.new_page()
    p3.insert_text(fitz.Point(50, 72), "Self-attention replaces recurrence entirely.", fontsize=11)

    # p4 is blank
    doc.new_page()

    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_process_valid_pdf(sample_pdf_bytes):
    """Test extracting structured content from a valid multi-page PDF."""
    processor = PDFProcessor()
    doc = processor.process_file(sample_pdf_bytes, filename="resnet_paper.pdf")

    assert isinstance(doc, Document)
    assert doc.document_id.startswith("doc_")
    assert doc.metadata.original_filename == "resnet_paper.pdf"
    assert doc.metadata.total_pages_in_pdf == 3
    assert doc.metadata.page_count == 3
    assert doc.metadata.has_extractable_text is True
    assert "Residual Learning" in doc.metadata.title

    # Test pages
    assert len(doc.pages) == 3
    assert doc.pages[0].page_number == 1
    assert "Deeper neural networks" in doc.pages[0].text
    assert doc.pages[1].page_number == 2
    assert "residual mapping" in doc.pages[1].text
    assert doc.pages[2].page_number == 3
    assert "Experimental Results" in doc.pages[2].text


def test_skip_blank_pages(pdf_with_blank_pages_bytes):
    """Test that blank pages are ignored while preserving original page numbers."""
    processor = PDFProcessor(ignore_empty_pages=True)
    doc = processor.process_file(pdf_with_blank_pages_bytes, filename="blank_test.pdf")

    assert doc.metadata.total_pages_in_pdf == 4
    # Only pages 1 and 3 have text
    assert len(doc.pages) == 2
    assert doc.pages[0].page_number == 1
    assert doc.pages[1].page_number == 3
    assert "Attention Mechanisms" in doc.pages[0].text
    assert "Self-attention" in doc.pages[1].text


def test_metadata_fallback_to_filename():
    """Test that missing PDF metadata and title fall back cleanly to cleaned filename."""
    # Create empty text page
    doc_fitz = fitz.open()
    p = doc_fitz.new_page()
    p.insert_text(fitz.Point(50, 72), "Just some generic text without title header.", fontsize=10)
    pdf_bytes = doc_fitz.write()
    doc_fitz.close()

    processor = PDFProcessor()
    doc = processor.process_file(pdf_bytes, filename="quantum_computing_advances_2026.pdf")

    assert doc.metadata.original_filename == "quantum_computing_advances_2026.pdf"
    assert "Quantum Computing Advances 2026" in doc.metadata.title


def test_unique_document_ids(sample_pdf_bytes):
    """Test that each processing run generates a unique document_id."""
    processor = PDFProcessor()
    doc1 = processor.process_file(sample_pdf_bytes, filename="paper.pdf")
    doc2 = processor.process_file(sample_pdf_bytes, filename="paper.pdf")

    assert doc1.document_id != doc2.document_id
    assert doc1.metadata.sha256_hash == doc2.metadata.sha256_hash


def test_corrupted_pdf_handling():
    """Test that corrupted bytes raise PDFProcessingError gracefully."""
    corrupted_bytes = b"%PDF-1.4\nSome corrupted garbage content that cannot be parsed as a PDF"
    processor = PDFProcessor()

    with pytest.raises(PDFProcessingError) as exc_info:
        processor.process_file(corrupted_bytes, filename="broken.pdf")
    assert "Corrupted or invalid PDF" in str(exc_info.value) or "Failed to parse" in str(exc_info.value)


def test_empty_bytes_handling():
    """Test that empty bytes raise PDFProcessingError gracefully."""
    processor = PDFProcessor()
    with pytest.raises(PDFProcessingError) as exc_info:
        processor.process_file(b"", filename="empty.pdf")
    assert "empty" in str(exc_info.value).lower()


def test_document_model_serialization(sample_pdf_bytes):
    """Test serialization to dict and JSON round-trip."""
    processor = PDFProcessor()
    doc = processor.process_file(sample_pdf_bytes, filename="resnet.pdf")

    # JSON round trip
    json_str = doc.to_json()
    reconstructed = Document.from_json(json_str)

    assert reconstructed.document_id == doc.document_id
    assert reconstructed.title == doc.title
    assert len(reconstructed.pages) == len(doc.pages)
    assert reconstructed.pages[0].text == doc.pages[0].text
    assert reconstructed.total_words == doc.total_words


def test_file_utils_operations(tmp_path, sample_pdf_bytes):
    """Test file storage, unique naming, loading, and deletion."""
    papers_dir = tmp_path / "papers"
    processed_dir = tmp_path / "processed"

    # Test safe saving with duplicate filename
    path1, name1 = save_uploaded_pdf(sample_pdf_bytes, "paper.pdf", dest_dir=papers_dir)
    assert path1.exists()
    assert name1 == "paper.pdf"

    path2, name2 = save_uploaded_pdf(sample_pdf_bytes, "paper.pdf", dest_dir=papers_dir)
    assert path2.exists()
    assert name2 == "paper_1.pdf"

    # Process and save JSON
    processor = PDFProcessor()
    doc = processor.process_file(sample_pdf_bytes, filename=name1, storage_filename=name1)
    json_path = save_processed_document(doc, dest_dir=processed_dir)
    assert json_path.exists()

    # Load single document
    loaded_doc = load_processed_document(doc.document_id, dest_dir=processed_dir)
    assert loaded_doc is not None
    assert loaded_doc.document_id == doc.document_id
    assert loaded_doc.metadata.title == doc.metadata.title

    # Load all documents
    all_docs = load_all_processed_documents(dest_dir=processed_dir)
    assert len(all_docs) == 1
    assert all_docs[0].document_id == doc.document_id

    # Delete document
    deleted = delete_document(doc.document_id, papers_dir=papers_dir, processed_dir=processed_dir)
    assert deleted is True
    assert not json_path.exists()
    assert not path1.exists()
