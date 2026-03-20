import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.text.converter import (
    DocumentChunk,
    chunk_document_text,
    extract_text_from_pdf,
)


class TestExtractTextFromPdf:
    """Test PDF extraction function."""

    async def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="File not found"):
            await extract_text_from_pdf(Path("/nonexistent/file.pdf"))

    async def test_not_a_pdf(self, tmp_path):
        txt_file = tmp_path / "file.txt"
        txt_file.write_text("hello")
        with pytest.raises(ValueError, match="Not a PDF file"):
            await extract_text_from_pdf(txt_file)

    async def test_extracts_text(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy")
        with patch("app.text.converter.pymupdf4llm.to_markdown", return_value="# Hello\nWorld"):
            result = await extract_text_from_pdf(pdf_path)
            assert result == "# Hello\nWorld"


class TestChunkDocumentText:
    """Test generic fixed-size chunking."""

    def test_empty_text_returns_empty(self):
        assert chunk_document_text("") == []

    def test_whitespace_only_returns_empty(self):
        assert chunk_document_text("   \n\n   ") == []

    def test_single_paragraph(self):
        text = "This is a single paragraph with some text."
        chunks = chunk_document_text(text, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].section_id == "document-1"
        assert chunks[0].subpart == "document"

    def test_custom_source_name(self):
        text = "Some text content."
        chunks = chunk_document_text(text, source_name="rfc1035")
        assert len(chunks) == 1
        assert chunks[0].section_id == "rfc1035-1"
        assert chunks[0].title == "rfc1035 chunk 1"
        assert chunks[0].subpart == "rfc1035"

    def test_multiple_paragraphs_single_chunk(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_document_text(text, chunk_size=1000)
        assert len(chunks) == 1
        assert "Paragraph one." in chunks[0].text
        assert "Paragraph three." in chunks[0].text

    def test_splits_on_paragraph_boundary(self):
        # Create paragraphs that exceed chunk_size
        para1 = "A" * 100
        para2 = "B" * 100
        para3 = "C" * 100
        text = f"{para1}\n\n{para2}\n\n{para3}"
        chunks = chunk_document_text(text, chunk_size=150, chunk_overlap=0)
        # Should split into multiple chunks
        assert len(chunks) >= 2

    def test_overlap_between_chunks(self):
        para1 = "First paragraph content."
        para2 = "Second paragraph content."
        para3 = "Third paragraph content."
        text = f"{para1}\n\n{para2}\n\n{para3}"
        chunks = chunk_document_text(text, chunk_size=50, chunk_overlap=30)
        # With overlap, some content should appear in multiple chunks
        assert len(chunks) >= 2

    def test_large_paragraph_splits_by_sentences(self):
        # A single paragraph with many sentences that exceeds chunk_size
        sentences = [f"Sentence number {i} is here." for i in range(20)]
        text = " ".join(sentences)
        chunks = chunk_document_text(text, chunk_size=100, chunk_overlap=0)
        assert len(chunks) > 1
        # All content should be represented
        all_text = " ".join(c.text for c in chunks)
        assert "Sentence number 0" in all_text
        assert "Sentence number 19" in all_text

    def test_collapses_excessive_newlines(self):
        text = "Para one.\n\n\n\n\nPara two."
        chunks = chunk_document_text(text, chunk_size=1000)
        assert len(chunks) == 1
        # Should not contain 3+ consecutive newlines
        assert "\n\n\n" not in chunks[0].text

    def test_sequential_section_ids(self):
        paras = [f"Paragraph {i}." for i in range(5)]
        text = "\n\n".join(paras)
        chunks = chunk_document_text(text, chunk_size=30, chunk_overlap=0)
        ids = [c.section_id for c in chunks]
        # IDs should be sequential
        for i, sid in enumerate(ids, start=1):
            assert sid == f"document-{i}"

    def test_returns_document_chunk_dataclass(self):
        text = "Some content."
        chunks = chunk_document_text(text)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert isinstance(chunk, DocumentChunk)
        assert hasattr(chunk, "section_id")
        assert hasattr(chunk, "title")
        assert hasattr(chunk, "subpart")
        assert hasattr(chunk, "text")
