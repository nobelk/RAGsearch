import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf4llm


@dataclass
class DocumentChunk:
    section_id: str
    title: str
    subpart: str
    text: str


async def extract_text_from_pdf(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {file_path}")
    text = await asyncio.to_thread(pymupdf4llm.to_markdown, file_path)
    return text


# Default chunk size and overlap for generic chunking
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200


def chunk_document_text(
    markdown_text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    source_name: str = "document",
) -> list[DocumentChunk]:
    """Split markdown text into chunks using paragraph-aware fixed-size splitting.

    This is a generic chunker that works on any text format. It splits on
    paragraph boundaries (double newlines) and groups paragraphs into chunks
    of approximately `chunk_size` characters with `chunk_overlap` overlap.
    """
    if not markdown_text.strip():
        return []

    # Normalize whitespace: collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", markdown_text.strip())

    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if not paragraphs:
        return []

    chunks: list[DocumentChunk] = []
    current_text: list[str] = []
    current_length = 0
    chunk_index = 0

    for paragraph in paragraphs:
        para_len = len(paragraph)

        # If a single paragraph exceeds chunk_size, split it by sentences
        if para_len > chunk_size:
            # Flush current buffer first
            if current_text:
                chunk_index += 1
                chunks.append(
                    DocumentChunk(
                        section_id=f"{source_name}-{chunk_index}",
                        title=f"{source_name} chunk {chunk_index}",
                        subpart=source_name,
                        text="\n\n".join(current_text),
                    )
                )
                current_text = []
                current_length = 0

            # Split large paragraph into sentence-level sub-chunks
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            sub_text: list[str] = []
            sub_length = 0
            for sentence in sentences:
                if sub_length + len(sentence) > chunk_size and sub_text:
                    chunk_index += 1
                    chunks.append(
                        DocumentChunk(
                            section_id=f"{source_name}-{chunk_index}",
                            title=f"{source_name} chunk {chunk_index}",
                            subpart=source_name,
                            text=" ".join(sub_text),
                        )
                    )
                    # Keep overlap
                    overlap_text: list[str] = []
                    overlap_len = 0
                    for s in reversed(sub_text):
                        if overlap_len + len(s) > chunk_overlap:
                            break
                        overlap_text.insert(0, s)
                        overlap_len += len(s)
                    sub_text = overlap_text
                    sub_length = overlap_len

                sub_text.append(sentence)
                sub_length += len(sentence)

            if sub_text:
                chunk_index += 1
                chunks.append(
                    DocumentChunk(
                        section_id=f"{source_name}-{chunk_index}",
                        title=f"{source_name} chunk {chunk_index}",
                        subpart=source_name,
                        text=" ".join(sub_text),
                    )
                )
            continue

        # Would adding this paragraph exceed chunk_size?
        if current_length + para_len > chunk_size and current_text:
            chunk_index += 1
            chunks.append(
                DocumentChunk(
                    section_id=f"{source_name}-{chunk_index}",
                    title=f"{source_name} chunk {chunk_index}",
                    subpart=source_name,
                    text="\n\n".join(current_text),
                )
            )

            # Build overlap from tail of current_text
            overlap_parts: list[str] = []
            overlap_len = 0
            for part in reversed(current_text):
                if overlap_len + len(part) > chunk_overlap:
                    break
                overlap_parts.insert(0, part)
                overlap_len += len(part)
            current_text = overlap_parts
            current_length = overlap_len

        current_text.append(paragraph)
        current_length += para_len

    # Flush remaining text
    if current_text:
        chunk_index += 1
        chunks.append(
            DocumentChunk(
                section_id=f"{source_name}-{chunk_index}",
                title=f"{source_name} chunk {chunk_index}",
                subpart=source_name,
                text="\n\n".join(current_text),
            )
        )

    return chunks
