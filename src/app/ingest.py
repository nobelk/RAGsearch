import asyncio
import sys
from pathlib import Path

from app.text.converter import chunk_document_text, extract_text_from_pdf
from app.vectorstore import ensure_collection, upsert_chunks


async def ingest_pdf(pdf_path: Path) -> int:
    """Extract, chunk, and upsert a single PDF. Returns chunk count."""
    print(f"Processing {pdf_path.name}...")
    text = await extract_text_from_pdf(pdf_path)
    chunks = chunk_document_text(text, source_name=pdf_path.stem)
    if not chunks:
        print(f"  No chunks extracted from {pdf_path.name}, skipping.")
        return 0
    await upsert_chunks(chunks)
    print(f"  Upserted {len(chunks)} chunks from {pdf_path.name}.")
    return len(chunks)


async def ingest_directory(data_dir: Path) -> None:
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF file(s) in {data_dir}")
    await ensure_collection()

    total = 0
    errors = 0
    for pdf_path in pdf_files:
        try:
            total += await ingest_pdf(pdf_path)
        except Exception as exc:
            print(f"  ERROR processing {pdf_path.name}: {exc}")
            errors += 1

    print(f"Done. {total} total chunks upserted from {len(pdf_files)} file(s).")
    if errors:
        print(f"WARNING: {errors} file(s) failed to process.")
        sys.exit(1)


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    if not data_dir.is_dir():
        print(f"Error: directory not found: {data_dir}")
        sys.exit(1)
    asyncio.run(ingest_directory(data_dir))
