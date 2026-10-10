from pathlib import Path

from app.services.document_service import extract_text_from_pdf
from app.services.chunking_service import chunk_text
from app.models.document_chunk import DocumentChunk


PDF_PATH = "data/raw/sample.pdf"


# Step 1: Extract text from PDF
text = extract_text_from_pdf(PDF_PATH)

print("=" * 60)
print("DOCUMENT PROCESSING PIPELINE")
print("=" * 60)

print(f"\nExtracted characters: {len(text)}")


# Step 2: Split text into chunks
chunks = chunk_text(
    text,
    chunk_size=500,
    overlap=50
)

print(f"Total chunks: {len(chunks)}")


# Step 3: Add metadata to every chunk
document_chunks = []

source_file = Path(PDF_PATH).name

for index, chunk in enumerate(chunks):
    document_chunk = DocumentChunk(
        chunk_id=f"{source_file}-{index}",
        text=chunk,
        source_file=source_file,
        chunk_index=index
    )

    document_chunks.append(document_chunk)


# Step 4: Display the chunks and metadata
for chunk in document_chunks:
    print(f"\n--- Chunk {chunk.chunk_index} ---")
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Source: {chunk.source_file}")
    print(f"Text: {chunk.text}")