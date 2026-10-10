from app.services.chunking_service import chunk_text


text = """
Artificial intelligence is transforming healthcare.
Machine learning can help doctors identify diseases.
Deep learning can analyze medical images.
Natural language processing can understand medical documents.
AI systems can assist professionals by retrieving relevant information quickly.
"""


chunks = chunk_text(
    text,
    chunk_size=100,
    overlap=20
)


print("=" * 60)
print("CHUNKING TEST")
print("=" * 60)

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)