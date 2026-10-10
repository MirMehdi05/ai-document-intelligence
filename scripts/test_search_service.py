from app.services.embedding_service import EmbeddingService
from app.services.chunking_service import chunk_text
from app.services.document_service import extract_text_from_pdf
from app.services.search_service import search_chunks


PDF_PATH = "data/raw/sample.pdf"

embedding_service = EmbeddingService()

text = extract_text_from_pdf(PDF_PATH)

chunks = chunk_text(
    text,
    chunk_size=500,
    overlap=50
)

chunk_embeddings = embedding_service.generate_embeddings(chunks)

query = "How does Python help in artificial intelligence?"

query_embedding = embedding_service.generate_embedding(query)

results = search_chunks(
    query_embedding=query_embedding,
    chunk_embeddings=chunk_embeddings,
    chunks=chunks,
    top_k=3
)

print("=" * 60)
print("SEARCH SERVICE TEST")
print("=" * 60)

print(f"\nQuery: {query}")

for rank, result in enumerate(results, start=1):

    print(f"\nRESULT {rank}")
    print("-" * 60)

    print(f"Score: {result['score']:.4f}")
    print(f"Chunk index: {result['index']}")

    print("\nText:")
    print(result["chunk"])