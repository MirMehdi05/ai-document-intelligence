from app.services.embedding_service import EmbeddingService
from app.services.chunking_service import chunk_text
from app.services.document_service import extract_text_from_pdf
from sklearn.metrics.pairwise import cosine_similarity


PDF_PATH = "data/raw/sample.pdf"

embedding_service = EmbeddingService()

# 1. Extract document text
text = extract_text_from_pdf(PDF_PATH)

# 2. Split document into chunks
chunks = chunk_text(
    text,
    chunk_size=500,
    overlap=50
)

# 3. Generate embeddings for all chunks
chunk_embeddings = embedding_service.generate_embeddings(chunks)

# 4. User's search query
query = "How does Python help in artificial intelligence?"

# 5. Generate query embedding
query_embedding = embedding_service.generate_embedding(query)

# 6. Calculate similarity with every chunk
similarities = cosine_similarity(
    [query_embedding],
    chunk_embeddings
)[0]

# 7. Rank chunks by similarity
ranked_indices = similarities.argsort()[::-1]

# 8. Retrieve top 3 chunks
top_k = 3

print("=" * 60)
print("TOP-K SEMANTIC SEARCH")
print("=" * 60)

print(f"\nQuery:")
print(query)

print("\n" + "-" * 60)

for rank, index in enumerate(ranked_indices[:top_k], start=1):

    print(f"\nRESULT {rank}")
    print("-" * 60)

    print(f"Similarity score: {similarities[index]:.4f}")

    print("\nChunk:")
    print(chunks[index])