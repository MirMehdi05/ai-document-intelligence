from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStore


embedding_service = EmbeddingService()

texts = [
    "Python is used for artificial intelligence.",
    "Machine learning models can be trained using Python.",
    "The weather is cold today."
]

embeddings = embedding_service.generate_embeddings(texts)

dimension = len(embeddings[0])

vector_store = VectorStore(dimension)

vector_store.add_embeddings(embeddings)

query = "How is Python used in AI?"

query_embedding = embedding_service.generate_embedding(query)

distances, indices = vector_store.search(
    query_embedding,
    top_k=2
)

print("=" * 60)
print("VECTOR STORE TEST")
print("=" * 60)

print(f"\nQuery: {query}")

for rank, index in enumerate(indices, start=1):

    print(f"\nRESULT {rank}")
    print("-" * 60)

    print(f"Index: {index}")
    print(f"Distance: {distances[rank - 1]:.4f}")
    print(f"Text: {texts[index]}")