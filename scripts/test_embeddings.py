from app.services.embedding_service import EmbeddingService


embedding_service = EmbeddingService()


text = "Python is used to build artificial intelligence applications."

embedding = embedding_service.generate_embedding(text)


print("=" * 60)
print("EMBEDDING TEST")
print("=" * 60)

print(f"Original text: {text}")
print(f"Embedding dimensions: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")