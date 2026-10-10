from app.services.embedding_service import EmbeddingService
from sklearn.metrics.pairwise import cosine_similarity


embedding_service = EmbeddingService()


sentence_a = "Python is commonly used for artificial intelligence development."

sentence_b = "Developers use Python to build machine learning applications."

sentence_c = "The weather is very cold today."


# Generate embeddings
embedding_a = embedding_service.generate_embedding(sentence_a)
embedding_b = embedding_service.generate_embedding(sentence_b)
embedding_c = embedding_service.generate_embedding(sentence_c)


# Calculate similarity
similarity_ab = cosine_similarity(
    [embedding_a],
    [embedding_b]
)[0][0]

similarity_ac = cosine_similarity(
    [embedding_a],
    [embedding_c]
)[0][0]


print("=" * 60)
print("SEMANTIC SIMILARITY TEST")
print("=" * 60)

print("\nSentence A:")
print(sentence_a)

print("\nSentence B:")
print(sentence_b)

print("\nSentence C:")
print(sentence_c)

print("\n" + "-" * 60)

print(f"\nSimilarity A ↔ B: {similarity_ab:.4f}")
print(f"Similarity A ↔ C: {similarity_ac:.4f}")