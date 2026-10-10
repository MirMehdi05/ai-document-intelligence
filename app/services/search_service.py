from sklearn.metrics.pairwise import cosine_similarity


def search_chunks(
    query_embedding: list[float],
    chunk_embeddings: list[list[float]],
    chunks: list[str],
    top_k: int = 3,
    min_score: float = 0.0
) -> list[dict]:

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if not chunks:
        return []

    if len(chunks) != len(chunk_embeddings):
        raise ValueError(
            "Number of chunks must match number of embeddings"
        )

    top_k = min(top_k, len(chunks))

    similarities = cosine_similarity(
        [query_embedding],
        chunk_embeddings
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:
        score = float(similarities[index])

        if score < min_score:
            continue

        results.append({
            "chunk": chunks[index],
            "score": score,
            "index": int(index)
        })

    return results