from functools import lru_cache
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    """Load each embedding model once per app process to reduce memory usage."""
    return SentenceTransformer(model_name)


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = _load_model(model_name)

    def generate_embedding(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()
