import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)

    def add_embeddings(
        self,
        embeddings: list[list[float]]
    ):
        vectors = np.array(
            embeddings,
            dtype="float32"
        )

        self.index.add(vectors)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3
    ):
        query_vector = np.array(
            [query_embedding],
            dtype="float32"
        )

        distances, indices = self.index.search(
            query_vector,
            top_k
        )

        return distances[0], indices[0]