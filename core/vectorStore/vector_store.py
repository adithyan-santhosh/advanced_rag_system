import faiss
import numpy as np


class FAISSVectorStore:
    def __init__(self, dimension: int):
        """
        Cosine similarity using Inner Product + normalization
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.texts = []

    def add_embeddings(self, embeddings, texts):
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query_embedding, top_k=3):
        query_embedding = np.array([query_embedding]).astype("float32")

        # Normalize query
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "score": float(scores[0][i])
                })

        return results
